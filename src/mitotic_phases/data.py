"""Metadata analysis, stratified splitting, datasets, transforms, and DataLoaders."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.transforms import InterpolationMode, v2

from mitotic_phases.config import DataConfig, LabelMode
from mitotic_phases.reproducibility import make_generator, seed_worker

logger = logging.getLogger(__name__)

IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)
REQUIRED_COLUMNS: tuple[str, ...] = (
    "dataset",
    "uid",
    "expert1_label",
    "expert2_label",
    "majority_atypical",
)


@dataclass(frozen=True)
class DatasetSummary:
    """Structured summary statistics of expert annotations in AMI-Br metadata."""

    total_samples: int
    phase_agreement_rate: float
    atypical_consensus_count: int | None
    atypical_majority_count: int
    atypical_disagreement_count: int | None
    strictly_normal_count: int | None
    class_distribution: dict[str, float]
    top_disagreements: tuple[tuple[str, str, int], ...]


@dataclass(frozen=True)
class SplitDataFrames:
    """Stratified train/val/test DataFrames with a shared class vocabulary."""

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    classes: tuple[str, ...]
    class_to_idx: dict[str, int]


@dataclass(frozen=True)
class DataBundle:
    """Complete data pipeline output: DataLoaders, split tables, and transforms."""

    loaders: dict[str, DataLoader[tuple[torch.Tensor, torch.Tensor]]]
    splits: SplitDataFrames
    transforms: dict[str, v2.Compose]


def _validate_columns(df: pd.DataFrame) -> None:
    """Ensure required annotation columns exist in the metadata DataFrame."""
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Metadata DataFrame is missing required columns: {missing}")


def summarize_metadata(df: pd.DataFrame, top_k_confusions: int = 5) -> DatasetSummary:
    """Compute inter-expert agreement and class distribution statistics.

    Args:
        df: Raw AMI-Br metadata DataFrame.
        top_k_confusions: Number of most frequent expert disagreement pairs to return.

    Returns:
        Structured DatasetSummary without plotting side effects.
    """
    _validate_columns(df)
    total = len(df)
    if total == 0:
        raise ValueError("Cannot summarize an empty metadata DataFrame.")

    phase_agreement = df["expert1_label"] == df["expert2_label"]
    agreement_rate = float(phase_agreement.mean())

    atypical_cols = ["expert1_atypical", "expert2_atypical", "expert3_atypical"]
    if all(col in df.columns for col in atypical_cols):
        votes = df[atypical_cols].astype(int).sum(axis=1)
        atypical_consensus: int | None = int((votes == 3).sum())
        atypical_disagreement: int | None = int(((votes > 0) & (votes < 3)).sum())
        strictly_normal: int | None = int((votes == 0).sum())
    else:
        atypical_consensus = None
        atypical_disagreement = None
        strictly_normal = None

    atypical_majority = int(df["majority_atypical"].astype(bool).sum())

    disagreements = df.loc[~phase_agreement]
    top_pairs: list[tuple[str, str, int]] = []
    if not disagreements.empty:
        grouped = (
            disagreements.groupby(["expert1_label", "expert2_label"])
            .size()
            .sort_values(ascending=False)
            .head(top_k_confusions)
        )
        for pair, count in grouped.items():
            if isinstance(pair, tuple) and len(pair) == 2:
                top_pairs.append((str(pair[0]), str(pair[1]), int(count)))

    raw_dist = df["expert1_label"].value_counts(normalize=True)
    class_distribution = {str(label): float(freq) for label, freq in raw_dist.items()}

    return DatasetSummary(
        total_samples=total,
        phase_agreement_rate=agreement_rate,
        atypical_consensus_count=atypical_consensus,
        atypical_majority_count=atypical_majority,
        atypical_disagreement_count=atypical_disagreement,
        strictly_normal_count=strictly_normal,
        class_distribution=class_distribution,
        top_disagreements=tuple(top_pairs),
    )


def filter_metadata(df: pd.DataFrame, config: DataConfig) -> pd.DataFrame:
    """Filter AMI-Br metadata by mitosis mode, prefix consistency, and minimum support.

    Args:
        df: Raw metadata DataFrame.
        config: Immutable data configuration.

    Returns:
        Filtered DataFrame with an added ``expert_group`` column and reset index.
    """
    _validate_columns(df)
    is_atypical = config.mode == "AMF"
    filtered = df.loc[df["majority_atypical"].astype(bool) == is_atypical].copy()

    prefix = config.mode
    filtered = filtered.loc[
        filtered["expert1_label"].astype(str).str.startswith(prefix)
        & filtered["expert2_label"].astype(str).str.startswith(prefix)
    ].copy()

    counts = filtered["expert1_label"].value_counts()
    valid_classes = set(counts[counts >= config.min_class_count].index.astype(str))
    dropped = sorted(set(counts.index.astype(str)) - valid_classes)
    if dropped:
        logger.warning(
            "Dropped classes below min_class_count=%d: %s",
            config.min_class_count,
            dropped,
        )

    filtered = filtered.loc[
        filtered["expert1_label"].isin(valid_classes)
        & filtered["expert2_label"].isin(valid_classes)
    ].copy()

    if filtered.empty:
        raise ValueError(
            f"No samples remaining after filtering for mode={config.mode!r} "
            f"with min_class_count={config.min_class_count}."
        )

    filtered["expert_group"] = np.where(
        filtered["expert1_label"] == filtered["expert2_label"],
        "Group A (Consensus)",
        "Group B (Disagreement)",
    )
    return filtered.reset_index(drop=True)


def build_class_mapping(df: pd.DataFrame) -> tuple[tuple[str, ...], dict[str, int]]:
    """Build a single deterministic class vocabulary from Expert 1 and Expert 2 labels.

    Args:
        df: Filtered metadata DataFrame.

    Returns:
        Tuple of ``(sorted_class_names, class_to_idx)``.
    """
    unique_labels = set(df["expert1_label"].astype(str).unique()) | set(
        df["expert2_label"].astype(str).unique()
    )
    if not unique_labels:
        raise ValueError("No class labels found in metadata DataFrame.")
    classes = tuple(sorted(unique_labels))
    class_to_idx = {name: idx for idx, name in enumerate(classes)}
    return classes, class_to_idx


def encode_soft_label(label1_idx: int, label2_idx: int, num_classes: int) -> torch.Tensor:
    """Construct a bimodal probabilistic target vector from two expert votes.

    Each expert contributes 0.5 probability mass to their chosen class. When both
    experts agree, this reduces to a standard one-hot vector.

    Args:
        label1_idx: Class index annotated by Expert 1.
        label2_idx: Class index annotated by Expert 2.
        num_classes: Total number of classes.

    Returns:
        Float32 tensor of shape ``(num_classes,)`` summing to 1.0.
    """
    if num_classes < 2:
        raise ValueError(f"num_classes must be >= 2, got {num_classes}")
    if not (0 <= label1_idx < num_classes) or not (0 <= label2_idx < num_classes):
        raise ValueError(
            f"Label indices ({label1_idx}, {label2_idx}) out of range [0, {num_classes})."
        )
    target = torch.zeros(num_classes, dtype=torch.float32)
    target[label1_idx] += 0.5
    target[label2_idx] += 0.5
    return target


def _can_stratify(labels: pd.Series, test_size: float) -> bool:
    """Return True if sklearn train_test_split can stratify on ``labels`` safely."""
    n_samples = len(labels)
    if n_samples < 2:
        return False
    counts = labels.value_counts()
    if int(counts.min()) < 2:
        return False
    n_classes = len(counts)
    n_test = int(round(n_samples * test_size))
    n_train = n_samples - n_test
    return n_test >= n_classes and n_train >= n_classes


def split_metadata(
    df: pd.DataFrame,
    config: DataConfig,
    seed: int,
) -> SplitDataFrames:
    """Perform stratified train/val/test splitting preserving soft-label distributions.

    Args:
        df: Filtered metadata DataFrame.
        config: Immutable data configuration.
        seed: Random state for reproducible splitting.

    Returns:
        SplitDataFrames containing train, val, and test DataFrames plus shared vocabulary.
    """
    classes, class_to_idx = build_class_mapping(df)
    work_df = df.copy()

    idx1 = work_df["expert1_label"].map(class_to_idx).astype(int)
    idx2 = work_df["expert2_label"].map(class_to_idx).astype(int)
    low = np.minimum(idx1.to_numpy(), idx2.to_numpy())
    high = np.maximum(idx1.to_numpy(), idx2.to_numpy())
    stratify_key = pd.Series(
        [f"{a}-{b}" for a, b in zip(low, high, strict=True)],
        index=work_df.index,
        dtype="object",
    )

    key_counts = stratify_key.value_counts()
    rare_keys = set(key_counts[key_counts < 3].index)
    if rare_keys:
        rare_mask = stratify_key.isin(rare_keys)
        stratify_key.loc[rare_mask] = work_df.loc[rare_mask, "expert1_label"].astype(str)

    temp_ratio = config.val_ratio + config.test_ratio
    strat_first: pd.Series | None = None
    if _can_stratify(stratify_key, temp_ratio):
        strat_first = stratify_key
    elif _can_stratify(work_df["expert1_label"], temp_ratio):
        logger.warning("Falling back to Expert 1 stratification for train/temp split.")
        strat_first = work_df["expert1_label"]
    else:
        logger.warning("Dataset too small for stratified train/temp split; using unstratified.")

    train_df, temp_df = train_test_split(
        work_df,
        test_size=temp_ratio,
        stratify=strat_first,
        random_state=seed,
    )

    test_share_of_temp = config.test_ratio / temp_ratio
    strat_second: pd.Series | None = None
    if _can_stratify(temp_df["expert1_label"], test_share_of_temp):
        strat_second = temp_df["expert1_label"]
    else:
        logger.warning("Temp split too small for stratified val/test split; using unstratified.")

    val_df, test_df = train_test_split(
        temp_df,
        test_size=test_share_of_temp,
        stratify=strat_second,
        random_state=seed,
    )

    logger.info(
        "Dataset split complete: train=%d, val=%d, test=%d (%d classes).",
        len(train_df),
        len(val_df),
        len(test_df),
        len(classes),
    )
    return SplitDataFrames(
        train=train_df.reset_index(drop=True),
        val=val_df.reset_index(drop=True),
        test=test_df.reset_index(drop=True),
        classes=classes,
        class_to_idx=class_to_idx,
    )


class _AdditiveGaussianNoise:
    """Fallback additive Gaussian noise transform for torchvision < 0.18 compatibility."""

    def __init__(self, mean: float = 0.0, sigma: float = 0.01) -> None:
        self.mean = mean
        self.sigma = sigma

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.sigma <= 0.0:
            return tensor
        noise = torch.randn_like(tensor) * self.sigma + self.mean
        return (tensor + noise).clamp(0.0, 1.0)


class MacenkoAugment:
    """Stochastic multi-reference Macenko H&E stain normalization augmentation.

    Uses PyTorch's RNG for DataLoader multi-worker safety and falls back gracefully
    when stain deconvolution encounters degenerate/blank image patches.
    """

    def __init__(self, ref_image_paths: Sequence[str | Path], p: float = 0.5) -> None:
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"p must lie in [0, 1], got {p}")
        self.p = p

        try:
            import cv2
            import torchstain
        except ImportError as exc:
            raise ImportError(
                "MacenkoAugment requires optional dependencies. "
                "Install them via: pip install -e '.[full]'"
            ) from exc

        self._normalizers: list[Any] = []
        for raw_path in ref_image_paths:
            path = Path(raw_path)
            if not path.is_file():
                logger.warning("Macenko reference image not found, skipping: %s", path)
                continue
            bgr = cv2.imread(str(path))
            if bgr is None:
                logger.warning("Failed to decode Macenko reference image: %s", path)
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            normalizer = torchstain.normalizers.MacenkoNormalizer(backend="numpy")
            normalizer.fit(rgb)
            self._normalizers.append(normalizer)

        if not self._normalizers:
            raise ValueError(
                "MacenkoAugment requires at least one valid, readable reference image."
            )

    def __call__(self, img: Image.Image) -> Image.Image:
        if self.p <= 0.0 or float(torch.rand(1).item()) >= self.p:
            return img

        idx = int(torch.randint(0, len(self._normalizers), (1,)).item())
        normalizer = self._normalizers[idx]
        img_np = np.asarray(img)
        try:
            normalized, _, _ = normalizer.normalize(I=img_np, stains=True)
            return Image.fromarray(np.asarray(normalized, dtype=np.uint8))
        except Exception as exc:
            logger.debug("Macenko normalization failed on patch; returning original: %s", exc)
            return img


def build_transforms(
    config: DataConfig,
    ref_paths: Sequence[str | Path] | None = None,
) -> dict[str, v2.Compose]:
    """Build training and evaluation torchvision v2 transformation pipelines.

    Args:
        config: Immutable data configuration.
        ref_paths: Optional reference image paths used when ``config.use_macenko`` is True.

    Returns:
        Dictionary with ``'train'`` and ``'val'`` composed transform pipelines.
    """
    train_ops: list[Any] = []

    if config.use_macenko:
        if not ref_paths:
            raise ValueError("use_macenko=True requires non-empty ref_paths.")
        train_ops.append(MacenkoAugment(ref_paths, p=config.macenko_prob))

    gaussian_noise: Any
    if hasattr(v2, "GaussianNoise"):
        gaussian_noise = v2.GaussianNoise(mean=0.0, sigma=0.01)
    else:
        gaussian_noise = _AdditiveGaussianNoise(mean=0.0, sigma=0.01)

    train_ops.extend(
        [
            v2.ToImage(),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            v2.RandomRotation(degrees=180, fill=255),
            v2.RandomResizedCrop(
                size=(224, 224),
                scale=(0.8, 1.0),
                ratio=(0.95, 1.05),
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            ),
            v2.ElasticTransform(alpha=15.0, sigma=3.0, fill=255),
            v2.ToDtype(torch.float32, scale=True),
            gaussian_noise,
            v2.Normalize(mean=list(IMAGENET_MEAN), std=list(IMAGENET_STD)),
        ]
    )

    val_ops: list[Any] = [
        v2.ToImage(),
        v2.Resize(size=(224, 224), interpolation=InterpolationMode.BILINEAR, antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=list(IMAGENET_MEAN), std=list(IMAGENET_STD)),
    ]

    return {
        "train": v2.Compose(train_ops),
        "val": v2.Compose(val_ops),
    }


class MitosisDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """PyTorch Dataset supporting both probabilistic soft labels and hard labels."""

    def __init__(
        self,
        data: pd.DataFrame,
        root_dir: str | Path,
        class_to_idx: dict[str, int],
        label_mode: LabelMode = "soft",
        transform: Callable[[Image.Image], torch.Tensor] | None = None,
    ) -> None:
        if not class_to_idx:
            raise ValueError("class_to_idx must be a non-empty mapping.")
        if label_mode not in ("soft", "hard"):
            raise ValueError(f"label_mode must be 'soft' or 'hard', got {label_mode!r}")

        self.data = data.reset_index(drop=True)
        self.root_dir = Path(root_dir)
        self.class_to_idx = dict(class_to_idx)
        self.classes: tuple[str, ...] = tuple(
            k for k, _ in sorted(self.class_to_idx.items(), key=lambda item: item[1])
        )
        self.num_classes = len(self.classes)
        self.label_mode: LabelMode = label_mode
        self.transform = transform
        self._fallback_to_tensor = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])

        exp1_series = self.data["expert1_label"].astype(str)
        exp2_series = self.data["expert2_label"].astype(str)
        unknown_labels = (set(exp1_series.unique()) | set(exp2_series.unique())) - set(
            self.class_to_idx.keys()
        )
        if unknown_labels:
            raise ValueError(
                f"Dataset contains labels missing from class_to_idx: {sorted(unknown_labels)}"
            )

        self.labels1: npt.NDArray[np.int64] = exp1_series.map(self.class_to_idx).to_numpy(
            dtype=np.int64
        )
        self.labels2: npt.NDArray[np.int64] = exp2_series.map(self.class_to_idx).to_numpy(
            dtype=np.int64
        )

    def __len__(self) -> int:
        return len(self.data)

    def get_image_path(self, idx: int) -> Path:
        """Resolve the patch image filepath for sample index ``idx``."""
        row = self.data.iloc[idx]
        return self.root_dir / f"{row['dataset']}_{row['uid']}.png"

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_path = self.get_image_path(idx)
        with Image.open(img_path) as handle:
            image = handle.convert("RGB")

        tensor_img: torch.Tensor = (
            self.transform(image) if self.transform is not None else self._fallback_to_tensor(image)
        )

        l1 = int(self.labels1[idx])
        if self.label_mode == "hard":
            target = torch.tensor(l1, dtype=torch.long)
        else:
            l2 = int(self.labels2[idx])
            target = encode_soft_label(l1, l2, self.num_classes)

        return tensor_img, target


def _resolve_macenko_refs(
    train_df: pd.DataFrame,
    config: DataConfig,
    seed: int,
) -> list[Path] | None:
    """Resolve reference patch paths for Macenko augmentation from train split only."""
    if not config.use_macenko:
        return None
    if config.macenko_ref_paths:
        return [Path(p) for p in config.macenko_ref_paths]

    n_refs = min(config.macenko_num_refs, len(train_df))
    sampled = train_df.sample(n=n_refs, random_state=seed)
    root = Path(config.patches_dir)
    return [root / f"{row['dataset']}_{row['uid']}.png" for _, row in sampled.iterrows()]


def prepare_dataloaders(
    df: pd.DataFrame,
    config: DataConfig,
    seed: int,
    device: torch.device | None = None,
) -> DataBundle:
    """Orchestrate filtering, splitting, transforms, datasets, and deterministic loaders.

    Args:
        df: Raw AMI-Br metadata DataFrame.
        config: Immutable data configuration.
        seed: Global seed for splitting, sampling, and DataLoader worker seeding.
        device: Optional target device used to enable ``pin_memory`` on CUDA only.

    Returns:
        DataBundle containing train/val/test loaders, splits, and transforms.
    """
    filtered = filter_metadata(df, config)
    splits = split_metadata(filtered, config, seed=seed)

    ref_paths = _resolve_macenko_refs(splits.train, config, seed=seed)
    data_transforms = build_transforms(config, ref_paths=ref_paths)

    train_dataset = MitosisDataset(
        data=splits.train,
        root_dir=config.patches_dir,
        class_to_idx=splits.class_to_idx,
        label_mode=config.label_mode,
        transform=data_transforms["train"],
    )
    val_dataset = MitosisDataset(
        data=splits.val,
        root_dir=config.patches_dir,
        class_to_idx=splits.class_to_idx,
        label_mode=config.label_mode,
        transform=data_transforms["val"],
    )
    test_dataset = MitosisDataset(
        data=splits.test,
        root_dir=config.patches_dir,
        class_to_idx=splits.class_to_idx,
        label_mode=config.label_mode,
        transform=data_transforms["val"],
    )

    sampler: WeightedRandomSampler | None = None
    if config.use_sampler:
        class_counts = splits.train["expert1_label"].value_counts()
        inv_freq = 1.0 / class_counts
        sample_weights: list[float] = [
            float(w) for w in splits.train["expert1_label"].map(inv_freq).to_numpy()
        ]
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(splits.train),
            replacement=True,
            generator=make_generator(seed),
        )
        logger.info("WeightedRandomSampler enabled for imbalanced training classes.")

    pin_memory = device is not None and device.type == "cuda"
    loader_generator = make_generator(seed)

    train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=config.num_workers,
        worker_init_fn=seed_worker,
        generator=loader_generator,
        pin_memory=pin_memory,
    )
    val_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        worker_init_fn=seed_worker,
        pin_memory=pin_memory,
    )
    test_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        worker_init_fn=seed_worker,
        pin_memory=pin_memory,
    )

    return DataBundle(
        loaders={"train": train_loader, "val": val_loader, "test": test_loader},
        splits=splits,
        transforms=data_transforms,
    )
