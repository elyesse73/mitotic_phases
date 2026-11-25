"""Integration and smoke tests for determinism, shared splits, ConvNeXt, Trainer, and plotting."""

from __future__ import annotations

import math
import random
from pathlib import Path
from unittest.mock import patch

import matplotlib
import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

from mitotic_phases import visualization as vis_mod
from mitotic_phases.config import DataConfig, ModelConfig, TrainingConfig
from mitotic_phases.data import (
    MacenkoAugment,
    MitosisDataset,
    build_class_mapping,
    filter_metadata,
    prepare_dataloaders,
)
from mitotic_phases.metrics import InferenceOutputs, evaluate_groups
from mitotic_phases.model import (
    build_convnext,
    extract_convnext_features_and_logits,
    get_trainable_parameters,
)
from mitotic_phases.reproducibility import seed_everything
from mitotic_phases.trainer import Trainer, load_checkpoint, snapshot_state_dict_to_cpu
from mitotic_phases.visualization import plot_confidence_calibration, plot_gradcam_explanation


def _create_synthetic_dataset(tmp_path: Path, n_per_class: int = 12) -> tuple[pd.DataFrame, Path]:
    """Create synthetic RGB patches and an AMI-Br-compatible metadata DataFrame."""
    patches_dir = tmp_path / "patches"
    patches_dir.mkdir(parents=True, exist_ok=True)

    classes = ["NMF anaphase-telophase", "NMF metaphase", "NMF prometaphase"]
    rows: list[dict[str, object]] = []
    uid = 0

    rng = np.random.default_rng(0)
    for cls_idx, cls_name in enumerate(classes):
        for i in range(n_per_class):
            uid += 1
            # Introduce ~25% inter-expert disagreements between adjacent phases
            exp2_name = classes[(cls_idx + 1) % len(classes)] if i % 4 == 0 else cls_name
            img_arr = rng.integers(0, 255, size=(32, 32, 3), dtype=np.uint8)
            Image.fromarray(img_arr).save(patches_dir / f"SYNTH_{uid}.png")
            rows.append(
                {
                    "dataset": "SYNTH",
                    "uid": uid,
                    "expert1_label": cls_name,
                    "expert2_label": exp2_name,
                    "expert1_atypical": False,
                    "expert2_atypical": False,
                    "expert3_atypical": False,
                    "majority_atypical": False,
                }
            )

    return pd.DataFrame(rows), patches_dir


def test_seed_everything_determinism() -> None:
    """Two calls with the same seed must reproduce identical Python, NumPy, and PyTorch draws."""
    seed_everything(45, deterministic=True)
    py_a = [random.random() for _ in range(5)]
    np_a = np.random.rand(5)
    pt_a = torch.randn(5)

    seed_everything(45, deterministic=True)
    py_b = [random.random() for _ in range(5)]
    np_b = np.random.rand(5)
    pt_b = torch.randn(5)

    assert py_a == py_b
    assert np.allclose(np_a, np_b)
    assert torch.allclose(pt_a, pt_b)


def test_shared_class_vocabulary_across_all_splits(tmp_path: Path) -> None:
    """Train, val, and test datasets must share the exact same class_to_idx mapping."""
    df, patches_dir = _create_synthetic_dataset(tmp_path, n_per_class=12)
    cfg = DataConfig(
        patches_dir=str(patches_dir),
        mode="NMF",
        label_mode="soft",
        min_class_count=5,
        batch_size=4,
        num_workers=0,
        use_sampler=True,
        use_macenko=False,
    )

    bundle = prepare_dataloaders(df, cfg, seed=45, device=torch.device("cpu"))
    train_ds = bundle.loaders["train"].dataset
    val_ds = bundle.loaders["val"].dataset
    test_ds = bundle.loaders["test"].dataset

    assert isinstance(train_ds, MitosisDataset)
    assert isinstance(val_ds, MitosisDataset)
    assert isinstance(test_ds, MitosisDataset)
    assert train_ds.class_to_idx == val_ds.class_to_idx == test_ds.class_to_idx
    assert train_ds.classes == bundle.splits.classes

    images, targets = next(iter(bundle.loaders["val"]))
    assert images.shape[1:] == (3, 224, 224)
    assert targets.shape == (images.shape[0], len(bundle.splits.classes))
    assert torch.allclose(targets.sum(dim=1), torch.ones(images.shape[0]))


def test_hard_label_mode_returns_integer_targets(tmp_path: Path) -> None:
    """Switching label_mode to 'hard' returns 1D int64 targets compatible with CrossEntropyLoss."""
    df, patches_dir = _create_synthetic_dataset(tmp_path, n_per_class=10)
    cfg = DataConfig(
        patches_dir=str(patches_dir),
        mode="NMF",
        label_mode="hard",
        min_class_count=5,
        batch_size=4,
        num_workers=0,
        use_sampler=False,
        use_macenko=False,
    )

    bundle = prepare_dataloaders(df, cfg, seed=45, device=torch.device("cpu"))
    _, targets = next(iter(bundle.loaders["val"]))
    assert targets.ndim == 1
    assert targets.dtype == torch.long


def test_build_class_mapping_includes_expert2_only_labels() -> None:
    """Vocabulary construction must union Expert 1 and Expert 2 labels to avoid KeyErrors."""
    df = pd.DataFrame(
        {
            "dataset": ["D", "D"],
            "uid": [1, 2],
            "expert1_label": ["NMF metaphase", "NMF metaphase"],
            "expert2_label": ["NMF metaphase", "NMF ring shape"],
            "majority_atypical": [False, False],
        }
    )
    classes, mapping = build_class_mapping(df)
    assert classes == ("NMF metaphase", "NMF ring shape")
    assert mapping == {"NMF metaphase": 0, "NMF ring shape": 1}


def test_filter_metadata_drops_rare_classes() -> None:
    """Classes with fewer samples than min_class_count are excluded from both expert columns."""
    df = pd.DataFrame(
        {
            "dataset": ["D"] * 6,
            "uid": list(range(6)),
            "expert1_label": ["NMF A", "NMF A", "NMF A", "NMF B", "NMF B", "NMF C"],
            "expert2_label": ["NMF A", "NMF A", "NMF B", "NMF B", "NMF B", "NMF C"],
            "majority_atypical": [False] * 6,
        }
    )
    cfg = DataConfig(mode="NMF", min_class_count=2)
    filtered = filter_metadata(df, cfg)
    assert "NMF C" not in set(filtered["expert1_label"])
    assert len(filtered) == 5


@pytest.mark.parametrize(
    ("train_mode", "expect_backbone_frozen"),
    [
        ("feature_extraction", True),
        ("fine_tune", False),
        ("full", False),
    ],
)
def test_convnext_freezing_modes_and_single_pass_consistency(
    train_mode: str,
    expect_backbone_frozen: bool,
) -> None:
    """ConvNeXt freezing policies behave correctly and single-pass extraction matches forward()."""
    model = build_convnext(
        ModelConfig(
            architecture="convnext_tiny",
            pretrained=False,
            train_mode=train_mode,  # type: ignore[arg-type]
        ),
        num_classes=4,
    )
    model.eval()

    first_stage_trainable = any(p.requires_grad for p in model.features[0].parameters())
    last_stage_trainable = any(p.requires_grad for p in model.features[7].parameters())
    head_trainable = any(p.requires_grad for p in model.classifier[2].parameters())

    assert head_trainable is True
    if expect_backbone_frozen:
        assert not first_stage_trainable and not last_stage_trainable
    elif train_mode == "fine_tune":
        assert not first_stage_trainable and last_stage_trainable
    else:
        assert first_stage_trainable and last_stage_trainable

    assert len(get_trainable_parameters(model)) > 0

    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        direct_logits = model(x)
        feat_maps, embeddings, extracted_logits = extract_convnext_features_and_logits(model, x)

    assert feat_maps.ndim == 4
    assert embeddings.shape == (2, 768)
    assert extracted_logits.shape == (2, 4)
    assert torch.allclose(direct_logits, extracted_logits, atol=1e-5)


class _TinyClassifier(nn.Module):
    """Minimal fast network for CPU trainer smoke tests."""

    def __init__(self, num_classes: int = 4) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 16 * 16, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.net(x)
        return out


class _IdentityLogitsModule(nn.Module):
    """Pass-through logits module with a single frozen-step scale for exact metric tests."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale


@pytest.mark.parametrize("label_mode", ["soft", "hard"])
def test_trainer_smoke_metadata_and_raw_checkpoint_reload(
    tmp_path: Path,
    label_mode: str,
) -> None:
    """Trainer saves metadata-rich checkpoints and load_checkpoint also loads raw state_dicts."""
    seed_everything(45, deterministic=True)
    x = torch.randn(12, 3, 16, 16)
    if label_mode == "soft":
        y = torch.zeros(12, 4, dtype=torch.float32)
        for i in range(12):
            y[i, i % 4] += 0.5
            y[i, (i + (1 if i % 3 == 0 else 0)) % 4] += 0.5
    else:
        y = torch.arange(12, dtype=torch.long) % 4

    ds = TensorDataset(x, y)
    loaders: dict[str, DataLoader[tuple[torch.Tensor, torch.Tensor]]] = {
        "train": DataLoader(ds, batch_size=4, shuffle=True),  # type: ignore[arg-type]
        "val": DataLoader(ds, batch_size=4, shuffle=False),  # type: ignore[arg-type]
    }

    classes = ("NMF a", "NMF b", "NMF c", "NMF d")
    class_to_idx = {name: i for i, name in enumerate(classes)}

    model = _TinyClassifier(num_classes=4)
    trainer = Trainer(
        model=model,
        training_config=TrainingConfig(epochs=2, learning_rate=1e-3, weight_decay=0.01),
        label_mode=label_mode,  # type: ignore[arg-type]
        device=torch.device("cpu"),
    )
    ckpt_path = tmp_path / f"best_{label_mode}.pth"
    result = trainer.fit(
        loaders,
        checkpoint_path=ckpt_path,
        classes=classes,
        class_to_idx=class_to_idx,
        architecture="convnext_tiny",
        train_mode="full",
    )

    assert len(result.history) == 2
    assert ckpt_path.is_file()

    cpu_snap = snapshot_state_dict_to_cpu(model)
    assert all(tensor.device.type == "cpu" for tensor in cpu_snap.values())

    fresh_model = _TinyClassifier(num_classes=4)
    meta = load_checkpoint(fresh_model, ckpt_path, device=torch.device("cpu"))
    assert meta["label_mode"] == label_mode
    assert meta["classes"] == list(classes)
    assert meta["class_to_idx"] == class_to_idx
    assert meta["epoch"] == result.best_epoch
    for p1, p2 in zip(model.parameters(), fresh_model.parameters(), strict=True):
        assert torch.allclose(p1, p2)

    # Backward compatibility: raw state_dict checkpoint loads cleanly with empty metadata
    raw_ckpt = tmp_path / f"raw_{label_mode}.pth"
    torch.save(cpu_snap, raw_ckpt)
    legacy_model = _TinyClassifier(num_classes=4)
    raw_meta = load_checkpoint(legacy_model, raw_ckpt, device=torch.device("cpu"))
    assert raw_meta == {}
    for p1, p2 in zip(model.parameters(), legacy_model.parameters(), strict=True):
        assert torch.allclose(p1, p2)


def test_trainer_sample_weighted_bounded_accuracy_on_unequal_batches() -> None:
    """Epoch bounded accuracy must weight by sample count rather than averaging batch means."""
    # 5 samples split into batches of size 4 and 1:
    # Batch 1 (4 samples): 1 valid + 3 invalid -> batch acc = 0.25
    # Batch 2 (1 sample):  1 valid             -> batch acc = 1.00
    # Unweighted batch mean = 0.625, true sample-weighted accuracy = 2 / 5 = 0.40
    probs = torch.tensor(
        [
            [0.92, 0.05, 0.03],  # consensus 0 -> valid
            [0.90, 0.07, 0.03],  # ambiguous {0,1}, overconfident (>0.75) -> invalid
            [0.90, 0.07, 0.03],  # ambiguous {0,1}, overconfident (>0.75) -> invalid
            [0.10, 0.85, 0.05],  # consensus 0, predicts 1 -> invalid
            [0.60, 0.35, 0.05],  # ambiguous {0,1}, calibrated (<=0.75) -> valid
        ],
        dtype=torch.float32,
    )
    logits = torch.log(probs)
    targets = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.5, 0.5, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
        ],
        dtype=torch.float32,
    )

    ds = TensorDataset(logits, targets)
    loaders: dict[str, DataLoader[tuple[torch.Tensor, torch.Tensor]]] = {
        "train": DataLoader(ds, batch_size=4, shuffle=False),  # type: ignore[arg-type]
        "val": DataLoader(ds, batch_size=4, shuffle=False),  # type: ignore[arg-type]
    }

    model = _IdentityLogitsModule()
    zero_step_opt = torch.optim.SGD(model.parameters(), lr=0.0)
    trainer = Trainer(
        model=model,
        training_config=TrainingConfig(epochs=1, learning_rate=1e-4, weight_decay=0.0),
        label_mode="soft",
        device=torch.device("cpu"),
        optimizer=zero_step_opt,
    )
    result = trainer.fit(loaders)
    assert math.isclose(result.history[0].val_metric, 0.4, rel_tol=1e-6)


def test_macenko_augment_validation_prob_and_fallback(tmp_path: Path) -> None:
    """MacenkoAugment validates probability bounds, refs, and handles linalg failures safely."""
    with pytest.raises(ValueError, match="p must lie"):
        MacenkoAugment([], p=1.5)

    pytest.importorskip("cv2")
    pytest.importorskip("torchstain")

    with pytest.raises(ValueError, match="at least one valid"):
        MacenkoAugment([], p=0.5)

    with pytest.raises(ValueError, match="at least one valid"):
        MacenkoAugment([tmp_path / "missing_ref.png"], p=0.5)

    # Synthetic H&E-like pink/purple patch for valid Macenko normalizer fitting
    ref_path = tmp_path / "ref.png"
    rng = np.random.default_rng(99)
    tissue = np.zeros((64, 64, 3), dtype=np.uint8)
    tissue[:, :, 0] = rng.integers(140, 230, size=(64, 64), dtype=np.uint8)
    tissue[:, :, 1] = rng.integers(70, 150, size=(64, 64), dtype=np.uint8)
    tissue[:, :, 2] = rng.integers(130, 220, size=(64, 64), dtype=np.uint8)
    Image.fromarray(tissue).save(ref_path)

    pil_input = Image.fromarray(tissue)

    aug_never = MacenkoAugment([ref_path], p=0.0)
    assert aug_never(pil_input) is pil_input

    aug_always = MacenkoAugment([ref_path], p=1.0)
    out_always = aug_always(pil_input)
    assert isinstance(out_always, Image.Image)
    assert out_always.size == pil_input.size

    with patch.object(
        aug_always._normalizers[0],
        "normalize",
        side_effect=RuntimeError("singular matrix"),
    ):
        assert aug_always(pil_input) is pil_input


def test_headless_calibration_and_gradcam_fallback_plots(tmp_path: Path) -> None:
    """Headless plotting works under Agg and Grad-CAM++ fallback avoids UnboundLocalError."""
    matplotlib.use("Agg")
    probs = np.array(
        [
            [0.95, 0.03, 0.02],
            [0.91, 0.06, 0.03],
            [0.55, 0.40, 0.05],
            [0.50, 0.45, 0.05],
        ],
        dtype=np.float64,
    )
    top2 = np.argsort(probs, axis=1)[:, ::-1][:, :2].astype(np.int64)
    outputs = InferenceOutputs(
        probs=probs,
        preds_top1=top2[:, 0].copy(),
        preds_top2=top2,
        confidences=probs.max(axis=1),
    )
    grp = evaluate_groups(outputs, np.array([0, 0, 0, 1]), np.array([0, 0, 1, 0]))

    cal_png = tmp_path / "calibration.png"
    plot_confidence_calibration(grp, save_path=cal_png, show=False)
    assert cal_png.is_file()
    assert cal_png.stat().st_size > 0

    df, patches_dir = _create_synthetic_dataset(tmp_path, n_per_class=6)
    cfg = DataConfig(
        patches_dir=str(patches_dir),
        mode="NMF",
        label_mode="soft",
        min_class_count=3,
        batch_size=4,
        num_workers=0,
        use_sampler=False,
        use_macenko=False,
    )
    bundle = prepare_dataloaders(df, cfg, seed=45, device=torch.device("cpu"))
    val_ds = bundle.loaders["val"].dataset
    assert isinstance(val_ds, MitosisDataset)

    model = build_convnext(
        ModelConfig(
            architecture="convnext_tiny",
            pretrained=False,
            train_mode="feature_extraction",
        ),
        num_classes=len(bundle.splits.classes),
    )

    fallback_png = tmp_path / "gradcam_fallback.png"
    with patch.object(vis_mod, "HAS_GRAD_CAM", False):
        plot_gradcam_explanation(
            model=model,
            dataset=val_ds,
            idx=0,
            device=torch.device("cpu"),
            save_path=fallback_png,
            show=False,
        )
    assert fallback_png.is_file()
    assert fallback_png.stat().st_size > 0

    if vis_mod.HAS_GRAD_CAM:
        real_png = tmp_path / "gradcam_real.png"
        plot_gradcam_explanation(
            model=model,
            dataset=val_ds,
            idx=0,
            device=torch.device("cpu"),
            save_path=real_png,
            show=False,
        )
        assert real_png.is_file()
        assert real_png.stat().st_size > 0
