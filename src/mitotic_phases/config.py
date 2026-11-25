"""Typed, immutable configuration schemas and YAML I/O utilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

import torch
import yaml

MitosisMode = Literal["NMF", "AMF"]
LabelMode = Literal["soft", "hard"]
ModelArchitecture = Literal["convnext_tiny", "convnext_small", "convnext_base"]
TrainMode = Literal["full", "fine_tune", "feature_extraction"]

_VALID_MODES: frozenset[str] = frozenset({"NMF", "AMF"})
_VALID_LABEL_MODES: frozenset[str] = frozenset({"soft", "hard"})
_VALID_ARCHITECTURES: frozenset[str] = frozenset(
    {"convnext_tiny", "convnext_small", "convnext_base"}
)
_VALID_TRAIN_MODES: frozenset[str] = frozenset({"full", "fine_tune", "feature_extraction"})


@dataclass(frozen=True)
class ExperimentConfig:
    """Global experiment runtime and reproducibility settings."""

    seed: int = 45
    deterministic: bool = True
    device: str = "auto"
    output_dir: str = "outputs"

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ValueError(f"seed must be non-negative, got {self.seed}")
        if not self.output_dir.strip():
            raise ValueError("output_dir must be a non-empty path string")


@dataclass(frozen=True)
class DataConfig:
    """Dataset, splitting, augmentation, and DataLoader settings."""

    csv_path: str = "data/AMI-BR.csv"
    patches_dir: str = "data/patches/normal"
    mode: MitosisMode = "NMF"
    label_mode: LabelMode = "soft"
    min_class_count: int = 10
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    batch_size: int = 16
    num_workers: int = 2
    use_sampler: bool = True
    use_macenko: bool = False
    macenko_prob: float = 0.5
    macenko_num_refs: int = 5
    macenko_ref_paths: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.mode not in _VALID_MODES:
            raise ValueError(f"mode must be one of {sorted(_VALID_MODES)}, got {self.mode!r}")
        if self.label_mode not in _VALID_LABEL_MODES:
            raise ValueError(
                f"label_mode must be one of {sorted(_VALID_LABEL_MODES)}, got {self.label_mode!r}"
            )
        if self.min_class_count < 1:
            raise ValueError(f"min_class_count must be >= 1, got {self.min_class_count}")

        ratios = (self.train_ratio, self.val_ratio, self.test_ratio)
        if any(r <= 0.0 or r >= 1.0 for r in ratios):
            raise ValueError(f"Split ratios must each lie in (0, 1), got {ratios}")
        if abs(sum(ratios) - 1.0) > 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, got {sum(ratios):.6f}")

        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.num_workers < 0:
            raise ValueError(f"num_workers must be >= 0, got {self.num_workers}")
        if not (0.0 <= self.macenko_prob <= 1.0):
            raise ValueError(f"macenko_prob must lie in [0, 1], got {self.macenko_prob}")
        if self.macenko_num_refs < 1:
            raise ValueError(f"macenko_num_refs must be >= 1, got {self.macenko_num_refs}")


@dataclass(frozen=True)
class ModelConfig:
    """Backbone architecture and fine-tuning policy."""

    architecture: ModelArchitecture = "convnext_tiny"
    pretrained: bool = True
    train_mode: TrainMode = "full"

    def __post_init__(self) -> None:
        if self.architecture not in _VALID_ARCHITECTURES:
            raise ValueError(
                f"architecture must be one of {sorted(_VALID_ARCHITECTURES)}, "
                f"got {self.architecture!r}"
            )
        if self.train_mode not in _VALID_TRAIN_MODES:
            raise ValueError(
                f"train_mode must be one of {sorted(_VALID_TRAIN_MODES)}, got {self.train_mode!r}"
            )


@dataclass(frozen=True)
class TrainingConfig:
    """Optimization hyperparameters and evaluation thresholds."""

    epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 0.1
    bounded_acc_low_threshold: float = 0.25
    bounded_acc_high_threshold: float = 0.75

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {self.epochs}")
        if self.learning_rate <= 0.0:
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate}")
        if self.weight_decay < 0.0:
            raise ValueError(f"weight_decay must be >= 0, got {self.weight_decay}")
        if not (0.0 <= self.bounded_acc_low_threshold <= 1.0):
            raise ValueError(
                "bounded_acc_low_threshold must lie in [0, 1], "
                f"got {self.bounded_acc_low_threshold}"
            )
        if not (0.0 <= self.bounded_acc_high_threshold <= 1.0):
            raise ValueError(
                "bounded_acc_high_threshold must lie in [0, 1], "
                f"got {self.bounded_acc_high_threshold}"
            )
        if self.bounded_acc_low_threshold > self.bounded_acc_high_threshold:
            raise ValueError("bounded_acc_low_threshold must be <= bounded_acc_high_threshold")


@dataclass(frozen=True)
class PipelineConfig:
    """Top-level immutable configuration aggregating all pipeline sections."""

    experiment: ExperimentConfig = ExperimentConfig()
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()


def resolve_device(device_spec: str) -> torch.device:
    """Resolve a device specification ('auto', 'cuda', 'cpu', etc.) to a torch.device."""
    if device_spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_spec)


def _get_section(raw: dict[str, Any], name: str) -> dict[str, Any]:
    """Extract and validate an optional YAML section as a dictionary."""
    section = raw.get(name, {})
    if section is None:
        return {}
    if not isinstance(section, dict):
        raise ValueError(f"Config section '{name}' must be a mapping, got {type(section).__name__}")
    return cast(dict[str, Any], section)


def load_config(path: str | Path) -> PipelineConfig:
    """Load and validate a PipelineConfig from a YAML file.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        Validated immutable PipelineConfig instance.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        ValueError: If the YAML root or any field fails validation.
    """
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        raw_loaded: Any = yaml.safe_load(handle)

    if raw_loaded is None:
        raw: dict[str, Any] = {}
    elif isinstance(raw_loaded, dict):
        raw = cast(dict[str, Any], raw_loaded)
    else:
        raise ValueError(f"YAML root must be a mapping, got {type(raw_loaded).__name__}")

    exp_raw = _get_section(raw, "experiment")
    data_raw = dict(_get_section(raw, "data"))
    model_raw = _get_section(raw, "model")
    train_raw = _get_section(raw, "training")

    if "macenko_ref_paths" in data_raw:
        ref_paths = data_raw["macenko_ref_paths"]
        if not isinstance(ref_paths, list | tuple):
            raise ValueError("data.macenko_ref_paths must be a list of path strings")
        data_raw["macenko_ref_paths"] = tuple(str(p) for p in ref_paths)

    return PipelineConfig(
        experiment=ExperimentConfig(**exp_raw),
        data=DataConfig(**data_raw),
        model=ModelConfig(**model_raw),
        training=TrainingConfig(**train_raw),
    )


def save_config(config: PipelineConfig, path: str | Path) -> None:
    """Serialize a PipelineConfig to YAML for experiment reproducibility.

    Args:
        config: Pipeline configuration to persist.
        path: Destination YAML file path.
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(config)
    payload["data"]["macenko_ref_paths"] = list(config.data.macenko_ref_paths)
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
