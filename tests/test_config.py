"""Unit tests for immutable YAML configuration loading, serialization, and validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from mitotic_phases.config import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    PipelineConfig,
    TrainingConfig,
    load_config,
    resolve_device,
    save_config,
)


def test_default_yaml_loads_cleanly() -> None:
    """Repository default.yaml should parse into a valid PipelineConfig."""
    default_path = Path("configs/default.yaml")
    if not default_path.is_file():
        pytest.skip("configs/default.yaml not found relative to current working directory.")

    cfg = load_config(default_path)
    assert cfg.experiment.seed == 45
    assert cfg.experiment.deterministic is True
    assert cfg.data.mode == "NMF"
    assert cfg.data.label_mode == "soft"
    assert cfg.model.architecture == "convnext_tiny"
    assert cfg.training.epochs == 50


def test_save_and_load_config_roundtrip(tmp_path: Path) -> None:
    """Saving and reloading a PipelineConfig must preserve all fields identically."""
    original = PipelineConfig(
        experiment=ExperimentConfig(seed=123, deterministic=False, device="cpu", output_dir="out"),
        data=DataConfig(
            csv_path="custom/meta.csv",
            patches_dir="custom/patches",
            mode="AMF",
            label_mode="hard",
            batch_size=8,
            num_workers=0,
            use_macenko=True,
            macenko_prob=0.3,
            macenko_ref_paths=("ref1.png", "ref2.png"),
        ),
        model=ModelConfig(
            architecture="convnext_small",
            pretrained=False,
            train_mode="fine_tune",
        ),
        training=TrainingConfig(epochs=5, learning_rate=5e-5, weight_decay=0.05),
    )

    yaml_path = tmp_path / "nested" / "config.yaml"
    save_config(original, yaml_path)
    reloaded = load_config(yaml_path)

    assert reloaded == original


def test_load_missing_file_raises(tmp_path: Path) -> None:
    """Loading a non-existent config file must raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "does_not_exist.yaml")


def test_invalid_split_ratios_raise() -> None:
    """Split ratios that do not sum to 1.0 must fail validation."""
    with pytest.raises(ValueError, match="sum to 1.0"):
        DataConfig(train_ratio=0.7, val_ratio=0.1, test_ratio=0.1)


def test_invalid_thresholds_raise() -> None:
    """low_threshold exceeding high_threshold must fail validation."""
    with pytest.raises(ValueError, match="bounded_acc_low_threshold"):
        TrainingConfig(bounded_acc_low_threshold=0.8, bounded_acc_high_threshold=0.7)


def test_invalid_modes_raise() -> None:
    """Invalid enum-like literal strings must raise ValueError at construction."""
    with pytest.raises(ValueError, match="mode"):
        DataConfig(mode="INVALID")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="label_mode"):
        DataConfig(label_mode="semi")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="train_mode"):
        ModelConfig(train_mode="partial")  # type: ignore[arg-type]


def test_resolve_device_cpu() -> None:
    """Explicit 'cpu' device specification resolves cleanly."""
    device = resolve_device("cpu")
    assert device.type == "cpu"
