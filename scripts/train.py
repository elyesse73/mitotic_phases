"""CLI entrypoint for training ConvNeXt mitotic phase classifiers (soft or hard labels)."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, replace
from pathlib import Path

import matplotlib
import pandas as pd

from mitotic_phases.config import (
    LabelMode,
    PipelineConfig,
    load_config,
    resolve_device,
    save_config,
)
from mitotic_phases.data import prepare_dataloaders, summarize_metadata
from mitotic_phases.model import build_convnext
from mitotic_phases.reproducibility import seed_everything
from mitotic_phases.trainer import Trainer


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for training."""
    parser = argparse.ArgumentParser(
        description="Train a ConvNeXt mitotic phase classifier with soft or hard expert labels."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--label-mode",
        choices=["soft", "hard"],
        default=None,
        help="Override config.data.label_mode ('soft' or 'hard').",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override config.training.epochs.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override config.experiment.device ('auto', 'cpu', 'cuda').",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override config.experiment.output_dir.",
    )
    return parser.parse_args()


def _apply_overrides(config: PipelineConfig, args: argparse.Namespace) -> PipelineConfig:
    """Return a new immutable PipelineConfig reflecting CLI overrides."""
    exp_cfg = config.experiment
    data_cfg = config.data
    train_cfg = config.training

    if args.device is not None:
        exp_cfg = replace(exp_cfg, device=args.device)
    if args.output_dir is not None:
        exp_cfg = replace(exp_cfg, output_dir=str(args.output_dir))
    if args.label_mode is not None:
        mode: LabelMode = args.label_mode
        data_cfg = replace(data_cfg, label_mode=mode)
    if args.epochs is not None:
        train_cfg = replace(train_cfg, epochs=args.epochs)

    return PipelineConfig(
        experiment=exp_cfg,
        data=data_cfg,
        model=config.model,
        training=train_cfg,
    )


def main() -> None:
    """Execute full deterministic training workflow and persist artifacts."""
    matplotlib.use("Agg")
    from mitotic_phases.visualization import plot_training_history

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logger = logging.getLogger(__name__)

    args = parse_args()
    config = _apply_overrides(load_config(args.config), args)

    output_dir = Path(config.experiment.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, output_dir / "resolved_config.yaml")

    seed_everything(
        seed=config.experiment.seed,
        deterministic=config.experiment.deterministic,
    )
    device = resolve_device(config.experiment.device)

    df = pd.read_csv(config.data.csv_path)
    summary = summarize_metadata(df)
    logger.info(
        "Loaded %d annotated cells (inter-expert phase agreement: %.2f%%).",
        summary.total_samples,
        summary.phase_agreement_rate * 100.0,
    )

    bundle = prepare_dataloaders(
        df=df,
        config=config.data,
        seed=config.experiment.seed,
        device=device,
    )
    model = build_convnext(config.model, num_classes=len(bundle.splits.classes))

    trainer = Trainer(
        model=model,
        training_config=config.training,
        label_mode=config.data.label_mode,
        device=device,
    )
    checkpoint_path = output_dir / f"best_model_{config.data.label_mode}.pth"
    result = trainer.fit(
        bundle.loaders,
        checkpoint_path=checkpoint_path,
        classes=bundle.splits.classes,
        class_to_idx=bundle.splits.class_to_idx,
        architecture=config.model.architecture,
        train_mode=config.model.train_mode,
    )

    history_payload = {
        "best_epoch": result.best_epoch,
        "best_val_loss": result.best_val_loss,
        "checkpoint_path": str(result.checkpoint_path) if result.checkpoint_path else None,
        "classes": list(bundle.splits.classes),
        "history": [asdict(entry) for entry in result.history],
    }
    history_path = output_dir / "training_history.json"
    history_path.write_text(json.dumps(history_payload, indent=2), encoding="utf-8")
    logger.info("Saved training history to %s.", history_path)

    plot_training_history(result.history, save_path=output_dir / "training_curves.png")


if __name__ == "__main__":
    main()
