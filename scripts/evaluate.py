"""CLI entrypoint for comprehensive evaluation, calibration analysis, and explainability."""

from __future__ import annotations

import argparse
import json
import logging
import random
from dataclasses import replace
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import numpy.typing as npt
import pandas as pd

from mitotic_phases.config import PipelineConfig, load_config, resolve_device
from mitotic_phases.data import MitosisDataset, encode_soft_label, prepare_dataloaders
from mitotic_phases.metrics import (
    collect_predictions,
    evaluate_against_experts,
    evaluate_entropy_by_agreement,
    evaluate_groups,
    evaluate_top2_indulgent,
    soft_confusion_matrix,
)
from mitotic_phases.model import build_convnext
from mitotic_phases.reproducibility import seed_everything
from mitotic_phases.trainer import load_checkpoint


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for model evaluation."""
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a trained ConvNeXt checkpoint on AMI-Br consensus/disagreement splits."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to trained model weights (.pth).",
    )
    parser.add_argument(
        "--split",
        choices=["val", "test"],
        default="test",
        help="Evaluation split to run diagnostics on.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write metrics JSON and evaluation plots.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override config.experiment.device ('auto', 'cpu', 'cuda').",
    )
    parser.add_argument(
        "--skip-tsne",
        action="store_true",
        help="Skip t-SNE latent space projections (faster on CPU).",
    )
    parser.add_argument(
        "--num-gradcam",
        type=int,
        default=3,
        help="Number of random samples to render with Grad-CAM++ (0 to disable).",
    )
    return parser.parse_args()


def _build_soft_ground_truth_matrix(dataset: MitosisDataset) -> npt.NDArray[np.float64]:
    """Construct the (N, C) bimodal soft target matrix for evaluation regardless of label_mode."""
    targets = [
        encode_soft_label(int(l1), int(l2), dataset.num_classes).numpy()
        for l1, l2 in zip(dataset.labels1, dataset.labels2, strict=True)
    ]
    return np.asarray(targets, dtype=np.float64)


def main() -> None:
    """Run full evaluation battery and persist metrics and diagnostic plots."""
    matplotlib.use("Agg")
    from mitotic_phases.visualization import (
        extract_latent_bundle,
        plot_confidence_calibration,
        plot_entropy_distribution,
        plot_expert_confusion_matrices,
        plot_gradcam_explanation,
        plot_soft_confusion_matrix,
        plot_top2_indulgent_matrices,
        plot_tsne_confidence,
        plot_tsne_dual_expert,
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logger = logging.getLogger(__name__)

    args = parse_args()
    base_config = load_config(args.config)
    exp_cfg = base_config.experiment
    if args.device is not None:
        exp_cfg = replace(exp_cfg, device=args.device)
    if args.output_dir is not None:
        exp_cfg = replace(exp_cfg, output_dir=str(args.output_dir))

    # Avoid downloading ImageNet weights when loading a trained checkpoint.
    model_cfg = replace(base_config.model, pretrained=False)
    config = PipelineConfig(
        experiment=exp_cfg,
        data=base_config.data,
        model=model_cfg,
        training=base_config.training,
    )

    output_dir = Path(config.experiment.output_dir)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(
        seed=config.experiment.seed,
        deterministic=config.experiment.deterministic,
    )
    device = resolve_device(config.experiment.device)

    df = pd.read_csv(config.data.csv_path)
    bundle = prepare_dataloaders(
        df=df,
        config=config.data,
        seed=config.experiment.seed,
        device=device,
    )

    loader = bundle.loaders[args.split]
    dataset = loader.dataset
    if not isinstance(dataset, MitosisDataset):
        raise TypeError(f"Expected MitosisDataset, got {type(dataset).__name__}")

    model = build_convnext(config.model, num_classes=len(bundle.splits.classes))
    metadata = load_checkpoint(model, args.checkpoint, device=device)
    if metadata:
        logger.info("Checkpoint metadata: %s", sorted(metadata.keys()))
    model = model.to(device)

    outputs = collect_predictions(model, loader, device=device)
    exp1 = dataset.labels1
    exp2 = dataset.labels2
    classes = dataset.classes

    expert_res = evaluate_against_experts(outputs.preds_top1, exp1, exp2, classes)
    top2_res = evaluate_top2_indulgent(outputs.preds_top2, exp1, exp2, classes)
    group_res = evaluate_groups(outputs, exp1, exp2)
    entropy_res = evaluate_entropy_by_agreement(outputs.probs, exp1, exp2)

    y_true_soft = _build_soft_ground_truth_matrix(dataset)
    cm_soft = soft_confusion_matrix(y_true_soft, outputs.probs, normalize=True)

    metrics_payload: dict[str, Any] = {
        "split": args.split,
        "checkpoint": str(args.checkpoint),
        "checkpoint_metadata": metadata,
        "n_samples": len(dataset),
        "classes": list(classes),
        "inter_expert_agreement": expert_res.inter_expert_agreement,
        "vs_expert1": {
            "accuracy": expert_res.accuracy_vs_expert1,
            "balanced_accuracy": expert_res.balanced_accuracy_vs_expert1,
            "macro_f1": expert_res.macro_f1_vs_expert1,
            "top2_indulgent_accuracy": top2_res.top2_accuracy_vs_expert1,
        },
        "vs_expert2": {
            "accuracy": expert_res.accuracy_vs_expert2,
            "balanced_accuracy": expert_res.balanced_accuracy_vs_expert2,
            "macro_f1": expert_res.macro_f1_vs_expert2,
            "top2_indulgent_accuracy": top2_res.top2_accuracy_vs_expert2,
        },
        "group_a_consensus": {
            "count": group_res.n_consensus,
            "accuracy": group_res.consensus_accuracy,
            "mean_confidence": group_res.consensus_mean_confidence,
            "std_confidence": group_res.consensus_std_confidence,
            "mean_entropy": entropy_res.mean_entropy_consensus,
        },
        "group_b_disagreement": {
            "count": group_res.n_disagreement,
            "double_coverage_accuracy": group_res.disagreement_double_coverage,
            "relaxed_accuracy": group_res.disagreement_relaxed_accuracy,
            "mean_confidence": group_res.disagreement_mean_confidence,
            "std_confidence": group_res.disagreement_std_confidence,
            "mean_entropy": entropy_res.mean_entropy_disagreement,
        },
    }

    metrics_path = output_dir / f"evaluation_{args.split}.json"
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    logger.info("Saved evaluation summary to %s.", metrics_path)
    logger.info(
        "Group A (n=%d): Acc=%.2f%%, Conf=%.1f%% | Group B (n=%d): DCA=%.2f%%, Conf=%.1f%%",
        group_res.n_consensus,
        group_res.consensus_accuracy * 100.0,
        group_res.consensus_mean_confidence * 100.0,
        group_res.n_disagreement,
        group_res.disagreement_double_coverage * 100.0,
        group_res.disagreement_mean_confidence * 100.0,
    )

    plot_expert_confusion_matrices(
        expert_res,
        save_path=figures_dir / "confusion_vs_experts.png",
    )
    plot_top2_indulgent_matrices(
        top2_res,
        save_path=figures_dir / "confusion_top2_indulgent.png",
    )
    plot_soft_confusion_matrix(
        cm_soft,
        classes=classes,
        save_path=figures_dir / "soft_confusion_matrix.png",
    )
    plot_confidence_calibration(
        group_res,
        save_path=figures_dir / "confidence_calibration.png",
    )
    plot_entropy_distribution(
        entropy_res,
        save_path=figures_dir / "entropy_distribution.png",
    )

    if not args.skip_tsne and len(dataset) >= 3:
        latent = extract_latent_bundle(model, loader, device=device, max_samples=2000)
        exp1_names = dataset.data["expert1_label"].astype(str).tolist()
        exp2_names = dataset.data["expert2_label"].astype(str).tolist()
        plot_tsne_confidence(
            latent,
            expert1_labels=exp1_names,
            expert2_labels=exp2_names,
            seed=config.experiment.seed,
            save_path=figures_dir / "tsne_confidence.png",
        )
        plot_tsne_dual_expert(
            latent,
            expert1_labels=exp1_names,
            expert2_labels=exp2_names,
            classes=classes,
            seed=config.experiment.seed,
            save_path=figures_dir / "tsne_dual_expert.png",
        )

    if args.num_gradcam > 0 and len(dataset) > 0:
        n_samples = min(args.num_gradcam, len(dataset))
        rng = random.Random(config.experiment.seed)
        indices = rng.sample(range(len(dataset)), k=n_samples)
        for sample_idx in indices:
            plot_gradcam_explanation(
                model=model,
                dataset=dataset,
                idx=sample_idx,
                device=device,
                save_path=figures_dir / f"gradcam_sample_{sample_idx}.png",
            )


if __name__ == "__main__":
    main()
