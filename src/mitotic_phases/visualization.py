"""Headless-safe plotting utilities for calibration, confusion matrices, t-SNE, and Grad-CAM++."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import seaborn as sns
import torch
from matplotlib.figure import Figure
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from torchvision import models as tv_models

from mitotic_phases.data import IMAGENET_MEAN, IMAGENET_STD, MitosisDataset
from mitotic_phases.metrics import (
    EntropyAnalysisResult,
    ExpertComparisonResult,
    GroupEvaluationResult,
    Top2IndulgentResult,
    normalize_confusion_matrix,
)
from mitotic_phases.model import extract_convnext_features_and_logits
from mitotic_phases.trainer import EpochMetrics

logger = logging.getLogger(__name__)

try:
    from pytorch_grad_cam import GradCAMPlusPlus
    from pytorch_grad_cam.utils.image import show_cam_on_image

    HAS_GRAD_CAM = True
except ImportError:
    GradCAMPlusPlus = None
    show_cam_on_image = None
    HAS_GRAD_CAM = False


@dataclass(frozen=True)
class LatentFeatureBundle:
    """Single-pass ConvNeXt latent representations and confidences for visualization."""

    upstream_features: npt.NDArray[np.float32] | None
    downstream_features: npt.NDArray[np.float32]
    confidences: npt.NDArray[np.float64]


def _finalize_figure(
    fig: Figure,
    save_path: str | Path | None = None,
    show: bool = False,
    dpi: int = 150,
) -> Figure:
    """Persist and/or display a Matplotlib figure safely without leaking memory."""
    fig.tight_layout()
    if save_path is not None:
        out_path = Path(save_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        logger.info("Saved figure to %s.", out_path)
    if show:
        plt.show()
    if save_path is not None and not show:
        plt.close(fig)
    return fig


def denormalize_image(img_tensor: torch.Tensor) -> npt.NDArray[np.float32]:
    """
    Convert an ImageNet-normalized ``(3, H, W)`` tensor to an ``(H, W, 3)`` array in ``[0, 1]``.
    """
    mean = np.asarray(IMAGENET_MEAN, dtype=np.float32)
    std = np.asarray(IMAGENET_STD, dtype=np.float32)
    img = img_tensor.detach().cpu().permute(1, 2, 0).numpy().astype(np.float32)
    img = std * img + mean
    return np.clip(img, 0.0, 1.0)


def format_prediction_text(
    scores: torch.Tensor | npt.NDArray[np.floating[Any]],
    class_names: Sequence[str],
    top_k: int = 3,
) -> str:
    """Format ground-truth (hard or soft) labels or predicted probabilities as readable text."""
    if isinstance(scores, torch.Tensor):
        if scores.ndim == 0:
            idx = int(scores.item())
            return str(class_names[idx])
        values = scores.detach().cpu().numpy()
        total = float(values.sum())
        entries: list[str] = []
        for idx, score in enumerate(values):
            val = float(score)
            if val <= 0.0:
                continue
            if abs(total - 1.0) < 1e-5 and abs(val - 1.0) < 1e-5:
                entries.append(str(class_names[idx]))
            else:
                entries.append(f"{class_names[idx]} ({val * 100:.0f}%)")
        return ", ".join(entries) if entries else "No Label"

    probs = np.asarray(scores, dtype=np.float64)
    order = np.argsort(probs)[::-1][: min(top_k, len(probs))]
    return ", ".join(f"{class_names[int(i)]} ({probs[int(i)] * 100:.2f}%)" for i in order)


def plot_training_history(
    history: Sequence[EpochMetrics],
    save_path: str | Path | None = None,
    show: bool = False,
) -> Figure:
    """Plot training and validation loss and primary metric curves across epochs."""
    if not history:
        raise ValueError("Cannot plot empty training history.")

    epochs = [m.epoch for m in history]
    train_loss = [m.train_loss for m in history]
    val_loss = [m.val_loss for m in history]
    train_metric = [m.train_metric for m in history]
    val_metric = [m.val_metric for m in history]
    metric_label = history[0].metric_name.replace("_", " ").title()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(epochs, train_loss, label="Train Loss")
    axes[0].plot(epochs, val_loss, label="Validation Loss")
    axes[0].set_title("Loss Curve")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, train_metric, label=f"Train {metric_label}")
    axes[1].plot(epochs, val_metric, label=f"Validation {metric_label}")
    axes[1].set_title(metric_label)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel(metric_label)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    return _finalize_figure(fig, save_path=save_path, show=show)


def plot_expert_confusion_matrices(
    result: ExpertComparisonResult,
    save_path: str | Path | None = None,
    show: bool = False,
) -> Figure:
    """Plot normalized confusion matrices: Model vs. Expert 1, Model vs. Expert 2, Inter-Expert."""
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    panels = (
        (
            result.cm_vs_expert1,
            f"Model vs Expert 1 (Acc: {result.accuracy_vs_expert1 * 100:.1f}%)",
            "Model Prediction",
            "Expert 1 Label",
            "Blues",
        ),
        (
            result.cm_vs_expert2,
            f"Model vs Expert 2 (Acc: {result.accuracy_vs_expert2 * 100:.1f}%)",
            "Model Prediction",
            "Expert 2 Label",
            "Greens",
        ),
        (
            result.cm_inter_expert,
            f"Inter-Expert Agreement ({result.inter_expert_agreement * 100:.1f}%)",
            "Expert 2 Label",
            "Expert 1 Label",
            "Oranges",
        ),
    )

    for ax, (cm, title, xlabel, ylabel, cmap) in zip(axes, panels, strict=True):
        cm_norm = normalize_confusion_matrix(cm)
        sns.heatmap(
            cm_norm,
            annot=True,
            fmt=".1%",
            cmap=cmap,
            cbar=False,
            xticklabels=list(result.classes),
            yticklabels=list(result.classes),
            vmin=0.0,
            vmax=1.0,
            ax=ax,
        )
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    return _finalize_figure(fig, save_path=save_path, show=show)


def plot_top2_indulgent_matrices(
    result: Top2IndulgentResult,
    save_path: str | Path | None = None,
    show: bool = False,
) -> Figure:
    """Plot Top-2 oracle confusion matrices against both experts."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    panels = (
        (
            result.cm_vs_expert1,
            f"Model vs Expert 1 (Top-2 Acc: {result.top2_accuracy_vs_expert1 * 100:.2f}%)",
            "Expert 1 Label",
        ),
        (
            result.cm_vs_expert2,
            f"Model vs Expert 2 (Top-2 Acc: {result.top2_accuracy_vs_expert2 * 100:.2f}%)",
            "Expert 2 Label",
        ),
    )

    for ax, (cm, title, ylabel) in zip(axes, panels, strict=True):
        cm_norm = normalize_confusion_matrix(cm)
        sns.heatmap(
            cm_norm,
            annot=True,
            fmt=".1%",
            cmap="Purples",
            cbar=False,
            xticklabels=list(result.classes),
            yticklabels=list(result.classes),
            vmin=0.0,
            vmax=1.0,
            ax=ax,
        )
        ax.set_title(title)
        ax.set_xlabel("Prediction (Top-2 Oracle)")
        ax.set_ylabel(ylabel)

    return _finalize_figure(fig, save_path=save_path, show=show)


def plot_soft_confusion_matrix(
    cm_soft: npt.NDArray[np.float64],
    classes: Sequence[str],
    save_path: str | Path | None = None,
    show: bool = False,
) -> Figure:
    """Plot probabilistic soft confusion matrix (probability mass distribution)."""
    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(
        cm_soft,
        annot=True,
        fmt=".1%",
        cmap="Blues",
        xticklabels=list(classes),
        yticklabels=list(classes),
        vmin=0.0,
        vmax=1.0,
        ax=ax,
    )
    ax.set_title("Soft Confusion Matrix (Probability Mass Distribution)")
    ax.set_xlabel("Model Predicted Distribution")
    ax.set_ylabel("Expert Target Distribution")
    return _finalize_figure(fig, save_path=save_path, show=show)


def plot_confidence_calibration(
    result: GroupEvaluationResult,
    save_path: str | Path | None = None,
    show: bool = False,
) -> Figure:
    """Plot confidence density distributions for Group A (Consensus) vs. Group B (Disagreement)."""
    fig, ax = plt.subplots(figsize=(9, 5.5))

    if len(result.confidences_consensus) > 1 and np.std(result.confidences_consensus) > 1e-6:
        sns.kdeplot(
            result.confidences_consensus,
            fill=True,
            label=f"Group A: Consensus (n={result.n_consensus}, "
            f"mean={result.consensus_mean_confidence * 100:.1f}%)",
            color="tab:green",
            clip=(0.0, 1.0),
            ax=ax,
        )
    elif len(result.confidences_consensus) > 0:
        ax.hist(
            result.confidences_consensus,
            bins=15,
            alpha=0.5,
            density=True,
            color="tab:green",
            label=f"Group A: Consensus (n={result.n_consensus})",
        )

    if len(result.confidences_disagreement) > 1 and np.std(result.confidences_disagreement) > 1e-6:
        sns.kdeplot(
            result.confidences_disagreement,
            fill=True,
            label=f"Group B: Disagreement (n={result.n_disagreement}, "
            f"mean={result.disagreement_mean_confidence * 100:.1f}%)",
            color="tab:orange",
            clip=(0.0, 1.0),
            ax=ax,
        )
    elif len(result.confidences_disagreement) > 0:
        ax.hist(
            result.confidences_disagreement,
            bins=15,
            alpha=0.5,
            density=True,
            color="tab:orange",
            label=f"Group B: Disagreement (n={result.n_disagreement})",
        )

    ax.set_title("Model Confidence Calibration across Consensus vs. Disagreement Cases")
    ax.set_xlabel("Prediction Confidence (Max Softmax Probability)")
    ax.set_ylabel("Density")
    ax.set_xlim(0.0, 1.02)
    ax.legend()
    ax.grid(True, alpha=0.3)
    return _finalize_figure(fig, save_path=save_path, show=show)


def plot_entropy_distribution(
    result: EntropyAnalysisResult,
    save_path: str | Path | None = None,
    show: bool = False,
) -> Figure:
    """Plot predictive Shannon entropy histograms split by inter-expert agreement."""
    fig, ax = plt.subplots(figsize=(8, 5))
    if len(result.entropies_consensus) > 0:
        ax.hist(
            result.entropies_consensus,
            bins=25,
            alpha=0.55,
            density=True,
            color="tab:green",
            label=f"Consensus (mean={result.mean_entropy_consensus:.3f})",
        )
    if len(result.entropies_disagreement) > 0:
        ax.hist(
            result.entropies_disagreement,
            bins=25,
            alpha=0.55,
            density=True,
            color="tab:orange",
            label=f"Disagreement (mean={result.mean_entropy_disagreement:.3f})",
        )
    ax.set_title("Predictive Entropy Distribution")
    ax.set_xlabel("Shannon Entropy (nats)")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return _finalize_figure(fig, save_path=save_path, show=show)


def extract_latent_bundle(
    model: tv_models.ConvNeXt,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    max_samples: int | None = 2000,
    include_upstream: bool = False,
) -> LatentFeatureBundle:
    """Extract ConvNeXt embeddings and prediction confidences in a single forward pass."""
    model.eval()
    upstream_list: list[npt.NDArray[np.float32]] = []
    downstream_list: list[npt.NDArray[np.float32]] = []
    confidences_list: list[npt.NDArray[np.float64]] = []
    collected = 0

    with torch.no_grad():
        for inputs, _ in loader:
            inputs = inputs.to(device, non_blocking=(device.type == "cuda"))
            feat_maps, embeddings, logits = extract_convnext_features_and_logits(model, inputs)
            probs = torch.softmax(logits, dim=1)
            max_probs, _ = torch.max(probs, dim=1)

            if include_upstream:
                flat_up = feat_maps.reshape(feat_maps.size(0), -1)
                upstream_list.append(flat_up.cpu().numpy().astype(np.float32))
            downstream_list.append(embeddings.cpu().numpy().astype(np.float32))
            confidences_list.append(max_probs.cpu().numpy().astype(np.float64))

            collected += int(inputs.size(0))
            if max_samples is not None and collected >= max_samples:
                break

    if not downstream_list:
        raise ValueError("Cannot extract features from an empty DataLoader.")

    downstream = np.concatenate(downstream_list, axis=0)
    confidences = np.concatenate(confidences_list, axis=0)
    upstream = np.concatenate(upstream_list, axis=0) if include_upstream else None

    if max_samples is not None:
        downstream = downstream[:max_samples]
        confidences = confidences[:max_samples]
        if upstream is not None:
            upstream = upstream[:max_samples]

    return LatentFeatureBundle(
        upstream_features=upstream,
        downstream_features=downstream,
        confidences=confidences,
    )


def _compute_tsne_projection(
    features: npt.NDArray[np.float32],
    seed: int = 42,
    pca_max_dim: int = 50,
) -> npt.NDArray[np.float64]:
    """Compute a 2D t-SNE projection with automatic PCA pre-reduction for high dimensions."""
    n_samples, n_features = features.shape
    if n_samples < 3:
        raise ValueError(f"At least 3 samples are required for t-SNE, got {n_samples}.")

    reduced: npt.NDArray[np.float32] = features
    if n_features > pca_max_dim and n_samples > 2:
        n_components = min(pca_max_dim, n_samples - 1, n_features)
        pca = PCA(n_components=n_components, random_state=seed)
        reduced = np.asarray(pca.fit_transform(features), dtype=np.float32)

    perplexity = min(30.0, max(2.0, (n_samples - 1) / 3.0))
    init_mode = "pca" if reduced.shape[1] >= 2 else "random"
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=seed,
        init=init_mode,
        learning_rate="auto",
    )
    return np.asarray(tsne.fit_transform(reduced), dtype=np.float64)


def plot_tsne_confidence(
    bundle: LatentFeatureBundle,
    expert1_labels: Sequence[str],
    expert2_labels: Sequence[str],
    seed: int = 42,
    save_path: str | Path | None = None,
    show: bool = False,
) -> Figure:
    """Plot 2D t-SNE of penultimate embeddings colored by model confidence."""
    n_points = len(bundle.downstream_features)
    exp1 = np.asarray(expert1_labels[:n_points])
    exp2 = np.asarray(expert2_labels[:n_points])
    agreement_mask = exp1 == exp2

    embedded = _compute_tsne_projection(bundle.downstream_features, seed=seed)

    fig, ax = plt.subplots(figsize=(11, 8))
    sc_consensus = ax.scatter(
        embedded[agreement_mask, 0],
        embedded[agreement_mask, 1],
        c=bundle.confidences[agreement_mask],
        cmap="RdYlGn",
        marker="o",
        s=65,
        alpha=0.8,
        edgecolors="white",
        linewidths=0.6,
        vmin=0.4,
        vmax=1.0,
        label="Group A: Consensus",
    )
    ax.scatter(
        embedded[~agreement_mask, 0],
        embedded[~agreement_mask, 1],
        c=bundle.confidences[~agreement_mask],
        cmap="RdYlGn",
        marker="X",
        s=90,
        alpha=1.0,
        edgecolors="black",
        linewidths=0.5,
        vmin=0.4,
        vmax=1.0,
        label="Group B: Disagreement",
    )

    cbar = fig.colorbar(sc_consensus, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Prediction Confidence (Max Probability)", rotation=270, labelpad=18)
    ax.set_title("Latent Space (t-SNE) Colored by Model Confidence")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.35)
    return _finalize_figure(fig, save_path=save_path, show=show)


def plot_tsne_dual_expert(
    bundle: LatentFeatureBundle,
    expert1_labels: Sequence[str],
    expert2_labels: Sequence[str],
    classes: Sequence[str],
    seed: int = 42,
    save_path: str | Path | None = None,
    show: bool = False,
) -> Figure:
    """Plot side-by-side t-SNE embeddings colored by Expert 1 and Expert 2 labels."""
    n_points = len(bundle.downstream_features)
    exp1 = list(expert1_labels[:n_points])
    exp2 = list(expert2_labels[:n_points])
    markers_col = [
        "Agreement" if a == b else "Disagreement" for a, b in zip(exp1, exp2, strict=True)
    ]
    embedded = _compute_tsne_projection(bundle.downstream_features, seed=seed)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7.5))
    common_kwargs: dict[str, Any] = {
        "x": embedded[:, 0],
        "y": embedded[:, 1],
        "style": markers_col,
        "markers": {"Agreement": "o", "Disagreement": "X"},
        "hue_order": list(classes),
        "palette": "tab10",
        "s": 55,
        "alpha": 0.75,
    }

    sns.scatterplot(hue=exp1, ax=axes[0], **common_kwargs)
    axes[0].set_title("t-SNE Embeddings Colored by Expert 1")
    axes[0].grid(True, linestyle="--", alpha=0.3)

    sns.scatterplot(hue=exp2, ax=axes[1], legend=False, **common_kwargs)
    axes[1].set_title("t-SNE Embeddings Colored by Expert 2")
    axes[1].grid(True, linestyle="--", alpha=0.3)

    return _finalize_figure(fig, save_path=save_path, show=show)


def plot_gradcam_explanation(
    model: tv_models.ConvNeXt,
    dataset: MitosisDataset,
    idx: int,
    device: torch.device,
    save_path: str | Path | None = None,
    show: bool = False,
) -> Figure:
    """Generate a side-by-side original patch and Grad-CAM++ attribution map.

    Gracefully falls back to displaying the image and predicted probabilities if the
    optional ``grad-cam`` package is not installed.
    """
    model.eval()
    img_tensor, label_target = dataset[idx]
    img_display = denormalize_image(img_tensor)
    input_tensor = img_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        pred_probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    gt_text = format_prediction_text(label_target, dataset.classes)
    pred_text = format_prediction_text(pred_probs, dataset.classes)
    target_class = int(np.argmax(pred_probs))

    if not HAS_GRAD_CAM or GradCAMPlusPlus is None or show_cam_on_image is None:
        logger.warning(
            "Optional package 'grad-cam' is not installed; rendering prediction "
            "without CAM overlay."
        )
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(img_display)
        ax.set_title(
            f"Sample #{idx}\nTruth: {gt_text}\nPred: {pred_text}",
            fontsize=10,
            loc="left",
        )
        ax.axis("off")
        return _finalize_figure(fig, save_path=save_path, show=show)

    target_layers = [model.features[-1][-1]]
    grad_input = input_tensor.detach().requires_grad_(True)
    with torch.set_grad_enabled(True):
        with GradCAMPlusPlus(model=model, target_layers=target_layers) as cam:
            grayscale_cam = cam(input_tensor=grad_input, targets=None)[0, :]

    overlay = show_cam_on_image(img_display, grayscale_cam, use_rgb=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    axes[0].imshow(img_display)
    axes[0].set_title(f"Sample #{idx} - Ground Truth\n{gt_text}", fontsize=10, loc="left")
    axes[0].axis("off")

    axes[1].imshow(overlay)
    axes[1].set_title(
        f"Grad-CAM++ (Focus: {dataset.classes[target_class]})\n{pred_text}",
        fontsize=10,
        loc="left",
    )
    axes[1].axis("off")

    return _finalize_figure(fig, save_path=save_path, show=show)
