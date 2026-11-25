"""Pure evaluation metrics for probabilistic soft-labeling and expert disagreement."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import entropy as scipy_entropy
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader


@dataclass(frozen=True)
class InferenceOutputs:
    """Aggregated model predictions collected over a DataLoader."""

    probs: npt.NDArray[np.float64]
    preds_top1: npt.NDArray[np.int64]
    preds_top2: npt.NDArray[np.int64]
    confidences: npt.NDArray[np.float64]


@dataclass(frozen=True)
class GroupEvaluationResult:
    """Stratified performance and confidence calibration across Group A and Group B."""

    n_consensus: int
    n_disagreement: int
    consensus_accuracy: float
    consensus_mean_confidence: float
    consensus_std_confidence: float
    disagreement_double_coverage: float
    disagreement_relaxed_accuracy: float
    disagreement_mean_confidence: float
    disagreement_std_confidence: float
    confidences_consensus: npt.NDArray[np.float64]
    confidences_disagreement: npt.NDArray[np.float64]


@dataclass(frozen=True)
class ExpertComparisonResult:
    """Hard-decision evaluation against Expert 1, Expert 2, and inter-expert agreement."""

    classes: tuple[str, ...]
    accuracy_vs_expert1: float
    accuracy_vs_expert2: float
    balanced_accuracy_vs_expert1: float
    balanced_accuracy_vs_expert2: float
    macro_f1_vs_expert1: float
    macro_f1_vs_expert2: float
    inter_expert_agreement: float
    cm_vs_expert1: npt.NDArray[np.int64]
    cm_vs_expert2: npt.NDArray[np.int64]
    cm_inter_expert: npt.NDArray[np.int64]


@dataclass(frozen=True)
class Top2IndulgentResult:
    """Top-2 oracle evaluation separating near-misses from severe misclassifications."""

    classes: tuple[str, ...]
    top2_accuracy_vs_expert1: float
    top2_accuracy_vs_expert2: float
    cm_vs_expert1: npt.NDArray[np.int64]
    cm_vs_expert2: npt.NDArray[np.int64]


@dataclass(frozen=True)
class EntropyAnalysisResult:
    """Predictive Shannon entropy split by inter-expert consensus vs. disagreement."""

    entropies_consensus: npt.NDArray[np.float64]
    entropies_disagreement: npt.NDArray[np.float64]
    mean_entropy_consensus: float
    mean_entropy_disagreement: float


def compute_bounded_accuracy_counts(
    logits: torch.Tensor,
    targets: torch.Tensor,
    low_threshold: float = 0.25,
    high_threshold_on_ambiguity: float = 0.75,
) -> tuple[int, int]:
    """Count predictions satisfying confidence-bounded accuracy rules.

    A prediction is counted as correct if and only if:
    1. Its predicted class received positive expert support (``target > 0``).
    2. Its maximum softmax probability is at least ``low_threshold``.
    3. On ambiguous samples (expert target < 0.9), confidence does not exceed
       ``high_threshold_on_ambiguity`` (penalizing overconfident guesses).

    Args:
        logits: Raw model logits of shape ``(B, C)``.
        targets: Soft probability targets ``(B, C)`` or hard class indices ``(B,)``.
        low_threshold: Minimum confidence required for a valid prediction.
        high_threshold_on_ambiguity: Maximum allowed confidence on ambiguous samples.

    Returns:
        Tuple ``(num_correct, batch_size)`` for exact sample-weighted aggregation.
    """
    if logits.ndim != 2 or logits.shape[1] < 2:
        raise ValueError(f"logits must have shape (B, C) with C >= 2, got {tuple(logits.shape)}")

    batch_size, num_classes = logits.shape
    if batch_size == 0:
        return 0, 0

    if targets.ndim == 1:
        target_probs = F.one_hot(targets.long(), num_classes=num_classes).to(dtype=logits.dtype)
    elif targets.ndim == 2 and targets.shape == logits.shape:
        target_probs = targets.to(dtype=logits.dtype)
    else:
        raise ValueError(
            f"targets must have shape ({batch_size},) or ({batch_size}, {num_classes}), "
            f"got {tuple(targets.shape)}"
        )

    probs = torch.softmax(logits, dim=1)
    confidences, pred_idx = torch.max(probs, dim=1)
    target_vals = target_probs.gather(1, pred_idx.unsqueeze(1)).squeeze(1)

    cond_valid = target_vals > 0.0
    cond_confident = confidences >= low_threshold
    is_pure_target = target_vals >= 0.9
    respects_ambiguity = is_pure_target | (confidences <= high_threshold_on_ambiguity)

    correct = cond_valid & cond_confident & respects_ambiguity
    return int(correct.sum().item()), batch_size


def bounded_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    low_threshold: float = 0.25,
    high_threshold_on_ambiguity: float = 0.75,
) -> float:
    """Compute bounded accuracy as a scalar ratio in ``[0, 1]``."""
    num_correct, total = compute_bounded_accuracy_counts(
        logits=logits,
        targets=targets,
        low_threshold=low_threshold,
        high_threshold_on_ambiguity=high_threshold_on_ambiguity,
    )
    if total == 0:
        return float("nan")
    return num_correct / total


def double_coverage_accuracy(
    top2_preds: npt.NDArray[np.int64],
    expert1_labels: npt.NDArray[np.int64],
    expert2_labels: npt.NDArray[np.int64],
) -> float:
    """Compute Double Coverage Accuracy (DCA) on inter-expert disagreement cases.

    DCA verifies whether the unordered set of the model's top-2 predicted classes
    exactly matches the unordered pair of distinct expert annotations:
    ``{y_hat_1, y_hat_2} == {y_E1, y_E2}``.

    Args:
        top2_preds: Integer array of shape ``(N, 2)`` containing top-2 class indices.
        expert1_labels: Integer array of shape ``(N,)`` with Expert 1 class indices.
        expert2_labels: Integer array of shape ``(N,)`` with Expert 2 class indices.

    Returns:
        DCA score in ``[0, 1]``, or ``nan`` if ``N == 0``.

    Raises:
        ValueError: If array shapes mismatch or any sample has ``expert1 == expert2``.
    """
    top2 = np.asarray(top2_preds, dtype=np.int64)
    exp1 = np.asarray(expert1_labels, dtype=np.int64)
    exp2 = np.asarray(expert2_labels, dtype=np.int64)

    if top2.ndim != 2 or top2.shape[1] != 2:
        raise ValueError(f"top2_preds must have shape (N, 2), got {top2.shape}")
    if exp1.ndim != 1 or exp2.ndim != 1 or not (len(top2) == len(exp1) == len(exp2)):
        raise ValueError(
            "Length mismatch between top2_preds, expert1_labels, and expert2_labels: "
            f"{len(top2)}, {len(exp1)}, {len(exp2)}"
        )
    if len(top2) == 0:
        return float("nan")
    if bool(np.any(exp1 == exp2)):
        raise ValueError(
            "double_coverage_accuracy is defined only for discordant expert pairs "
            "(expert1_labels != expert2_labels)."
        )

    exact_match = ((top2[:, 0] == exp1) & (top2[:, 1] == exp2)) | (
        (top2[:, 0] == exp2) & (top2[:, 1] == exp1)
    )
    return float(np.mean(exact_match))


def soft_confusion_matrix(
    y_true_soft: npt.NDArray[np.float64] | torch.Tensor,
    y_pred_probs: npt.NDArray[np.float64] | torch.Tensor,
    normalize: bool = True,
) -> npt.NDArray[np.float64]:
    """Compute probabilistic soft confusion matrix via ``Y_true^T @ Y_pred``.

    Args:
        y_true_soft: Ground-truth probability distributions of shape ``(N, C)``.
        y_pred_probs: Model predicted probability distributions of shape ``(N, C)``.
        normalize: If True, normalize each row by total ground-truth class mass.

    Returns:
        Float64 matrix of shape ``(C, C)``.
    """
    y_true = (
        y_true_soft.detach().cpu().numpy().astype(np.float64)
        if isinstance(y_true_soft, torch.Tensor)
        else np.asarray(y_true_soft, dtype=np.float64)
    )
    y_pred = (
        y_pred_probs.detach().cpu().numpy().astype(np.float64)
        if isinstance(y_pred_probs, torch.Tensor)
        else np.asarray(y_pred_probs, dtype=np.float64)
    )

    if y_true.ndim != 2 or y_pred.ndim != 2 or y_true.shape != y_pred.shape:
        raise ValueError(
            f"y_true_soft and y_pred_probs must be 2D arrays with identical shape, "
            f"got {y_true.shape} and {y_pred.shape}"
        )

    cm_soft: npt.NDArray[np.float64] = y_true.T @ y_pred
    if not normalize:
        return cm_soft

    row_mass = y_true.sum(axis=0, keepdims=True).T
    return np.divide(
        cm_soft,
        row_mass,
        out=np.zeros_like(cm_soft, dtype=np.float64),
        where=row_mass > 0.0,
    )


def normalize_confusion_matrix(cm: npt.NDArray[np.int64]) -> npt.NDArray[np.float64]:
    """Row-normalize an integer confusion matrix safely (zero-sum rows stay zero)."""
    cm_float = np.asarray(cm, dtype=np.float64)
    row_sums = cm_float.sum(axis=1, keepdims=True)
    return np.divide(
        cm_float,
        row_sums,
        out=np.zeros_like(cm_float, dtype=np.float64),
        where=row_sums > 0.0,
    )


def compute_prediction_entropies(probs: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Compute per-sample Shannon entropy (in nats) of predicted class probabilities."""
    probs_arr = np.asarray(probs, dtype=np.float64)
    if probs_arr.ndim != 2:
        raise ValueError(f"probs must be 2D (N, C), got shape {probs_arr.shape}")
    clipped = np.clip(probs_arr, 1e-12, 1.0)
    return np.asarray(scipy_entropy(clipped, axis=1), dtype=np.float64)


def collect_predictions(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> InferenceOutputs:
    """Run inference over a non-shuffled DataLoader and gather probability outputs.

    Args:
        model: Trained PyTorch classification model.
        loader: Evaluation DataLoader (must have ``shuffle=False``).
        device: Computation device.

    Returns:
        InferenceOutputs containing softmax probabilities, top-1, top-2, and confidences.
    """
    model.eval()
    probs_batches: list[npt.NDArray[np.float64]] = []
    top2_batches: list[npt.NDArray[np.int64]] = []
    conf_batches: list[npt.NDArray[np.float64]] = []

    with torch.no_grad():
        for inputs, _ in loader:
            inputs = inputs.to(device, non_blocking=(device.type == "cuda"))
            logits = model(inputs)
            probs = torch.softmax(logits, dim=1)
            confidences, _ = torch.max(probs, dim=1)
            _, top2 = torch.topk(probs, k=2, dim=1)

            probs_batches.append(probs.cpu().numpy().astype(np.float64))
            top2_batches.append(top2.cpu().numpy().astype(np.int64))
            conf_batches.append(confidences.cpu().numpy().astype(np.float64))

    if not probs_batches:
        raise ValueError("Cannot collect predictions from an empty DataLoader.")

    all_probs = np.concatenate(probs_batches, axis=0)
    all_top2 = np.concatenate(top2_batches, axis=0)
    all_confidences = np.concatenate(conf_batches, axis=0)
    all_top1 = all_top2[:, 0].copy()

    return InferenceOutputs(
        probs=all_probs,
        preds_top1=all_top1,
        preds_top2=all_top2,
        confidences=all_confidences,
    )


def evaluate_groups(
    outputs: InferenceOutputs,
    expert1_labels: npt.NDArray[np.int64],
    expert2_labels: npt.NDArray[np.int64],
) -> GroupEvaluationResult:
    """Evaluate consensus accuracy (Group A) and ambiguity resolution/calibration (Group B)."""
    exp1 = np.asarray(expert1_labels, dtype=np.int64)
    exp2 = np.asarray(expert2_labels, dtype=np.int64)
    n_samples = len(outputs.preds_top1)
    if len(exp1) != n_samples or len(exp2) != n_samples:
        raise ValueError(
            f"Sample count mismatch: outputs={n_samples}, exp1={len(exp1)}, exp2={len(exp2)}"
        )

    mask_a = exp1 == exp2
    mask_b = ~mask_a
    n_a = int(mask_a.sum())
    n_b = int(mask_b.sum())

    conf_a = outputs.confidences[mask_a]
    conf_b = outputs.confidences[mask_b]

    if n_a > 0:
        acc_a = float(accuracy_score(exp1[mask_a], outputs.preds_top1[mask_a]))
        mean_conf_a = float(np.mean(conf_a))
        std_conf_a = float(np.std(conf_a))
    else:
        acc_a = float("nan")
        mean_conf_a = float("nan")
        std_conf_a = float("nan")

    if n_b > 0:
        dca_b = double_coverage_accuracy(outputs.preds_top2[mask_b], exp1[mask_b], exp2[mask_b])
        preds_b = outputs.preds_top1[mask_b]
        relaxed_b = float(np.mean((preds_b == exp1[mask_b]) | (preds_b == exp2[mask_b])))
        mean_conf_b = float(np.mean(conf_b))
        std_conf_b = float(np.std(conf_b))
    else:
        dca_b = float("nan")
        relaxed_b = float("nan")
        mean_conf_b = float("nan")
        std_conf_b = float("nan")

    return GroupEvaluationResult(
        n_consensus=n_a,
        n_disagreement=n_b,
        consensus_accuracy=acc_a,
        consensus_mean_confidence=mean_conf_a,
        consensus_std_confidence=std_conf_a,
        disagreement_double_coverage=dca_b,
        disagreement_relaxed_accuracy=relaxed_b,
        disagreement_mean_confidence=mean_conf_b,
        disagreement_std_confidence=std_conf_b,
        confidences_consensus=conf_a,
        confidences_disagreement=conf_b,
    )


def evaluate_against_experts(
    preds_top1: npt.NDArray[np.int64],
    expert1_labels: npt.NDArray[np.int64],
    expert2_labels: npt.NDArray[np.int64],
    classes: Sequence[str],
) -> ExpertComparisonResult:
    """Compute classification metrics and confusion matrices against both experts."""
    preds = np.asarray(preds_top1, dtype=np.int64)
    exp1 = np.asarray(expert1_labels, dtype=np.int64)
    exp2 = np.asarray(expert2_labels, dtype=np.int64)
    class_tuple = tuple(classes)
    label_indices = list(range(len(class_tuple)))

    cm1 = np.asarray(confusion_matrix(exp1, preds, labels=label_indices), dtype=np.int64)
    cm2 = np.asarray(confusion_matrix(exp2, preds, labels=label_indices), dtype=np.int64)
    cm_inter = np.asarray(confusion_matrix(exp1, exp2, labels=label_indices), dtype=np.int64)

    return ExpertComparisonResult(
        classes=class_tuple,
        accuracy_vs_expert1=float(accuracy_score(exp1, preds)),
        accuracy_vs_expert2=float(accuracy_score(exp2, preds)),
        balanced_accuracy_vs_expert1=float(balanced_accuracy_score(exp1, preds)),
        balanced_accuracy_vs_expert2=float(balanced_accuracy_score(exp2, preds)),
        macro_f1_vs_expert1=float(f1_score(exp1, preds, average="macro", zero_division=0)),
        macro_f1_vs_expert2=float(f1_score(exp2, preds, average="macro", zero_division=0)),
        inter_expert_agreement=float(accuracy_score(exp1, exp2)),
        cm_vs_expert1=cm1,
        cm_vs_expert2=cm2,
        cm_inter_expert=cm_inter,
    )


def evaluate_top2_indulgent(
    preds_top2: npt.NDArray[np.int64],
    expert1_labels: npt.NDArray[np.int64],
    expert2_labels: npt.NDArray[np.int64],
    classes: Sequence[str],
) -> Top2IndulgentResult:
    """Evaluate Top-2 oracle confusion matrices against Expert 1 and Expert 2."""
    top2 = np.asarray(preds_top2, dtype=np.int64)
    exp1 = np.asarray(expert1_labels, dtype=np.int64)
    exp2 = np.asarray(expert2_labels, dtype=np.int64)
    class_tuple = tuple(classes)
    label_indices = list(range(len(class_tuple)))

    in_top2_e1 = (top2[:, 0] == exp1) | (top2[:, 1] == exp1)
    in_top2_e2 = (top2[:, 0] == exp2) | (top2[:, 1] == exp2)

    corrected_e1 = np.where(in_top2_e1, exp1, top2[:, 0])
    corrected_e2 = np.where(in_top2_e2, exp2, top2[:, 0])

    cm1 = np.asarray(confusion_matrix(exp1, corrected_e1, labels=label_indices), dtype=np.int64)
    cm2 = np.asarray(confusion_matrix(exp2, corrected_e2, labels=label_indices), dtype=np.int64)

    return Top2IndulgentResult(
        classes=class_tuple,
        top2_accuracy_vs_expert1=float(np.mean(in_top2_e1)),
        top2_accuracy_vs_expert2=float(np.mean(in_top2_e2)),
        cm_vs_expert1=cm1,
        cm_vs_expert2=cm2,
    )


def evaluate_entropy_by_agreement(
    probs: npt.NDArray[np.float64],
    expert1_labels: npt.NDArray[np.int64],
    expert2_labels: npt.NDArray[np.int64],
) -> EntropyAnalysisResult:
    """Compare predictive entropy distributions across consensus vs. disagreement cases."""
    entropies = compute_prediction_entropies(probs)
    exp1 = np.asarray(expert1_labels, dtype=np.int64)
    exp2 = np.asarray(expert2_labels, dtype=np.int64)
    mask_a = exp1 == exp2

    ent_a = entropies[mask_a]
    ent_b = entropies[~mask_a]

    return EntropyAnalysisResult(
        entropies_consensus=ent_a,
        entropies_disagreement=ent_b,
        mean_entropy_consensus=float(np.mean(ent_a)) if len(ent_a) > 0 else float("nan"),
        mean_entropy_disagreement=float(np.mean(ent_b)) if len(ent_b) > 0 else float("nan"),
    )
