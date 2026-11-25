"""Unit tests for soft labels, Double Coverage Accuracy, bounded accuracy, and calibration."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from mitotic_phases.data import encode_soft_label
from mitotic_phases.metrics import (
    InferenceOutputs,
    bounded_accuracy,
    compute_bounded_accuracy_counts,
    double_coverage_accuracy,
    evaluate_against_experts,
    evaluate_entropy_by_agreement,
    evaluate_groups,
    evaluate_top2_indulgent,
    soft_confusion_matrix,
)


def test_encode_soft_label_consensus_and_disagreement() -> None:
    """Consensus yields one-hot; disagreement yields bimodal 0.5/0.5 distribution."""
    consensus = encode_soft_label(1, 1, num_classes=4)
    assert torch.allclose(consensus, torch.tensor([0.0, 1.0, 0.0, 0.0]))

    bimodal = encode_soft_label(1, 2, num_classes=4)
    assert torch.allclose(bimodal, torch.tensor([0.0, 0.5, 0.5, 0.0]))
    assert math.isclose(float(bimodal.sum().item()), 1.0, rel_tol=1e-6)


def test_encode_soft_label_out_of_range_raises() -> None:
    """Out-of-bounds class indices must raise ValueError."""
    with pytest.raises(ValueError):
        encode_soft_label(0, 4, num_classes=4)


def test_double_coverage_accuracy_order_invariant() -> None:
    """DCA matches unordered pairs {y_E1, y_E2} == {y_hat_1, y_hat_2}."""
    top2 = np.array(
        [
            [1, 2],  # exact match (same order)
            [2, 1],  # exact match (swapped order)
            [1, 3],  # only Expert 1 matched -> miss
            [0, 3],  # neither matched -> miss
        ],
        dtype=np.int64,
    )
    exp1 = np.array([1, 1, 1, 1], dtype=np.int64)
    exp2 = np.array([2, 2, 2, 2], dtype=np.int64)

    dca = double_coverage_accuracy(top2, exp1, exp2)
    assert math.isclose(dca, 0.5, rel_tol=1e-6)


def test_double_coverage_accuracy_rejects_consensus_samples() -> None:
    """Passing identical expert labels to DCA must raise ValueError."""
    top2 = np.array([[0, 1]], dtype=np.int64)
    with pytest.raises(ValueError, match="discordant"):
        double_coverage_accuracy(top2, np.array([0]), np.array([0]))


def test_bounded_accuracy_penalizes_overconfidence_on_ambiguity() -> None:
    """Overconfident predictions (>0.75) on bimodal targets must be marked incorrect."""
    # Sample 0: consensus class 0, high confidence (~0.95) -> correct
    # Sample 1: ambiguous (0, 1), moderate confidence (~0.55 on class 0) -> correct
    # Sample 2: ambiguous (0, 1), overconfident (~0.95 on class 0) -> incorrect
    # Sample 3: consensus class 2, predicts class 0 -> incorrect
    probs = torch.tensor(
        [
            [0.95, 0.02, 0.02, 0.01],
            [0.55, 0.40, 0.03, 0.02],
            [0.95, 0.03, 0.01, 0.01],
            [0.80, 0.10, 0.05, 0.05],
        ],
        dtype=torch.float32,
    )
    logits = torch.log(probs)
    targets = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0, 0.0],
            [0.5, 0.5, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )

    correct, total = compute_bounded_accuracy_counts(
        logits, targets, low_threshold=0.25, high_threshold_on_ambiguity=0.75
    )
    assert total == 4
    assert correct == 2
    assert math.isclose(bounded_accuracy(logits, targets), 0.5, rel_tol=1e-6)


def test_bounded_accuracy_handles_batch_size_one_and_hard_targets() -> None:
    """Batch size 1 must not trigger squeeze dimension collapse."""
    logits = torch.log(torch.tensor([[0.1, 0.85, 0.05]], dtype=torch.float32))
    hard_target = torch.tensor([1], dtype=torch.long)

    correct, total = compute_bounded_accuracy_counts(logits, hard_target)
    assert (correct, total) == (1, 1)


def test_soft_confusion_matrix_normalization_and_zero_mass_safety() -> None:
    """Normalized soft confusion matrix rows with positive mass sum to 1 without NaNs."""
    y_true = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
        ],
        dtype=np.float64,
    )
    y_pred = np.array(
        [
            [0.8, 0.2, 0.0],
            [0.4, 0.6, 0.0],
        ],
        dtype=np.float64,
    )

    cm = soft_confusion_matrix(y_true, y_pred, normalize=True)
    assert cm.shape == (3, 3)
    assert not np.isnan(cm).any()
    assert math.isclose(float(cm[0].sum()), 1.0, rel_tol=1e-6)
    assert math.isclose(float(cm[1].sum()), 1.0, rel_tol=1e-6)
    assert math.isclose(float(cm[2].sum()), 0.0, abs_tol=1e-12)


def test_group_expert_top2_and_entropy_evaluations() -> None:
    """End-to-end verification of Group A/B, expert comparison, top-2 oracle, and entropy."""
    probs = np.array(
        [
            [0.94, 0.03, 0.02, 0.01],  # Group A: consensus (0, 0), pred 0 -> hit
            [0.02, 0.92, 0.04, 0.02],  # Group A: consensus (1, 1), pred 1 -> hit
            [0.52, 0.44, 0.02, 0.02],  # Group B: disagree (0, 1), top2=(0, 1) -> DCA hit
            [0.48, 0.04, 0.44, 0.04],  # Group B: disagree (1, 2), top2=(0, 2) -> DCA miss
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
    exp1 = np.array([0, 1, 0, 1], dtype=np.int64)
    exp2 = np.array([0, 1, 1, 2], dtype=np.int64)
    classes = ("anaphase", "metaphase", "prometaphase", "ring")

    grp = evaluate_groups(outputs, exp1, exp2)
    assert grp.n_consensus == 2
    assert grp.n_disagreement == 2
    assert math.isclose(grp.consensus_accuracy, 1.0)
    assert math.isclose(grp.disagreement_double_coverage, 0.5)
    assert grp.consensus_mean_confidence > grp.disagreement_mean_confidence

    exp_res = evaluate_against_experts(outputs.preds_top1, exp1, exp2, classes)
    assert math.isclose(exp_res.inter_expert_agreement, 0.5)
    assert exp_res.cm_vs_expert1.shape == (4, 4)

    top2_res = evaluate_top2_indulgent(outputs.preds_top2, exp1, exp2, classes)
    assert math.isclose(top2_res.top2_accuracy_vs_expert2, 1.0)

    ent_res = evaluate_entropy_by_agreement(probs, exp1, exp2)
    assert ent_res.mean_entropy_disagreement > ent_res.mean_entropy_consensus
