"""Modular PyTorch training loop with CPU weight snapshots and metadata-rich checkpoints."""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from mitotic_phases.config import LabelMode, TrainingConfig
from mitotic_phases.metrics import compute_bounded_accuracy_counts
from mitotic_phases.model import get_trainable_parameters

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EpochMetrics:
    """Metrics recorded at the end of a single training epoch."""

    epoch: int
    train_loss: float
    val_loss: float
    train_metric: float
    val_metric: float
    metric_name: str


@dataclass(frozen=True)
class TrainingResult:
    """Summary returned after completing a training run."""

    best_epoch: int
    best_val_loss: float
    checkpoint_path: Path | None
    history: tuple[EpochMetrics, ...]


def snapshot_state_dict_to_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    """Clone model weights onto CPU RAM to avoid duplicating tensors in GPU VRAM."""
    return {key: tensor.detach().cpu().clone() for key, tensor in model.state_dict().items()}


def save_checkpoint(
    path: str | Path,
    model_state_cpu: Mapping[str, torch.Tensor],
    epoch: int,
    best_val_loss: float,
    label_mode: LabelMode,
    classes: Sequence[str] | None = None,
    class_to_idx: Mapping[str, int] | None = None,
    architecture: str | None = None,
    train_mode: str | None = None,
) -> Path:
    """Persist CPU weights and primitive metadata compatible with ``weights_only=True``."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "model_state_dict": dict(model_state_cpu),
        "epoch": int(epoch),
        "best_val_loss": float(best_val_loss),
        "label_mode": str(label_mode),
        "classes": list(classes) if classes is not None else [],
        "class_to_idx": dict(class_to_idx) if class_to_idx is not None else {},
        "architecture": str(architecture) if architecture is not None else "",
        "train_mode": str(train_mode) if train_mode is not None else "",
    }
    torch.save(payload, out_path)
    return out_path


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str | Path,
    device: torch.device,
) -> dict[str, Any]:
    """Load saved weights into ``model`` safely using ``weights_only=True``.

    Supports both structured payloads produced by :func:`save_checkpoint` and
    raw ``state_dict`` files.

    Args:
        model: Target PyTorch module.
        checkpoint_path: Path to the ``.pth`` / ``.pt`` weights file.
        device: Device onto which tensors are mapped.

    Returns:
        Dictionary of checkpoint metadata (empty dict for raw ``state_dict`` files).

    Raises:
        FileNotFoundError: If ``checkpoint_path`` does not exist.
        TypeError: If the loaded object is not a mapping.
    """
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {path}")

    loaded: Any = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(loaded, dict):
        raise TypeError(f"Expected checkpoint mapping, got {type(loaded).__name__}")

    if "model_state_dict" in loaded:
        state_dict = cast(dict[str, torch.Tensor], loaded["model_state_dict"])
        metadata: dict[str, Any] = {str(k): v for k, v in loaded.items() if k != "model_state_dict"}
    else:
        state_dict = cast(dict[str, torch.Tensor], loaded)
        metadata = {}

    model.load_state_dict(state_dict)
    logger.info("Loaded model weights from %s onto %s.", path, device)
    return metadata


class Trainer:
    """Encapsulated trainer supporting both probabilistic soft-label and hard-label regimes."""

    def __init__(
        self,
        model: nn.Module,
        training_config: TrainingConfig,
        label_mode: LabelMode,
        device: torch.device,
        optimizer: Optimizer | None = None,
        criterion: nn.Module | None = None,
        scheduler: LRScheduler | None = None,
    ) -> None:
        self.model = model.to(device)
        self.config = training_config
        self.label_mode: LabelMode = label_mode
        self.device = device
        self.criterion: nn.Module = criterion if criterion is not None else nn.CrossEntropyLoss()
        self.scheduler = scheduler

        if optimizer is not None:
            self.optimizer = optimizer
        else:
            trainable_params = get_trainable_parameters(self.model)
            if not trainable_params:
                raise ValueError("Model has no trainable parameters (requires_grad=True).")
            self.optimizer = AdamW(
                trainable_params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )

        self.metric_name = "bounded_accuracy" if self.label_mode == "soft" else "balanced_accuracy"

    def _run_epoch(
        self,
        loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
        is_train: bool,
    ) -> tuple[float, float]:
        """Run one full pass over ``loader`` and return sample-weighted loss and metric."""
        self.model.train(mode=is_train)

        running_loss = 0.0
        total_samples = 0
        total_bounded_correct = 0
        hard_preds: list[npt.NDArray[Any]] = []
        hard_targets: list[npt.NDArray[Any]] = []

        non_blocking = self.device.type == "cuda"

        for inputs, targets in loader:
            inputs = inputs.to(self.device, non_blocking=non_blocking)
            targets = targets.to(self.device, non_blocking=non_blocking)
            batch_size = int(inputs.size(0))

            if is_train:
                self.optimizer.zero_grad(set_to_none=True)

            with torch.set_grad_enabled(is_train):
                logits = self.model(inputs)
                loss = self.criterion(logits, targets)
                if is_train:
                    loss.backward()
                    self.optimizer.step()

            running_loss += float(loss.item()) * batch_size
            total_samples += batch_size

            with torch.no_grad():
                if self.label_mode == "soft":
                    batch_correct, _ = compute_bounded_accuracy_counts(
                        logits=logits,
                        targets=targets,
                        low_threshold=self.config.bounded_acc_low_threshold,
                        high_threshold_on_ambiguity=self.config.bounded_acc_high_threshold,
                    )
                    total_bounded_correct += batch_correct
                else:
                    preds = torch.argmax(logits, dim=1)
                    hard_preds.append(preds.cpu().numpy())
                    hard_targets.append(targets.cpu().numpy())

        if total_samples == 0:
            raise ValueError("Encountered an empty DataLoader during epoch execution.")

        avg_loss = running_loss / total_samples
        if self.label_mode == "soft":
            epoch_metric = total_bounded_correct / total_samples
        else:
            all_preds = np.concatenate(hard_preds, axis=0)
            all_targets = np.concatenate(hard_targets, axis=0)
            epoch_metric = float(balanced_accuracy_score(all_targets, all_preds))

        return avg_loss, epoch_metric

    def fit(
        self,
        dataloaders: dict[str, DataLoader[tuple[torch.Tensor, torch.Tensor]]],
        checkpoint_path: str | Path | None = None,
        classes: Sequence[str] | None = None,
        class_to_idx: Mapping[str, int] | None = None,
        architecture: str | None = None,
        train_mode: str | None = None,
    ) -> TrainingResult:
        """Train the model, persist best validation-loss weights, and restore them at finish.

        Args:
            dataloaders: Mapping containing ``'train'`` and ``'val'`` DataLoaders.
            checkpoint_path: Optional destination ``.pth`` path for best model weights.
            classes: Optional ordered class names stored in checkpoint metadata.
            class_to_idx: Optional class-to-index mapping stored in checkpoint metadata.
            architecture: Optional backbone name stored in checkpoint metadata.
            train_mode: Optional fine-tuning mode stored in checkpoint metadata.

        Returns:
            TrainingResult with best epoch metadata and full epoch history.
        """
        if "train" not in dataloaders or "val" not in dataloaders:
            raise ValueError("dataloaders must contain both 'train' and 'val' keys.")

        resolved_ckpt: Path | None = Path(checkpoint_path) if checkpoint_path is not None else None
        if resolved_ckpt is not None:
            resolved_ckpt.parent.mkdir(parents=True, exist_ok=True)

        best_val_loss = float("inf")
        best_epoch = 0
        best_state_cpu = snapshot_state_dict_to_cpu(self.model)
        history: list[EpochMetrics] = []

        start_time = time.monotonic()
        logger.info(
            "Starting %s-label training on %s for %d epochs.",
            self.label_mode,
            self.device,
            self.config.epochs,
        )

        for epoch in range(1, self.config.epochs + 1):
            train_loss, train_metric = self._run_epoch(dataloaders["train"], is_train=True)
            if self.scheduler is not None:
                self.scheduler.step()
            val_loss, val_metric = self._run_epoch(dataloaders["val"], is_train=False)

            record = EpochMetrics(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                train_metric=train_metric,
                val_metric=val_metric,
                metric_name=self.metric_name,
            )
            history.append(record)

            logger.info(
                "Epoch %d/%d | train_loss=%.4f | val_loss=%.4f | train_%s=%.4f | val_%s=%.4f",
                epoch,
                self.config.epochs,
                train_loss,
                val_loss,
                self.metric_name,
                train_metric,
                self.metric_name,
                val_metric,
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                best_state_cpu = snapshot_state_dict_to_cpu(self.model)
                if resolved_ckpt is not None:
                    save_checkpoint(
                        path=resolved_ckpt,
                        model_state_cpu=best_state_cpu,
                        epoch=best_epoch,
                        best_val_loss=best_val_loss,
                        label_mode=self.label_mode,
                        classes=classes,
                        class_to_idx=class_to_idx,
                        architecture=architecture,
                        train_mode=train_mode,
                    )
                    logger.info(
                        "Saved new best checkpoint (val_loss=%.4f) to %s.",
                        best_val_loss,
                        resolved_ckpt,
                    )

        elapsed = time.monotonic() - start_time
        logger.info(
            "Training complete in %.1fs. Best val_loss=%.4f at epoch %d.",
            elapsed,
            best_val_loss,
            best_epoch,
        )

        self.model.load_state_dict(best_state_cpu)
        return TrainingResult(
            best_epoch=best_epoch,
            best_val_loss=best_val_loss,
            checkpoint_path=resolved_ckpt if best_epoch > 0 else None,
            history=tuple(history),
        )
