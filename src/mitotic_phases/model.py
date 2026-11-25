"""ConvNeXt model factory, freezing policies, and single-pass feature extraction."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn
from torchvision import models as tv_models

from mitotic_phases.config import ModelArchitecture, ModelConfig

logger = logging.getLogger(__name__)

_CONVNEXT_REGISTRY: dict[ModelArchitecture, tuple[Callable[..., tv_models.ConvNeXt], Any]] = {
    "convnext_tiny": (tv_models.convnext_tiny, tv_models.ConvNeXt_Tiny_Weights.DEFAULT),
    "convnext_small": (tv_models.convnext_small, tv_models.ConvNeXt_Small_Weights.DEFAULT),
    "convnext_base": (tv_models.convnext_base, tv_models.ConvNeXt_Base_Weights.DEFAULT),
}


def build_convnext(config: ModelConfig, num_classes: int) -> tv_models.ConvNeXt:
    """Instantiate a ConvNeXt backbone with the configured transfer-learning policy.

    Args:
        config: Immutable model configuration specifying architecture, weights, and
            freezing mode (``'feature_extraction'``, ``'fine_tune'``, or ``'full'``).
        num_classes: Number of target mitotic classes.

    Returns:
        Configured ``ConvNeXt`` model with replaced linear classification head.
    """
    if num_classes < 2:
        raise ValueError(f"num_classes must be >= 2, got {num_classes}")

    builder, default_weights = _CONVNEXT_REGISTRY[config.architecture]
    weights = default_weights if config.pretrained else None
    model = builder(weights=weights)

    if config.train_mode == "feature_extraction":
        for param in model.features.parameters():
            param.requires_grad = False
    elif config.train_mode == "fine_tune":
        for name, child in model.features.named_children():
            trainable = name == "7"  # Unfreeze Stage 4 only
            for param in child.parameters():
                param.requires_grad = trainable
    else:  # "full"
        for param in model.parameters():
            param.requires_grad = True

    final_layer = model.classifier[2]
    if not isinstance(final_layer, nn.Linear):
        raise TypeError(
            f"Expected ConvNeXt classifier[2] to be nn.Linear, got {type(final_layer).__name__}"
        )
    model.classifier[2] = nn.Linear(final_layer.in_features, num_classes)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "Initialized %s (pretrained=%s, mode=%s): %d / %d trainable parameters.",
        config.architecture,
        config.pretrained,
        config.train_mode,
        trainable_params,
        total_params,
    )
    return model


def get_trainable_parameters(model: nn.Module) -> list[nn.Parameter]:
    """Return parameters with ``requires_grad=True`` for optimizer construction."""
    return [param for param in model.parameters() if param.requires_grad]


def extract_convnext_features_and_logits(
    model: tv_models.ConvNeXt,
    inputs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run a single forward pass returning spatial feature maps, embeddings, and logits.

    Args:
        model: ConvNeXt model instance.
        inputs: Input image batch of shape ``(B, 3, H, W)``.

    Returns:
        Tuple ``(feature_maps, embeddings, logits)``.
    """
    feature_maps: torch.Tensor = model.features(inputs)
    pooled: torch.Tensor = model.avgpool(feature_maps)
    normed: torch.Tensor = model.classifier[0](pooled)
    embeddings: torch.Tensor = model.classifier[1](normed)
    logits: torch.Tensor = model.classifier[2](embeddings)
    return feature_maps, embeddings, logits
