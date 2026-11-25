"""Deterministic seeding utilities for Python, NumPy, PyTorch CUDA, and DataLoader workers."""

from __future__ import annotations

import logging
import os
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Synchronize all RNG backends and optionally enforce deterministic CUDA kernels.

    Args:
        seed: Non-negative global random seed.
        deterministic: If True, disables cuDNN auto-tuner, sets CUBLAS workspace
            configuration, and enables deterministic PyTorch algorithms.
    """
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")

    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic

    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        logger.info("Deterministic execution enabled with global seed=%d.", seed)
    else:
        torch.use_deterministic_algorithms(False)
        logger.info("Seeded RNG backends with seed=%d (non-deterministic fast mode).", seed)


def seed_worker(worker_id: int) -> None:
    """DataLoader worker_init_fn ensuring distinct, reproducible RNG state per worker.

    Args:
        worker_id: Worker process index assigned by PyTorch DataLoader.
    """
    del worker_id  # PyTorch already derives a unique initial seed per worker.
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int) -> torch.Generator:
    """Create a seeded torch.Generator for reproducible DataLoader shuffling/sampling.

    Args:
        seed: Non-negative seed value.

    Returns:
        Seeded torch.Generator instance.
    """
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator
