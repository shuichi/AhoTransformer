"""Checkpoint utilities for AhoTransformer.

This module provides utilities for saving and loading model checkpoints.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    path: str | Path,
    metrics: dict[str, Any] | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    **kwargs: Any,
) -> None:
    """Save a training checkpoint.

    Args:
        model: The model to save.
        optimizer: The optimizer to save.
        epoch: Current epoch number.
        path: Path to save the checkpoint.
        metrics: Optional dictionary of metrics to save.
        scheduler: Optional learning rate scheduler to save.
        **kwargs: Additional data to save in the checkpoint.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics or {},
    }

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    checkpoint.update(kwargs)

    torch.save(checkpoint, path)


def load_checkpoint(
    path: str | Path,
    model: nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Load a training checkpoint.

    Args:
        path: Path to the checkpoint file.
        model: Optional model to load state into.
        optimizer: Optional optimizer to load state into.
        scheduler: Optional scheduler to load state into.
        device: Device to load the checkpoint to.

    Returns:
        Dictionary containing checkpoint data.

    Raises:
        FileNotFoundError: If the checkpoint file doesn't exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    map_location = device if device is not None else "cpu"
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)

    if model is not None and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint
