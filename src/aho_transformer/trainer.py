"""Trainer module for AhoTransformer.

This module provides a training loop implementation that can be used
for various machine learning experiments.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from aho_transformer.utils.checkpoint import save_checkpoint
from aho_transformer.utils.logging import get_logger

logger = get_logger(__name__)


class Trainer:
    """A training loop implementation for PyTorch models.

    This trainer provides a flexible training loop with support for
    validation, checkpointing, and early stopping.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        checkpoint_dir: str | Path | None = None,
        metrics_fn: Callable[[torch.Tensor, torch.Tensor], dict[str, float]] | None = None,
    ):
        """Initialize the trainer.

        Args:
            model: The model to train.
            optimizer: The optimizer to use.
            criterion: The loss function.
            device: Device to train on.
            scheduler: Optional learning rate scheduler.
            checkpoint_dir: Optional directory to save checkpoints.
            metrics_fn: Optional function to compute additional metrics.
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scheduler = scheduler
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.metrics_fn = metrics_fn

        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def train_epoch(self, train_loader: DataLoader) -> dict[str, float]:
        """Train for one epoch.

        Args:
            train_loader: DataLoader for training data.

        Returns:
            Dictionary of training metrics.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(train_loader, desc="Training")
        for batch in progress_bar:
            loss = self._train_step(batch)
            total_loss += loss
            num_batches += 1
            progress_bar.set_postfix({"loss": f"{loss:.4f}"})

        avg_loss = total_loss / max(num_batches, 1)
        return {"train_loss": avg_loss}

    def _train_step(self, batch: Any) -> float:
        """Perform a single training step.

        Args:
            batch: A batch of training data.

        Returns:
            Loss value for this step.
        """
        self.optimizer.zero_grad()

        # Handle different batch formats
        if isinstance(batch, (list, tuple)):
            if len(batch) == 2:
                inputs, targets = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                outputs = self.model(inputs, targets[:-1])
                loss = self.criterion(
                    outputs.reshape(-1, outputs.size(-1)), targets[1:].reshape(-1)
                )
            elif len(batch) == 4:
                # Sequence data with masks
                src, tgt, src_mask, tgt_mask = batch
                src = src.to(self.device)
                tgt = tgt.to(self.device)
                src_mask = src_mask.to(self.device)
                tgt_mask = tgt_mask.to(self.device)

                tgt_input = tgt[:-1, :]
                outputs = self.model(
                    src, tgt_input, src_padding_mask=src_mask, tgt_padding_mask=tgt_mask[:, :-1]
                )
                tgt_output = tgt[1:, :]
                loss = self.criterion(
                    outputs.reshape(-1, outputs.size(-1)), tgt_output.reshape(-1)
                )
            else:
                raise ValueError(f"Unexpected batch format with {len(batch)} elements")
        else:
            inputs = batch.to(self.device)
            outputs = self.model(inputs, inputs)
            loss = self.criterion(outputs.reshape(-1, outputs.size(-1)), inputs.reshape(-1))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        return loss.item()

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> dict[str, float]:
        """Validate the model.

        Args:
            val_loader: DataLoader for validation data.

        Returns:
            Dictionary of validation metrics.
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in tqdm(val_loader, desc="Validation"):
            loss = self._validate_step(batch)
            total_loss += loss
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        return {"val_loss": avg_loss}

    def _validate_step(self, batch: Any) -> float:
        """Perform a single validation step.

        Args:
            batch: A batch of validation data.

        Returns:
            Loss value for this step.
        """
        if isinstance(batch, (list, tuple)):
            if len(batch) == 2:
                inputs, targets = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                outputs = self.model(inputs, targets[:-1])
                loss = self.criterion(
                    outputs.reshape(-1, outputs.size(-1)), targets[1:].reshape(-1)
                )
            elif len(batch) == 4:
                src, tgt, src_mask, tgt_mask = batch
                src = src.to(self.device)
                tgt = tgt.to(self.device)
                src_mask = src_mask.to(self.device)
                tgt_mask = tgt_mask.to(self.device)

                tgt_input = tgt[:-1, :]
                outputs = self.model(
                    src, tgt_input, src_padding_mask=src_mask, tgt_padding_mask=tgt_mask[:, :-1]
                )
                tgt_output = tgt[1:, :]
                loss = self.criterion(
                    outputs.reshape(-1, outputs.size(-1)), tgt_output.reshape(-1)
                )
            else:
                raise ValueError(f"Unexpected batch format with {len(batch)} elements")
        else:
            inputs = batch.to(self.device)
            outputs = self.model(inputs, inputs)
            loss = self.criterion(outputs.reshape(-1, outputs.size(-1)), inputs.reshape(-1))

        return loss.item()

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        epochs: int = 10,
        early_stopping_patience: int | None = None,
    ) -> dict[str, list[float]]:
        """Run the full training loop.

        Args:
            train_loader: DataLoader for training data.
            val_loader: Optional DataLoader for validation data.
            epochs: Number of epochs to train.
            early_stopping_patience: Number of epochs to wait for improvement
                                    before early stopping. None to disable.

        Returns:
            Dictionary of training history (lists of metrics per epoch).
        """
        history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(epochs):
            logger.info(f"Epoch {epoch + 1}/{epochs}")

            # Training
            train_metrics = self.train_epoch(train_loader)
            history["train_loss"].append(train_metrics["train_loss"])
            logger.info(f"Train Loss: {train_metrics['train_loss']:.4f}")

            # Validation
            if val_loader is not None:
                val_metrics = self.validate(val_loader)
                history["val_loss"].append(val_metrics["val_loss"])
                logger.info(f"Val Loss: {val_metrics['val_loss']:.4f}")

                # Early stopping check
                if val_metrics["val_loss"] < best_val_loss:
                    best_val_loss = val_metrics["val_loss"]
                    patience_counter = 0

                    if self.checkpoint_dir:
                        save_checkpoint(
                            model=self.model,
                            optimizer=self.optimizer,
                            epoch=epoch,
                            path=self.checkpoint_dir / "best_model.pt",
                            metrics={"val_loss": best_val_loss},
                            scheduler=self.scheduler,
                        )
                else:
                    patience_counter += 1

                if (
                    early_stopping_patience is not None
                    and patience_counter >= early_stopping_patience
                ):
                    logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                    break

            # Learning rate scheduling
            if self.scheduler is not None:
                self.scheduler.step()

        return history
