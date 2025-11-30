"""Tests for the Trainer module."""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from aho_transformer.trainer import Trainer


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self, input_size: int = 10, output_size: int = 5) -> None:
        super().__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        return self.fc(x)


class TestTrainer:
    """Tests for Trainer class."""

    @pytest.fixture
    def simple_setup(self) -> tuple:
        """Create a simple setup for testing."""
        model = SimpleModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.MSELoss()
        device = torch.device("cpu")

        return model, optimizer, criterion, device

    @pytest.fixture
    def dummy_dataloader(self) -> DataLoader:
        """Create a dummy dataloader."""
        x = torch.randn(100, 10)
        y = torch.randn(100, 5)
        dataset = TensorDataset(x, y)
        return DataLoader(dataset, batch_size=10)

    def test_trainer_initialization(self, simple_setup: tuple) -> None:
        """Test trainer initialization."""
        model, optimizer, criterion, device = simple_setup
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )

        assert trainer.model is not None
        assert trainer.optimizer is optimizer
        assert trainer.criterion is criterion

    def test_train_epoch(
        self, simple_setup: tuple, dummy_dataloader: DataLoader
    ) -> None:
        """Test training for one epoch."""
        model, optimizer, criterion, device = simple_setup

        # Override _train_step for simple model
        class SimpleTrainer(Trainer):
            def _train_step(self, batch):
                self.optimizer.zero_grad()
                inputs, targets = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                outputs = self.model(inputs, None)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                return loss.item()

        trainer = SimpleTrainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )

        metrics = trainer.train_epoch(dummy_dataloader)

        assert "train_loss" in metrics
        assert metrics["train_loss"] > 0

    def test_validate(
        self, simple_setup: tuple, dummy_dataloader: DataLoader
    ) -> None:
        """Test validation."""
        model, optimizer, criterion, device = simple_setup

        class SimpleTrainer(Trainer):
            def _validate_step(self, batch):
                inputs, targets = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                outputs = self.model(inputs, None)
                loss = self.criterion(outputs, targets)
                return loss.item()

        trainer = SimpleTrainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )

        metrics = trainer.validate(dummy_dataloader)

        assert "val_loss" in metrics
        assert metrics["val_loss"] > 0
