"""Tests for utility modules."""

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from aho_transformer.utils.checkpoint import load_checkpoint, save_checkpoint
from aho_transformer.utils.config import load_config, merge_configs, save_config
from aho_transformer.utils.logging import get_logger, setup_logging


class TestConfig:
    """Tests for configuration utilities."""

    def test_load_config(self, tmp_path: Path) -> None:
        """Test loading a configuration file."""
        config_content = """
model:
  d_model: 256
  nhead: 8
training:
  epochs: 10
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_content)

        config = load_config(config_file)

        assert config["model"]["d_model"] == 256
        assert config["model"]["nhead"] == 8
        assert config["training"]["epochs"] == 10

    def test_load_config_not_found(self) -> None:
        """Test that missing config raises an error."""
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent.yaml")

    def test_save_config(self, tmp_path: Path) -> None:
        """Test saving a configuration file."""
        config = {"model": {"d_model": 512}, "training": {"epochs": 20}}
        config_file = tmp_path / "saved_config.yaml"

        save_config(config, config_file)

        assert config_file.exists()
        loaded = load_config(config_file)
        assert loaded["model"]["d_model"] == 512

    def test_merge_configs(self) -> None:
        """Test merging configurations."""
        base = {"a": 1, "b": {"c": 2, "d": 3}}
        override = {"b": {"c": 10}, "e": 5}

        merged = merge_configs(base, override)

        assert merged["a"] == 1
        assert merged["b"]["c"] == 10
        assert merged["b"]["d"] == 3
        assert merged["e"] == 5


class TestLogging:
    """Tests for logging utilities."""

    def test_setup_logging(self) -> None:
        """Test setting up logging."""
        setup_logging(log_level="DEBUG")
        logger = get_logger("test")
        assert logger.name == "test"

    def test_setup_logging_with_file(self, tmp_path: Path) -> None:
        """Test setting up logging with file output."""
        log_file = tmp_path / "test.log"
        setup_logging(log_level="INFO", log_file=log_file)

        logger = get_logger("test_file")
        logger.info("Test message")

        assert log_file.exists()


class TestCheckpoint:
    """Tests for checkpoint utilities."""

    @pytest.fixture
    def simple_model(self) -> nn.Module:
        """Create a simple model for testing."""
        return nn.Linear(10, 5)

    def test_save_and_load_checkpoint(
        self, simple_model: nn.Module, tmp_path: Path
    ) -> None:
        """Test saving and loading a checkpoint."""
        optimizer = torch.optim.SGD(simple_model.parameters(), lr=0.01)
        checkpoint_path = tmp_path / "checkpoint.pt"

        save_checkpoint(
            model=simple_model,
            optimizer=optimizer,
            epoch=5,
            path=checkpoint_path,
            metrics={"loss": 0.5},
        )

        assert checkpoint_path.exists()

        # Create new model and load
        new_model = nn.Linear(10, 5)
        new_optimizer = torch.optim.SGD(new_model.parameters(), lr=0.01)

        checkpoint = load_checkpoint(
            path=checkpoint_path,
            model=new_model,
            optimizer=new_optimizer,
        )

        assert checkpoint["epoch"] == 5
        assert checkpoint["metrics"]["loss"] == 0.5

        # Check that model weights are loaded
        for p1, p2 in zip(simple_model.parameters(), new_model.parameters()):
            assert torch.equal(p1, p2)

    def test_load_checkpoint_not_found(self) -> None:
        """Test that missing checkpoint raises an error."""
        with pytest.raises(FileNotFoundError):
            load_checkpoint("nonexistent.pt")

    def test_save_checkpoint_with_scheduler(
        self, simple_model: nn.Module, tmp_path: Path
    ) -> None:
        """Test saving checkpoint with scheduler."""
        optimizer = torch.optim.SGD(simple_model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        checkpoint_path = tmp_path / "checkpoint_with_scheduler.pt"

        save_checkpoint(
            model=simple_model,
            optimizer=optimizer,
            epoch=3,
            path=checkpoint_path,
            scheduler=scheduler,
        )

        new_model = nn.Linear(10, 5)
        new_optimizer = torch.optim.SGD(new_model.parameters(), lr=0.01)
        new_scheduler = torch.optim.lr_scheduler.StepLR(new_optimizer, step_size=1)

        checkpoint = load_checkpoint(
            path=checkpoint_path,
            model=new_model,
            optimizer=new_optimizer,
            scheduler=new_scheduler,
        )

        assert "scheduler_state_dict" in checkpoint
