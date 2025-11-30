#!/usr/bin/env python3
"""Training script for AhoTransformer.

This script provides a complete training pipeline that can be customized
through configuration files or command-line arguments.

Usage:
    python scripts/train.py --config configs/default.yaml
    python scripts/train.py --config configs/default.yaml --epochs 100
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import torch.nn as nn

from aho_transformer.data.dataloader import create_sequence_dataloaders
from aho_transformer.data.dataset import SequenceDataset
from aho_transformer.models.transformer import AhoTransformer
from aho_transformer.trainer import Trainer
from aho_transformer.utils.checkpoint import load_checkpoint
from aho_transformer.utils.config import load_config, merge_configs
from aho_transformer.utils.logging import get_logger, setup_logging


def get_device(device_str: str) -> torch.device:
    """Get the device to use for training.

    Args:
        device_str: Device string (cpu, cuda, auto).

    Returns:
        torch.device instance.
    """
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def create_dummy_data(
    num_samples: int = 1000,
    vocab_size: int = 100,
    max_seq_len: int = 32,
) -> SequenceDataset:
    """Create dummy data for testing the training pipeline.

    Args:
        num_samples: Number of samples to generate.
        vocab_size: Size of the vocabulary.
        max_seq_len: Maximum sequence length.

    Returns:
        SequenceDataset with dummy data.
    """
    import random

    random.seed(42)

    src_sequences = []
    tgt_sequences = []

    for _ in range(num_samples):
        seq_len = random.randint(5, max_seq_len)
        src = [random.randint(1, vocab_size - 1) for _ in range(seq_len)]
        # Simple task: reverse the sequence
        tgt = [0] + src[::-1] + [0]  # Add BOS/EOS tokens
        src_sequences.append(src)
        tgt_sequences.append(tgt)

    return SequenceDataset(src_sequences, tgt_sequences, pad_idx=0)


def main() -> None:
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train AhoTransformer model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save outputs (overrides config)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cpu, cuda, auto)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of epochs (overrides config)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size (overrides config)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate (overrides config)",
    )
    parser.add_argument(
        "--use-dummy-data",
        action="store_true",
        help="Use dummy data for testing",
    )

    args = parser.parse_args()

    # Load configuration
    config_path = Path(args.config)
    if config_path.exists():
        config = load_config(config_path)
    else:
        print(f"Config file not found: {config_path}, using defaults")
        config = {}

    # Apply command-line overrides
    overrides = {}
    if args.output_dir:
        overrides["training"] = overrides.get("training", {})
        overrides["training"]["output_dir"] = args.output_dir
    if args.epochs:
        overrides["training"] = overrides.get("training", {})
        overrides["training"]["epochs"] = args.epochs
    if args.batch_size:
        overrides["training"] = overrides.get("training", {})
        overrides["training"]["batch_size"] = args.batch_size
    if args.lr:
        overrides["training"] = overrides.get("training", {})
        overrides["training"]["learning_rate"] = args.lr
    if args.device:
        overrides["training"] = overrides.get("training", {})
        overrides["training"]["device"] = args.device

    config = merge_configs(config, overrides)

    # Set up defaults
    training_config = config.get("training", {})
    model_config = config.get("model", {})

    output_dir = Path(training_config.get("output_dir", "outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set up logging
    setup_logging(
        log_level=training_config.get("log_level", "INFO"),
        log_file=output_dir / "train.log",
    )
    logger = get_logger(__name__)

    logger.info("Starting training")
    logger.info(f"Configuration: {config}")

    # Get device
    device = get_device(training_config.get("device", "auto"))
    logger.info(f"Using device: {device}")

    # Create dataset
    if args.use_dummy_data:
        logger.info("Using dummy data for testing")
        dataset = create_dummy_data(
            num_samples=training_config.get("num_samples", 1000),
            vocab_size=model_config.get("vocab_size", 100),
            max_seq_len=model_config.get("max_seq_length", 32),
        )
    else:
        # TODO: Implement actual data loading
        logger.info("No data specified, using dummy data")
        dataset = create_dummy_data(
            num_samples=training_config.get("num_samples", 1000),
            vocab_size=model_config.get("vocab_size", 100),
            max_seq_len=model_config.get("max_seq_length", 32),
        )

    # Create data loaders
    train_loader, val_loader, test_loader = create_sequence_dataloaders(
        dataset=dataset,
        batch_size=training_config.get("batch_size", 32),
        train_ratio=training_config.get("train_ratio", 0.8),
        val_ratio=training_config.get("val_ratio", 0.1),
        test_ratio=training_config.get("test_ratio", 0.1),
        pad_idx=0,
        seed=training_config.get("seed", 42),
    )

    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Val batches: {len(val_loader)}")
    logger.info(f"Test batches: {len(test_loader)}")

    # Create model
    model = AhoTransformer(
        vocab_size=model_config.get("vocab_size", 100),
        d_model=model_config.get("d_model", 256),
        nhead=model_config.get("nhead", 4),
        num_encoder_layers=model_config.get("num_encoder_layers", 3),
        num_decoder_layers=model_config.get("num_decoder_layers", 3),
        dim_feedforward=model_config.get("dim_feedforward", 512),
        dropout=model_config.get("dropout", 0.1),
        max_seq_length=model_config.get("max_seq_length", 512),
    )

    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create optimizer and scheduler
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training_config.get("learning_rate", 1e-4),
        weight_decay=training_config.get("weight_decay", 0.0),
    )

    scheduler = None
    if training_config.get("use_scheduler", True):
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=training_config.get("scheduler_step_size", 10),
            gamma=training_config.get("scheduler_gamma", 0.1),
        )

    # Loss function
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding

    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        checkpoint = load_checkpoint(
            path=args.resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
        )
        start_epoch = checkpoint.get("epoch", 0) + 1
        logger.info(f"Resuming from epoch {start_epoch}")

    # Create trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        scheduler=scheduler,
        checkpoint_dir=output_dir / "checkpoints",
    )

    # Train
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=training_config.get("epochs", 10),
        early_stopping_patience=training_config.get("early_stopping_patience", None),
    )

    logger.info("Training completed")
    logger.info(f"Final train loss: {history['train_loss'][-1]:.4f}")
    if history["val_loss"]:
        logger.info(f"Final val loss: {history['val_loss'][-1]:.4f}")


if __name__ == "__main__":
    main()
