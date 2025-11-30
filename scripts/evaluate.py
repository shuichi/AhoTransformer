#!/usr/bin/env python3
"""Evaluation script for AhoTransformer.

This script provides evaluation utilities for trained models.

Usage:
    python scripts/evaluate.py --checkpoint outputs/checkpoints/best_model.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from aho_transformer.data.dataloader import create_sequence_dataloaders
from aho_transformer.data.dataset import SequenceDataset
from aho_transformer.models.transformer import AhoTransformer
from aho_transformer.utils.checkpoint import load_checkpoint
from aho_transformer.utils.config import load_config
from aho_transformer.utils.logging import get_logger, setup_logging


def get_device(device_str: str) -> torch.device:
    """Get the device to use for evaluation.

    Args:
        device_str: Device string (cpu, cuda, auto).

    Returns:
        torch.device instance.
    """
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def create_dummy_data(
    num_samples: int = 100,
    vocab_size: int = 100,
    max_seq_len: int = 32,
) -> SequenceDataset:
    """Create dummy data for testing the evaluation pipeline.

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
        tgt = [0] + src[::-1] + [0]
        src_sequences.append(src)
        tgt_sequences.append(tgt)

    return SequenceDataset(src_sequences, tgt_sequences, pad_idx=0)


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate the model on a dataset.

    Args:
        model: The model to evaluate.
        data_loader: DataLoader for the evaluation data.
        criterion: The loss function.
        device: Device to evaluate on.

    Returns:
        Dictionary of evaluation metrics.
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    num_batches = 0

    for batch in tqdm(data_loader, desc="Evaluating"):
        if len(batch) == 4:
            src, tgt, src_mask, tgt_mask = batch
            src = src.to(device)
            tgt = tgt.to(device)
            src_mask = src_mask.to(device)
            tgt_mask = tgt_mask.to(device)

            tgt_input = tgt[:-1, :]
            outputs = model(
                src, tgt_input, src_padding_mask=src_mask, tgt_padding_mask=tgt_mask[:, :-1]
            )
            tgt_output = tgt[1:, :]

            loss = criterion(outputs.reshape(-1, outputs.size(-1)), tgt_output.reshape(-1))

            # Calculate accuracy (excluding padding)
            predictions = outputs.argmax(dim=-1)
            mask = tgt_output != 0  # Non-padding tokens
            correct = (predictions == tgt_output) & mask
            total_correct += correct.sum().item()
            total_tokens += mask.sum().item()
        else:
            raise ValueError(f"Unexpected batch format with {len(batch)} elements")

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    accuracy = total_correct / max(total_tokens, 1)

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "perplexity": torch.exp(torch.tensor(avg_loss)).item(),
    }


def main() -> None:
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate AhoTransformer model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/eval",
        help="Directory to save evaluation outputs",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (cpu, cuda, auto)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--use-dummy-data",
        action="store_true",
        help="Use dummy data for testing",
    )

    args = parser.parse_args()

    # Set up output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set up logging
    setup_logging(log_level="INFO", log_file=output_dir / "eval.log")
    logger = get_logger(__name__)

    logger.info("Starting evaluation")

    # Load configuration
    config_path = Path(args.config)
    if config_path.exists():
        config = load_config(config_path)
    else:
        logger.warning(f"Config file not found: {config_path}, using defaults")
        config = {}

    model_config = config.get("model", {})

    # Get device
    device = get_device(args.device)
    logger.info(f"Using device: {device}")

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

    # Load checkpoint
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = load_checkpoint(path=args.checkpoint, model=model, device=device)
    logger.info(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")

    model = model.to(device)

    # Create dataset
    if args.use_dummy_data:
        logger.info("Using dummy data for testing")
        dataset = create_dummy_data(
            num_samples=100,
            vocab_size=model_config.get("vocab_size", 100),
            max_seq_len=model_config.get("max_seq_length", 32),
        )
    else:
        logger.info("No data specified, using dummy data")
        dataset = create_dummy_data(
            num_samples=100,
            vocab_size=model_config.get("vocab_size", 100),
            max_seq_len=model_config.get("max_seq_length", 32),
        )

    # Create data loader
    _, _, test_loader = create_sequence_dataloaders(
        dataset=dataset,
        batch_size=args.batch_size,
        train_ratio=0.0,
        val_ratio=0.0,
        test_ratio=1.0,
        pad_idx=0,
    )

    # Loss function
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # Evaluate
    metrics = evaluate_model(model, test_loader, criterion, device)

    logger.info("Evaluation Results:")
    logger.info(f"  Loss: {metrics['loss']:.4f}")
    logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"  Perplexity: {metrics['perplexity']:.4f}")

    # Save results
    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
