"""CLI entry points for AhoTransformer.

This module provides command-line interface entry points for training
and evaluation.
"""

from __future__ import annotations

import argparse
import sys


def train_cli() -> None:
    """CLI entry point for training."""
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
        default="outputs",
        help="Directory to save outputs",
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
        default="auto",
        help="Device to use (cpu, cuda, auto)",
    )

    args = parser.parse_args()

    # Import here to avoid circular imports and speed up CLI help
    from aho_transformer.utils.config import load_config
    from aho_transformer.utils.logging import setup_logging

    setup_logging()

    config = load_config(args.config)
    print(f"Loaded configuration from {args.config}")
    print(f"Configuration: {config}")
    print(f"Output directory: {args.output_dir}")

    # Training logic would go here
    print("Training not yet implemented. Use scripts/train.py for training.")


def eval_cli() -> None:
    """CLI entry point for evaluation."""
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

    args = parser.parse_args()

    # Import here to avoid circular imports and speed up CLI help
    from aho_transformer.utils.config import load_config
    from aho_transformer.utils.logging import setup_logging

    setup_logging()

    load_config(args.config)
    print(f"Loaded configuration from {args.config}")
    print(f"Evaluating checkpoint: {args.checkpoint}")
    print(f"Output directory: {args.output_dir}")

    # Evaluation logic would go here
    print("Evaluation not yet implemented. Use scripts/evaluate.py for evaluation.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "eval":
        sys.argv.pop(1)
        eval_cli()
    else:
        train_cli()
