# AhoTransformer

A Transformer that acts "aho" only for multiples of 3 and numbers that contain the digit 3.

## Overview

AhoTransformer is a PyTorch-based machine learning experiment framework. It provides a template for building and training Transformer models with a modular architecture.

## Installation

### From source

```bash
# Clone the repository
git clone https://github.com/shuichi/AhoTransformer.git
cd AhoTransformer

# Install in development mode
pip install -e ".[dev]"
```

### Using requirements.txt

```bash
pip install -r requirements.txt
```

## Project Structure

```
AhoTransformer/
├── configs/
│   └── default.yaml          # Default configuration
├── scripts/
│   ├── train.py              # Training script
│   └── evaluate.py           # Evaluation script
├── src/aho_transformer/
│   ├── __init__.py
│   ├── cli.py                # CLI entry points
│   ├── trainer.py            # Training loop
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py        # Dataset implementations
│   │   └── dataloader.py     # DataLoader utilities
│   ├── models/
│   │   ├── __init__.py
│   │   └── transformer.py    # Transformer model
│   └── utils/
│       ├── __init__.py
│       ├── checkpoint.py     # Checkpoint utilities
│       ├── config.py         # Configuration utilities
│       └── logging.py        # Logging utilities
├── tests/                    # Unit tests
├── pyproject.toml            # Project configuration
└── requirements.txt          # Dependencies
```

## Quick Start

### Training

```bash
# Train with default configuration
python scripts/train.py --config configs/default.yaml

# Train with dummy data for testing
python scripts/train.py --use-dummy-data --epochs 5

# Train with custom settings
python scripts/train.py --config configs/default.yaml --epochs 20 --batch-size 64 --lr 0.001
```

### Evaluation

```bash
# Evaluate a trained model
python scripts/evaluate.py --checkpoint outputs/checkpoints/best_model.pt

# Evaluate with dummy data
python scripts/evaluate.py --checkpoint outputs/checkpoints/best_model.pt --use-dummy-data
```

### CLI Commands

After installation, you can use the CLI commands:

```bash
# Training
aho-train --config configs/default.yaml

# Evaluation
aho-eval --checkpoint outputs/checkpoints/best_model.pt
```

## Configuration

The configuration file (`configs/default.yaml`) allows you to customize:

- **Model settings**: vocabulary size, model dimensions, number of layers, etc.
- **Training settings**: learning rate, batch size, epochs, early stopping, etc.
- **Data settings**: data paths, padding indices, etc.

Example configuration:

```yaml
model:
  vocab_size: 100
  d_model: 256
  nhead: 4
  num_encoder_layers: 3
  num_decoder_layers: 3

training:
  epochs: 10
  batch_size: 32
  learning_rate: 0.0001
  device: auto
```

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=aho_transformer

# Run specific test file
pytest tests/test_model.py
```

### Linting

```bash
# Check code style
ruff check src tests

# Fix auto-fixable issues
ruff check --fix src tests
```

### Type Checking

```bash
mypy src
```

## API Reference

### AhoTransformer Model

```python
from aho_transformer import AhoTransformer

model = AhoTransformer(
    vocab_size=10000,
    d_model=512,
    nhead=8,
    num_encoder_layers=6,
    num_decoder_layers=6,
)

# Forward pass
output = model(src, tgt)

# Encode only
memory = model.encode(src)

# Decode with memory
output = model.decode(tgt, memory)
```

### Dataset

```python
from aho_transformer.data import AhoDataset, create_dataloaders

# Create dataset
dataset = AhoDataset(data, labels)

# Create data loaders
train_loader, val_loader, test_loader = create_dataloaders(
    dataset,
    batch_size=32,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
)
```

### Trainer

```python
from aho_transformer import Trainer

trainer = Trainer(
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    device=device,
    checkpoint_dir="outputs/checkpoints",
)

history = trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=10,
)
```

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.
