"""Tests for the dataset and dataloader modules."""

import pytest
import torch

from aho_transformer.data.dataloader import create_dataloaders, create_sequence_dataloaders
from aho_transformer.data.dataset import AhoDataset, SequenceDataset, collate_sequences


class TestAhoDataset:
    """Tests for AhoDataset."""

    def test_basic_dataset(self) -> None:
        """Test basic dataset creation and access."""
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        labels = [0, 1, 0]

        dataset = AhoDataset(data, labels)

        assert len(dataset) == 3
        sample, label = dataset[0]
        assert torch.equal(sample, torch.tensor([1, 2, 3]))
        assert torch.equal(label, torch.tensor(0))

    def test_dataset_without_labels(self) -> None:
        """Test dataset without labels."""
        data = [[1, 2, 3], [4, 5, 6]]
        dataset = AhoDataset(data)

        assert len(dataset) == 2
        sample, label = dataset[0]
        assert torch.equal(sample, torch.tensor([1, 2, 3]))
        assert label is None

    def test_dataset_with_transform(self) -> None:
        """Test dataset with custom transform."""
        data = [[1, 2, 3], [4, 5, 6]]

        def transform(x: list[int]) -> torch.Tensor:
            return torch.tensor(x, dtype=torch.float32) * 2

        dataset = AhoDataset(data, transform=transform)

        sample, _ = dataset[0]
        assert torch.equal(sample, torch.tensor([2.0, 4.0, 6.0]))


class TestSequenceDataset:
    """Tests for SequenceDataset."""

    def test_basic_sequence_dataset(self) -> None:
        """Test basic sequence dataset."""
        src = [[1, 2, 3], [4, 5, 6, 7]]
        tgt = [[10, 20], [30, 40, 50]]

        dataset = SequenceDataset(src, tgt)

        assert len(dataset) == 2
        src_seq, tgt_seq = dataset[0]
        assert torch.equal(src_seq, torch.tensor([1, 2, 3]))
        assert torch.equal(tgt_seq, torch.tensor([10, 20]))

    def test_sequence_dataset_length_mismatch(self) -> None:
        """Test that mismatched lengths raise an error."""
        src = [[1, 2, 3]]
        tgt = [[10, 20], [30, 40]]

        with pytest.raises(ValueError):
            SequenceDataset(src, tgt)

    def test_sequence_truncation(self) -> None:
        """Test sequence truncation."""
        src = [[1, 2, 3, 4, 5]]
        tgt = [[10, 20, 30, 40, 50]]

        dataset = SequenceDataset(src, tgt, max_src_len=3, max_tgt_len=2)

        src_seq, tgt_seq = dataset[0]
        assert len(src_seq) == 3
        assert len(tgt_seq) == 2


class TestCollateSequences:
    """Tests for collate_sequences function."""

    def test_collate_with_padding(self) -> None:
        """Test collation with padding."""
        batch = [
            (torch.tensor([1, 2, 3]), torch.tensor([10, 20])),
            (torch.tensor([4, 5]), torch.tensor([30, 40, 50])),
        ]

        src, tgt, src_mask, tgt_mask = collate_sequences(batch, pad_idx=0)

        # Check shapes (transposed to seq_len, batch)
        assert src.shape == (3, 2)  # max_src_len=3, batch=2
        assert tgt.shape == (3, 2)  # max_tgt_len=3, batch=2

        # Check padding
        assert src[2, 1] == 0  # Padded position
        assert tgt[0, 0] == 10  # Non-padded position

        # Check masks
        assert src_mask[1, 2]  # Padded position should be True
        assert not src_mask[0, 0]  # Non-padded should be False


class TestCreateDataloaders:
    """Tests for dataloader creation functions."""

    def test_create_dataloaders(self) -> None:
        """Test creating train/val/test dataloaders."""
        data = list(range(100))
        labels = [i % 2 for i in range(100)]
        dataset = AhoDataset(data, labels)

        train_loader, val_loader, test_loader = create_dataloaders(
            dataset,
            batch_size=10,
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
        )

        assert len(train_loader.dataset) == 80
        assert len(val_loader.dataset) == 10
        assert len(test_loader.dataset) == 10

    def test_create_dataloaders_invalid_ratios(self) -> None:
        """Test that invalid ratios raise an error."""
        data = list(range(100))
        dataset = AhoDataset(data)

        with pytest.raises(ValueError):
            create_dataloaders(
                dataset,
                train_ratio=0.5,
                val_ratio=0.3,
                test_ratio=0.3,  # Sum > 1
            )

    def test_create_sequence_dataloaders(self) -> None:
        """Test creating sequence dataloaders."""
        src = [[1, 2, 3] for _ in range(100)]
        tgt = [[10, 20, 30] for _ in range(100)]
        dataset = SequenceDataset(src, tgt)

        train_loader, val_loader, test_loader = create_sequence_dataloaders(
            dataset,
            batch_size=10,
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
        )

        # Check that batches are properly collated
        batch = next(iter(train_loader))
        assert len(batch) == 4  # src, tgt, src_mask, tgt_mask
