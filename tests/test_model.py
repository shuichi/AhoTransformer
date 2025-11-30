"""Tests for the Transformer model."""

import pytest
import torch

from aho_transformer.models.transformer import AhoTransformer, PositionalEncoding


class TestPositionalEncoding:
    """Tests for PositionalEncoding."""

    def test_output_shape(self) -> None:
        """Test that output shape matches input shape."""
        d_model = 64
        max_len = 100
        batch_size = 4
        seq_len = 20

        pe = PositionalEncoding(d_model=d_model, max_len=max_len)
        x = torch.randn(seq_len, batch_size, d_model)
        output = pe(x)

        assert output.shape == x.shape

    def test_different_sequence_lengths(self) -> None:
        """Test with different sequence lengths."""
        d_model = 32
        pe = PositionalEncoding(d_model=d_model, max_len=100)

        for seq_len in [5, 10, 50, 100]:
            x = torch.randn(seq_len, 2, d_model)
            output = pe(x)
            assert output.shape == x.shape


class TestAhoTransformer:
    """Tests for AhoTransformer model."""

    @pytest.fixture
    def model(self) -> AhoTransformer:
        """Create a test model."""
        return AhoTransformer(
            vocab_size=100,
            d_model=64,
            nhead=4,
            num_encoder_layers=2,
            num_decoder_layers=2,
            dim_feedforward=128,
            dropout=0.1,
            max_seq_length=50,
        )

    def test_forward_pass(self, model: AhoTransformer) -> None:
        """Test forward pass produces correct output shape."""
        batch_size = 4
        src_len = 10
        tgt_len = 8

        src = torch.randint(0, 100, (src_len, batch_size))
        tgt = torch.randint(0, 100, (tgt_len, batch_size))

        output = model(src, tgt)

        assert output.shape == (tgt_len, batch_size, 100)

    def test_encode(self, model: AhoTransformer) -> None:
        """Test encoder produces correct output shape."""
        batch_size = 4
        src_len = 10

        src = torch.randint(0, 100, (src_len, batch_size))
        memory = model.encode(src)

        assert memory.shape == (src_len, batch_size, model.d_model)

    def test_decode(self, model: AhoTransformer) -> None:
        """Test decoder produces correct output shape."""
        batch_size = 4
        src_len = 10
        tgt_len = 8

        src = torch.randint(0, 100, (src_len, batch_size))
        tgt = torch.randint(0, 100, (tgt_len, batch_size))

        memory = model.encode(src)
        output = model.decode(tgt, memory)

        assert output.shape == (tgt_len, batch_size, 100)

    def test_generate_square_subsequent_mask(self, model: AhoTransformer) -> None:
        """Test mask generation."""
        device = torch.device("cpu")
        mask = model.generate_square_subsequent_mask(5, device)

        assert mask.shape == (5, 5)
        # Upper triangle should be -inf
        assert mask[0, 1] == float("-inf")
        assert mask[0, 4] == float("-inf")
        # Diagonal and below should be 0
        assert mask[0, 0] == 0.0
        assert mask[4, 4] == 0.0
        assert mask[4, 0] == 0.0

    def test_with_masks(self, model: AhoTransformer) -> None:
        """Test forward pass with masks."""
        batch_size = 4
        src_len = 10
        tgt_len = 8

        src = torch.randint(0, 100, (src_len, batch_size))
        tgt = torch.randint(0, 100, (tgt_len, batch_size))

        tgt_mask = model.generate_square_subsequent_mask(tgt_len, src.device)
        src_padding_mask = torch.zeros(batch_size, src_len, dtype=torch.bool)
        tgt_padding_mask = torch.zeros(batch_size, tgt_len, dtype=torch.bool)

        output = model(
            src,
            tgt,
            tgt_mask=tgt_mask,
            src_padding_mask=src_padding_mask,
            tgt_padding_mask=tgt_padding_mask,
        )

        assert output.shape == (tgt_len, batch_size, 100)

    def test_num_classes(self) -> None:
        """Test model with custom number of output classes."""
        model = AhoTransformer(
            vocab_size=100,
            d_model=64,
            nhead=4,
            num_encoder_layers=2,
            num_decoder_layers=2,
            num_classes=10,
        )

        batch_size = 4
        src = torch.randint(0, 100, (10, batch_size))
        tgt = torch.randint(0, 100, (8, batch_size))

        output = model(src, tgt)
        assert output.shape == (8, batch_size, 10)

    def test_gradient_flow(self, model: AhoTransformer) -> None:
        """Test that gradients flow through the model."""
        src = torch.randint(0, 100, (10, 4))
        tgt = torch.randint(0, 100, (8, 4))

        output = model(src, tgt)
        loss = output.sum()
        loss.backward()

        # Check that gradients exist
        for param in model.parameters():
            assert param.grad is not None
