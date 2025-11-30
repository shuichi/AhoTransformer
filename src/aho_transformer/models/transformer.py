"""Transformer model implementation for AhoTransformer.

This module provides a simple Transformer model template that can be extended
for various machine learning experiments.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer models."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        """Initialize positional encoding.

        Args:
            d_model: The dimension of the model.
            max_len: Maximum sequence length.
            dropout: Dropout rate.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input tensor.

        Args:
            x: Input tensor of shape (seq_len, batch, d_model).

        Returns:
            Tensor with positional encoding added.
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class AhoTransformer(nn.Module):
    """A simple Transformer model for the AhoTransformer project.

    This is a template Transformer that can be customized for various
    sequence-to-sequence or classification tasks.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_seq_length: int = 512,
        num_classes: int | None = None,
    ):
        """Initialize the AhoTransformer model.

        Args:
            vocab_size: Size of the vocabulary.
            d_model: The dimension of the model.
            nhead: Number of attention heads.
            num_encoder_layers: Number of encoder layers.
            num_decoder_layers: Number of decoder layers.
            dim_feedforward: Dimension of feedforward network.
            dropout: Dropout rate.
            max_seq_length: Maximum sequence length.
            num_classes: Number of output classes for classification.
                        If None, uses vocab_size for seq2seq.
        """
        super().__init__()

        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_seq_length, dropout)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False,
        )

        output_size = num_classes if num_classes is not None else vocab_size
        self.fc_out = nn.Linear(d_model, output_size)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights using Xavier uniform initialization."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        tgt_mask: torch.Tensor | None = None,
        src_padding_mask: torch.Tensor | None = None,
        tgt_padding_mask: torch.Tensor | None = None,
        memory_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass of the Transformer.

        Args:
            src: Source sequence tensor of shape (src_len, batch).
            tgt: Target sequence tensor of shape (tgt_len, batch).
            src_mask: Source mask tensor.
            tgt_mask: Target mask tensor.
            src_padding_mask: Source padding mask.
            tgt_padding_mask: Target padding mask.
            memory_key_padding_mask: Memory key padding mask.

        Returns:
            Output tensor of shape (tgt_len, batch, output_size).
        """
        src = self.embedding(src) * math.sqrt(self.d_model)
        tgt = self.embedding(tgt) * math.sqrt(self.d_model)

        src = self.pos_encoder(src)
        tgt = self.pos_encoder(tgt)

        output = self.transformer(
            src,
            tgt,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )

        return self.fc_out(output)

    def encode(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        src_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode source sequence.

        Args:
            src: Source sequence tensor of shape (src_len, batch).
            src_mask: Source mask tensor.
            src_padding_mask: Source padding mask.

        Returns:
            Encoded memory tensor.
        """
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        return self.transformer.encoder(src, mask=src_mask, src_key_padding_mask=src_padding_mask)

    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        tgt_padding_mask: torch.Tensor | None = None,
        memory_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Decode target sequence with encoded memory.

        Args:
            tgt: Target sequence tensor of shape (tgt_len, batch).
            memory: Encoded memory from encoder.
            tgt_mask: Target mask tensor.
            tgt_padding_mask: Target padding mask.
            memory_key_padding_mask: Memory key padding mask.

        Returns:
            Decoded output tensor.
        """
        tgt = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt = self.pos_encoder(tgt)
        output = self.transformer.decoder(
            tgt,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return self.fc_out(output)

    @staticmethod
    def generate_square_subsequent_mask(sz: int, device: torch.device) -> torch.Tensor:
        """Generate a square mask for the sequence.

        Args:
            sz: Size of the mask.
            device: Device to create the mask on.

        Returns:
            A mask tensor where positions are masked with float('-inf').
        """
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float("-inf"))
        return mask
