"""
Core Neural Network Building Blocks for EEG Conformer architectures.
"""

from __future__ import annotations

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvFrontEnd(nn.Module):
    """EEG Spatial-Temporal Convolutional Front-End.
    Transforms (B, 1, n_channels, n_samples) raw window into a sequence of patch tokens.
    """

    def __init__(
        self,
        n_channels: int,
        embed_dim: int = 128,
        temp_kernel: int = 25,
        temp_filters: int = 32,
        stride: int = 10,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.embed_dim = embed_dim
        self.stride = stride

        # Temporal conv: (B, 1, C, T) -> (B, temp_filters, C, T)
        self.temp_conv = nn.Conv2d(
            in_channels=1,
            out_channels=temp_filters,
            kernel_size=(1, temp_kernel),
            padding=(0, temp_kernel // 2),
            bias=False,
        )
        self.bn_temp = nn.BatchNorm2d(temp_filters)

        # Spatial conv across channels: (B, temp_filters, C, T) -> (B, embed_dim, 1, T)
        self.spatial_conv = nn.Conv2d(
            in_channels=temp_filters,
            out_channels=embed_dim,
            kernel_size=(n_channels, 1),
            bias=False,
        )
        self.bn_spatial = nn.BatchNorm2d(embed_dim)

        # Temporal downsampling / stride patch projection
        self.pool = nn.AvgPool2d(kernel_size=(1, stride), stride=(1, stride))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Input x: (B, n_channels, n_samples) or (B, 1, n_channels, n_samples)
        Output: (B, num_patches, embed_dim)
        """
        if x.ndim == 3:
            x = x.unsqueeze(1)  # (B, 1, C, T)

        h = F.elu(self.bn_temp(self.temp_conv(x)))
        h = F.elu(self.bn_spatial(self.spatial_conv(h)))  # (B, embed_dim, 1, T)
        h = self.pool(h)                                  # (B, embed_dim, 1, num_patches)
        h = self.drop(h)

        h = h.squeeze(2).permute(0, 2, 1).contiguous()
        return h


class PositionalEncoding(nn.Module):
    """Learnable positional encoding for sequence tokens."""

    def __init__(self, embed_dim: int, max_len: int = 2000):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Input x: (B, seq_len, embed_dim) -> (B, seq_len, embed_dim)"""
        seq_len = x.shape[1]
        return x + self.pos_embed[:, :seq_len, :]


class CausalTransformerEncoderBlock(nn.Module):
    """Pre-norm Causal Transformer Encoder block with Multi-Head Self-Attention (MHSA)."""

    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 4,
        ffn_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        norm_x = self.ln1(x)
        attn_out, _ = self.attn(
            query=norm_x,
            key=norm_x,
            value=norm_x,
            attn_mask=causal_mask,
            need_weights=False,
        )
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return x


class Conv1DClassifierBackbone(nn.Module):
    """1D Convolutional Backbone for sequence feature classification."""

    def __init__(self, embed_dim: int = 128, hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.conv1 = nn.Conv1d(embed_dim, hidden_dim, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.drop = nn.Dropout(dropout)

        self.shortcut = nn.Conv1d(embed_dim, hidden_dim, kernel_size=1) if embed_dim != hidden_dim else nn.Identity()

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Input x: (B, seq_len, embed_dim) -> (B, hidden_dim * 2)"""
        h = x.permute(0, 2, 1)

        res = self.shortcut(h)
        h = F.gelu(self.bn1(self.conv1(h)))
        h = self.drop(h)
        h = F.gelu(self.bn2(self.conv2(h)) + res)
        h = self.drop(h)

        avg_feat = self.avg_pool(h).squeeze(-1)
        max_feat = self.max_pool(h).squeeze(-1)

        return torch.cat([avg_feat, max_feat], dim=-1)
