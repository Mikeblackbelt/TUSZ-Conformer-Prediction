"""
Causal Generative EEG-Conformer for Next-Patch/Token Prediction.

Architecture:
1. PatchEmbedding: Temporal Conv -> Spatial Conv front-end to extract patch tokens
   from (B, 1, C, T) EEG windows.
2. PositionalEncoding: Adds learnable sequence positional embeddings.
3. CausalTransformerEncoder: Stack of Pre-Norm Transformer / Conformer blocks
   with lower-triangular causal masking so token i only attends to tokens 1..i.
4. NextTokenPredictionHead: Per-position projection head predicting the next patch
   vector (or codebook index) for autoregressive generation of EEG transitions
   (e.g., interictal -> preictal -> ictal evolution).
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvFrontEnd(nn.Module):
    """
    EEG Spatial-Temporal Convolutional Front-End.
    Transforms (B, 1, n_channels, n_samples) raw window into a sequence of patch tokens.
    
    1. Temporal Convolution: extracts frequency/spectral features across time.
    2. Spatial Convolution: combines channels for spatial/montage features.
    3. Patching / Pooling: downsamples along the time dimension into discrete patch tokens.
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
        """
        Input x: (B, n_channels, n_samples) or (B, 1, n_channels, n_samples)
        Output: (B, num_patches, embed_dim)
        """
        if x.ndim == 3:
            x = x.unsqueeze(1)  # (B, 1, C, T)

        h = F.elu(self.bn_temp(self.temp_conv(x)))
        h = F.elu(self.bn_spatial(self.spatial_conv(h)))  # (B, embed_dim, 1, T)
        h = self.pool(h)                                  # (B, embed_dim, 1, num_patches)
        h = self.drop(h)

        # Reshape to token sequence: (B, num_patches, embed_dim)
        h = h.squeeze(2).permute(0, 2, 1).contiguous()
        return h


class PositionalEncoding(nn.Module):
    """
    Learnable or sinusoidal positional encoding for token sequences.
    """

    def __init__(self, embed_dim: int, max_len: int = 2000):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input x: (B, seq_len, embed_dim)
        Output: (B, seq_len, embed_dim)
        """
        seq_len = x.shape[1]
        return x + self.pos_embed[:, :seq_len, :]


class CausalTransformerEncoderBlock(nn.Module):
    """
    Pre-norm Causal Transformer Encoder block with Multi-Head Self-Attention (MHSA)
    and Feed-Forward Network (FFN / GEGLU style).
    """

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
        """
        x: (B, seq_len, embed_dim)
        causal_mask: (seq_len, seq_len) boolean / float mask preventing attention to future tokens.
        """
        # Multi-Head Causal Self-Attention
        norm_x = self.ln1(x)
        attn_out, _ = self.attn(
            query=norm_x,
            key=norm_x,
            value=norm_x,
            attn_mask=causal_mask,
            need_weights=False,
        )
        x = x + attn_out

        # FFN
        x = x + self.ffn(self.ln2(x))
        return x


class NextTokenPredictionHead(nn.Module):
    """
    Per-position next-token / patch prediction head.
    Predicts target patch feature vector for position t+1 from hidden state at position t.
    """

    def __init__(self, embed_dim: int, target_dim: Optional[int] = None, dropout: float = 0.1):
        super().__init__()
        out_dim = target_dim if target_dim is not None else embed_dim
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input x: (B, seq_len, embed_dim)
        Output: (B, seq_len, out_dim)
        """
        return self.head(x)


class CausalEEGConformer(nn.Module):
    """
    Causal Generative EEG-Conformer.
    
    Processes EEG windows autoregressively, predicting the next EEG token/patch
    from preceding tokens. Designed to model continuous phase dynamics and
    transitions (e.g. interictal -> preictal -> ictal evolution).
    """

    def __init__(
        self,
        n_channels: int,
        embed_dim: int = 128,
        depth: int = 6,
        num_heads: int = 4,
        ffn_dim: int = 512,
        temp_kernel: int = 25,
        stride: int = 10,
        dropout: float = 0.1,
        max_len: int = 2000,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.embed_dim = embed_dim
        self.stride = stride

        # Front-end Patch Embedding
        self.front_end = ConvFrontEnd(
            n_channels=n_channels,
            embed_dim=embed_dim,
            temp_kernel=temp_kernel,
            stride=stride,
            dropout=dropout,
        )

        # Positional Encoding
        self.pos_encoder = PositionalEncoding(embed_dim=embed_dim, max_len=max_len)

        # Causal Transformer Stack
        self.blocks = nn.ModuleList([
            CausalTransformerEncoderBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                dropout=dropout,
            )
            for _ in range(depth)
        ])

        self.ln_f = nn.LayerNorm(embed_dim)

        # Next-Token Prediction Head
        self.pred_head = NextTokenPredictionHead(embed_dim=embed_dim, target_dim=embed_dim, dropout=dropout)

    def _generate_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Creates lower-triangular mask (0 for valid, -inf for masked future positions).
        """
        mask = torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=device), diagonal=1)
        return mask

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Input x: (B, n_channels, n_samples) or (B, 1, n_channels, n_samples)
        
        Returns:
            predicted_next_tokens: (B, num_patches, embed_dim)
                At position i, predicts the patch token at position i+1.
            patch_tokens: (B, num_patches, embed_dim)
                Ground-truth patch tokens extracted by the conv front-end.
        """
        # 1. Front-end patch extraction
        tokens = self.front_end(x)  # (B, num_patches, embed_dim)
        seq_len = tokens.shape[1]

        # 2. Add positional encoding
        h = self.pos_encoder(tokens)

        # 3. Create Causal Mask
        causal_mask = self._generate_causal_mask(seq_len, device=x.device)

        # 4. Pass through Causal Transformer blocks
        for block in self.blocks:
            h = block(h, causal_mask)

        h = self.ln_f(h)

        # 5. Next-token prediction
        pred_next = self.pred_head(h)  # (B, num_patches, embed_dim)

        return pred_next, tokens

    def compute_loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        Autoregressive next-patch prediction loss (MSE Loss between predicted token_i+1
        and actual target token_i+1).
        """
        pred_next, tokens = self.forward(x)

        # Target for position t (0..N-2) is token at t+1 (1..N-1)
        preds = pred_next[:, :-1, :]
        targets = tokens[:, 1:, :]

        loss = F.mse_loss(preds, targets)
        return loss

    @torch.no_grad()
    def generate(self, seed_x: torch.Tensor, steps: int = 10) -> torch.Tensor:
        """
        Autoregressively generate 'steps' future patch tokens given a seed EEG window.
        
        seed_x: (B, n_channels, n_samples) seed recording.
        returns: generated patch sequence of shape (B, num_seed_patches + steps, embed_dim)
        """
        self.eval()
        tokens = self.front_end(seed_x)  # (B, seed_patches, embed_dim)

        for _ in range(steps):
            seq_len = tokens.shape[1]
            h = self.pos_encoder(tokens)
            causal_mask = self._generate_causal_mask(seq_len, device=seed_x.device)

            for block in self.blocks:
                h = block(h, causal_mask)

            h = self.ln_f(h)
            next_token = self.pred_head(h[:, -1:, :])  # Predict single next token (B, 1, embed_dim)
            tokens = torch.cat([tokens, next_token], dim=1)

        return tokens


if __name__ == "__main__":
    # Quick sanity test
    B, C, T = 2, 19, 1000
    dummy_eeg = torch.randn(B, C, T)

    model = CausalEEGConformer(n_channels=C)
    pred_next, tokens = model(dummy_eeg)
    loss = model.compute_loss(dummy_eeg)

    print(f"Input EEG shape: {dummy_eeg.shape}")
    print(f"Extracted patch tokens shape: {tokens.shape}")
    print(f"Predicted next tokens shape: {pred_next.shape}")
    print(f"Autoregressive next-patch loss: {loss.item():.4f}")

    gen_tokens = model.generate(dummy_eeg, steps=5)
    print(f"Autoregressively generated tokens shape: {gen_tokens.shape}")
