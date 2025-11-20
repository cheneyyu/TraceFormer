from typing import Optional

import torch
from torch import nn


def generate_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """Causal mask for transformer decoder (shape: seq_len x seq_len)."""
    return torch.triu(torch.ones(seq_len, seq_len, device=device) * float("-inf"), diagonal=1)


class Traceformer(nn.Module):
    """Continuous-space autoregressive transformer for scRNA trajectories."""

    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.projector = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.time_proj = nn.Linear(1, d_model)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.velocity_head = nn.Linear(d_model, input_dim)

    def forward(
        self,
        x: torch.Tensor,
        stemness: torch.Tensor,
        src_memory: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Autoregressively predict next-state expression.

        Parameters
        ----------
        x:
            Input expression tensor of shape (batch, seq_len, input_dim) using log1p HVGs.
        stemness:
            Normalised stemness scores in [0, 1] with shape (batch, seq_len, 1).
        src_memory:
            Optional memory for the decoder. Defaults to a zero vector.
        padding_mask:
            Boolean mask with shape (batch, seq_len) where True marks padding tokens.
        """
        batch_size, seq_len, _ = x.shape
        device = x.device
        tgt = self.projector(x) + self.time_proj(stemness)
        tgt = tgt.transpose(0, 1)  # (seq_len, batch, d_model)

        if src_memory is None:
            src_memory = torch.zeros(1, batch_size, tgt.size(-1), device=device)

        tgt_mask = generate_causal_mask(seq_len, device)
        decoded = self.decoder(
            tgt,
            src_memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=padding_mask,
        )
        decoded = decoded.transpose(0, 1)  # (batch, seq_len, d_model)
        delta = self.velocity_head(decoded)
        return x + delta


class DirectionAwareLoss(nn.Module):
    """Combine cosine similarity and MSE for direction-aware prediction."""

    def __init__(self, cosine_weight: float = 1.0, mse_weight: float = 0.1) -> None:
        super().__init__()
        self.cosine_weight = cosine_weight
        self.mse_weight = mse_weight
        self.cosine = nn.CosineSimilarity(dim=-1)
        self.mse = nn.MSELoss()

    def forward(self, pred_next: torch.Tensor, true_next: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
        pred_dir = pred_next - current
        true_dir = true_next - current
        cosine_loss = 1 - self.cosine(pred_dir, true_dir).mean()
        mse_loss = self.mse(pred_next, true_next)
        return self.cosine_weight * cosine_loss + self.mse_weight * mse_loss
