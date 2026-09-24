from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class ResidueQualityHead(nn.Module):
    """Predict one local quality score for every residue embedding."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int] = (512, 256, 128),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current = input_dim
        for hidden in hidden_dims:
            layers.extend((nn.Linear(current, hidden), nn.ReLU(), nn.Dropout(dropout)))
            current = hidden
        layers.append(nn.Linear(current, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        if embedding.ndim != 2:
            raise ValueError(f"Expected [L, D], got {tuple(embedding.shape)}")
        return self.network(embedding).squeeze(-1)

