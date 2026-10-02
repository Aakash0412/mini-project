"""Pseudo-state adapter for NeoRL -> ItaNet transfer."""

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn


@dataclass
class TransferAdapterConfig:
    target_state_dim: int = 111
    source_state_dim: int = 4
    hidden_dims: Sequence[int] = (128, 64)
    dropout: float = 0.0


class TransferAdapter(nn.Module):
    """Map the 111-D ItaNet state into a NeoRL-compatible 4-D pseudo-state.

    The adapter output is constrained to the observed NeoRL state range.
    This keeps the source policy input in the domain on which it was trained.
    """

    def __init__(
        self,
        config: TransferAdapterConfig | None = None,
        source_min: torch.Tensor | None = None,
        source_max: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.config = config or TransferAdapterConfig()

        dims = [self.config.target_state_dim, *self.config.hidden_dims]
        layers: list[nn.Module] = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.extend([nn.Linear(in_dim, out_dim), nn.LeakyReLU(inplace=True)])
            if self.config.dropout > 0:
                layers.append(nn.Dropout(self.config.dropout))
        layers.append(nn.Linear(dims[-1], self.config.source_state_dim))
        self.network = nn.Sequential(*layers)

        if source_min is None:
            source_min = torch.zeros(self.config.source_state_dim, dtype=torch.float32)
        if source_max is None:
            source_max = torch.ones(self.config.source_state_dim, dtype=torch.float32)

        source_min = torch.as_tensor(source_min, dtype=torch.float32)
        source_max = torch.as_tensor(source_max, dtype=torch.float32)

        if source_min.shape != (self.config.source_state_dim,):
            raise ValueError("source_min has the wrong shape")
        if source_max.shape != (self.config.source_state_dim,):
            raise ValueError("source_max has the wrong shape")
        if torch.any(source_max <= source_min):
            raise ValueError("source_max must be greater than source_min")

        self.register_buffer("source_min", source_min)
        self.register_buffer("source_max", source_max)

    def forward(self, target_state: torch.Tensor) -> torch.Tensor:
        if target_state.ndim != 2 or target_state.shape[1] != self.config.target_state_dim:
            raise ValueError(
                f"Expected [N, {self.config.target_state_dim}], "
                f"got {tuple(target_state.shape)}"
            )

        raw = self.network(target_state)
        unit = torch.sigmoid(raw)
        return self.source_min + unit * (self.source_max - self.source_min)

    @torch.no_grad()
    def predict_source_state(self, target_state: torch.Tensor) -> torch.Tensor:
        return self.forward(target_state)
