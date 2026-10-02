from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


SUPPORTED_DISCOUNTS = torch.tensor(
    [5., 10., 15., 20., 25., 30., 35., 40.]
)


@dataclass
class SoftPolicyDistillationV2Config:
    state_dim: int = 111
    action_dim: int = 8
    hidden_dims: Sequence[int] = (256, 128, 64)
    lr: float = 8e-4
    weight_decay: float = 1e-5
    temperature: float = 2.0
    seed: int = 42


class SupportedTargetPricingPolicyV2(nn.Module):
    """8-action target pricing policy: 5,10,...,40 percent."""

    def __init__(self, config: SoftPolicyDistillationV2Config):
        super().__init__()
        torch.manual_seed(config.seed)

        dims = [config.state_dim, *config.hidden_dims]
        layers = []

        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.LeakyReLU(0.01))

        layers.append(nn.Linear(dims[-1], config.action_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.network(states)

    def probabilities(
        self,
        states: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        return F.softmax(
            self.forward(states) / max(float(temperature), 1e-6),
            dim=-1,
        )

    def deterministic_action(
        self,
        states: torch.Tensor,
    ) -> torch.Tensor:
        return torch.argmax(self.forward(states), dim=-1)


def discount_from_index(index: torch.Tensor) -> torch.Tensor:
    return 5.0 + 5.0 * index.float()


def discount_index(discount: torch.Tensor) -> torch.Tensor:
    return (
        torch.round(discount / 5.0) - 1.0
    ).clamp(0, 7).long()


def continuous_to_soft_bins(
    discount: torch.Tensor,
    bandwidth: float = 3.0,
) -> torch.Tensor:
    """
    Convert a continuous discount to a normalized soft distribution over
    5,10,...,40%.

    Gaussian RBF weights are used so nearby discount levels retain
    probability mass. This is performed per sample, not per batch.
    """
    levels = SUPPORTED_DISCOUNTS.to(
        device=discount.device,
        dtype=discount.dtype,
    )

    distance = discount.unsqueeze(-1) - levels.unsqueeze(0)
    weights = torch.exp(
        -0.5 * (distance / bandwidth).pow(2)
    )

    return weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
