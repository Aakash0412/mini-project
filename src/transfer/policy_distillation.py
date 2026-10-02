from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn


@dataclass(frozen=True)
class PolicyDistillationConfig:
    state_dim: int = 111
    action_dim: int = 9
    hidden_dims: Sequence[int] = (256, 128, 64)
    learning_rate: float = 8e-4
    weight_decay: float = 1e-5
    temperature: float = 1.0
    seed: int = 42


class TargetPricingPolicy(nn.Module):
    """Discrete 9-action target policy used for transfer distillation."""

    def __init__(self, config: PolicyDistillationConfig):
        super().__init__()
        torch.manual_seed(config.seed)

        dims = [config.state_dim, *config.hidden_dims]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.extend([nn.Linear(dims[i], dims[i + 1]), nn.LeakyReLU(0.1)])
        layers.append(nn.Linear(dims[-1], config.action_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.network(states)

    def probabilities(self, states: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        return torch.softmax(self.forward(states) / max(float(temperature), 1e-6), dim=-1)

    def deterministic_action(self, states: torch.Tensor) -> torch.Tensor:
        return torch.argmax(self.forward(states), dim=-1)


def discount_index(discount: torch.Tensor) -> torch.Tensor:
    """Map a 5..40% discount to target action index 1..8; 0% is index 0."""
    return torch.clamp(torch.round(discount / 5.0), 0, 8).long()


def discount_from_index(index: torch.Tensor) -> torch.Tensor:
    return index.float() * 5.0
