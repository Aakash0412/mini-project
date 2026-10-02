from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
from torch.distributions import Normal


LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


@dataclass
class NeoRLSourcePolicyConfig:
    state_dim: int = 4
    action_dim: int = 2

    hidden_dims: Tuple[int, ...] = (256, 128, 64)

    learning_rate: float = 8e-4
    weight_decay: float = 1e-5

    seed: int = 42


class NeoRLSourcePolicy(nn.Module):
    """
    Continuous Gaussian policy for the NeoRL Sales Promotion source domain.

    Input:
        4-D NeoRL state

    Output:
        2-D normalized action:
            action[0] = normalized coupon count
            action[1] = normalized coupon discount factor

    The action representation follows the actual NeoRL dataset/API.
    """

    def __init__(self, config: NeoRLSourcePolicyConfig | None = None):
        super().__init__()

        self.config = config or NeoRLSourcePolicyConfig()

        torch.manual_seed(self.config.seed)

        layers = []
        input_dim = self.config.state_dim

        for hidden_dim in self.config.hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.LeakyReLU(inplace=True))
            input_dim = hidden_dim

        self.backbone = nn.Sequential(*layers)

        self.mean_layer = nn.Linear(
            input_dim,
            self.config.action_dim,
        )

        self.log_std_layer = nn.Linear(
            input_dim,
            self.config.action_dim,
        )

    def forward(self, states: torch.Tensor):
        x = self.backbone(states)

        mean = torch.sigmoid(self.mean_layer(x))

        log_std = self.log_std_layer(x)
        log_std = torch.clamp(
            log_std,
            LOG_STD_MIN,
            LOG_STD_MAX,
        )

        std = torch.exp(log_std)

        return mean, std, log_std

    def distribution(self, states: torch.Tensor):
        mean, std, _ = self.forward(states)
        return Normal(mean, std)

    def log_prob(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
    ):
        distribution = self.distribution(states)

        return distribution.log_prob(actions).sum(
            dim=-1,
            keepdim=True,
        )

    @torch.no_grad()
    def deterministic_action(
        self,
        states: torch.Tensor,
    ):
        mean, _, _ = self.forward(states)

        return torch.clamp(
            mean,
            min=0.0,
            max=1.0,
        )

    @torch.no_grad()
    def sample_action(
        self,
        states: torch.Tensor,
    ):
        distribution = self.distribution(states)

        action = distribution.sample()

        return torch.clamp(
            action,
            min=0.0,
            max=1.0,
        )