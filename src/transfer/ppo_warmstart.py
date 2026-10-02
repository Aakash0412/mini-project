from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.models.ppo_agent import PPOAgent
from src.transfer.neorl_source_policy import (
    NeoRLSourcePolicy,
    NeoRLSourcePolicyConfig,
)
from src.transfer.transfer_adapter import (
    TransferAdapter,
    TransferAdapterConfig,
)


@dataclass
class PPOWarmStartConfig:
    state_dim: int = 111
    action_dim: int = 8

    hidden_dims: Sequence[int] = (128, 128)

    learning_rate: float = 8e-4
    weight_decay: float = 1e-5

    batch_size: int = 512
    epochs: int = 10

    temperature: float = 1.0
    seed: int = 42


# ---------------------------------------------------------------------------
# Source-action -> target-discount mapping
# ---------------------------------------------------------------------------

TARGET_DISCOUNTS = np.array(
    [5, 10, 15, 20, 25, 30, 35, 40],
    dtype=np.float32,
)

SOURCE_LEVELS = np.array(
    [0.19, 0.18, 0.17, 0.16, 0.15, 0.14, 0.13, 0.12],
    dtype=np.float32,
)


# ---------------------------------------------------------------------------
# Target PPO actor
# ---------------------------------------------------------------------------

class WarmStartActor(torch.nn.Module):
    """
    Target-domain discrete actor.

    Input:
        111-D ItaNet state

    Output:
        8 action logits corresponding to:

        5, 10, 15, 20, 25, 30, 35, 40%
    """

    def __init__(
        self,
        state_dim: int = 111,
        action_dim: int = 8,
        hidden_dims: Sequence[int] = (128, 128),
    ) -> None:
        super().__init__()

        dims = [
            state_dim,
            *hidden_dims,
            action_dim,
        ]

        layers: list[torch.nn.Module] = []

        for in_dim, out_dim in zip(
            dims[:-1],
            dims[1:],
        ):
            layers.append(
                torch.nn.Linear(
                    in_dim,
                    out_dim,
                )
            )

            if out_dim != action_dim:
                layers.append(
                    torch.nn.ReLU()
                )

        self.network = torch.nn.Sequential(
            *layers
        )

    def forward(
        self,
        state: torch.Tensor,
    ) -> torch.Tensor:
        if state.ndim != 2:
            raise ValueError(
                f"Expected [N, state_dim], got {tuple(state.shape)}"
            )

        if state.shape[1] != self.network[0].in_features:
            raise ValueError(
                "Incorrect state dimension"
            )

        return self.network(state)


# ---------------------------------------------------------------------------
# Gaussian -> categorical target distribution
# ---------------------------------------------------------------------------

def gaussian_discount_distribution(
    mean_action: torch.Tensor,
    std_action: torch.Tensor,
) -> torch.Tensor:
    """
    Convert a Gaussian source-policy distribution over the normalized
    NeoRL discount-factor action into a categorical distribution over
    target discount levels:

        [5, 10, 15, 20, 25, 30, 35, 40]

    The source action is normalized such that:
        0.20 -> 10%
        0.30 -> 15%
        ...
        0.80 -> 40%

    Probability mass is assigned to each target action using midpoint
    boundaries between adjacent discount levels.
    """

    if mean_action.ndim != 2 or std_action.ndim != 2:
        raise ValueError(
            "mean_action and std_action must have shape [batch, action_dim]"
        )

    if mean_action.shape != std_action.shape:
        raise ValueError(
            "mean_action and std_action must have identical shapes"
        )

    if mean_action.shape[1] < 2:
        raise ValueError(
            "Expected source action dimension >= 2"
        )

    # NeoRL action dimension 1 is the coupon/discount-factor action.
    mean = mean_action[:, 1]
    std = std_action[:, 1].clamp_min(1e-6)

    # Target discount levels:
    # 5,10,...,40% correspond to normalized factors:
    # 0.1,0.2,...,0.8 under the local mapping.
    #
    # Midpoints between adjacent levels define categorical boundaries.
    centers = torch.arange(
        0.1,
        0.81,
        0.1,
        device=mean.device,
        dtype=mean.dtype,
    )

    boundaries = (centers[:-1] + centers[1:]) / 2.0

    # Shape:
    # mean/std       -> [B]
    # boundaries     -> [7]
    #
    # Explicitly expand them to [B, 7] so torch.distributions.Normal
    # receives compatible tensors.
    normal = torch.distributions.Normal(
        mean.unsqueeze(1),
        std.unsqueeze(1),
    )

    boundary_tensor = boundaries.unsqueeze(0).expand(
        mean.shape[0],
        -1,
    )

    cdf = normal.cdf(boundary_tensor)

    # Probability of each categorical bin.
    probabilities = torch.cat(
        [
            cdf[:, :1],
            cdf[:, 1:] - cdf[:, :-1],
            1.0 - cdf[:, -1:],
        ],
        dim=1,
    )

    # Numerical protection.
    probabilities = probabilities.clamp_min(1e-8)

    # Re-normalize so every row sums exactly to 1.
    probabilities = probabilities / probabilities.sum(
        dim=1,
        keepdim=True,
    )

    return probabilities
    
# ---------------------------------------------------------------------------
# Load adapter
# ---------------------------------------------------------------------------

def load_adapter(
    checkpoint_path: Path,
    device: torch.device,
) -> TransferAdapter:

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    config_dict = checkpoint["config"]
    state_dict = checkpoint["model_state_dict"]

    config = TransferAdapterConfig(
        target_state_dim=int(
            config_dict["target_state_dim"]
        ),
        source_state_dim=int(
            config_dict["source_state_dim"]
        ),
        hidden_dims=(128, 64),
        dropout=0.0,
    )

    adapter = TransferAdapter(
        config=config,
        source_min=state_dict["source_min"],
        source_max=state_dict["source_max"],
    ).to(device)

    adapter.load_state_dict(
        state_dict
    )

    adapter.eval()

    return adapter


# ---------------------------------------------------------------------------
# Load NeoRL source policy
# ---------------------------------------------------------------------------

def load_source_policy(
    checkpoint_path: Path,
    device: torch.device,
) -> NeoRLSourcePolicy:

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    config_dict = checkpoint["config"]

    config = NeoRLSourcePolicyConfig(
        state_dim=int(
            config_dict["state_dim"]
        ),
        action_dim=int(
            config_dict["action_dim"]
        ),
        hidden_dims=tuple(
            config_dict["hidden_dims"]
        ),
        learning_rate=float(
            config_dict["learning_rate"]
        ),
        weight_decay=float(
            config_dict["weight_decay"]
        ),
        seed=int(
            config_dict["seed"]
        ),
    )

    policy = NeoRLSourcePolicy(
        config
    ).to(device)

    policy.load_state_dict(
        checkpoint["model_state_dict"]
    )

    policy.eval()

    return policy


# ---------------------------------------------------------------------------
# Build soft transfer targets
# ---------------------------------------------------------------------------

@torch.no_grad()
def build_transfer_targets(
    states: torch.Tensor,
    adapter: TransferAdapter,
    source_policy: NeoRLSourcePolicy,
) -> torch.Tensor:

    pseudo_states = adapter(
        states
    )

    mean_action, std_action, _ = (
        source_policy.forward(
            pseudo_states
        )
    )

    target_distribution = (
        gaussian_discount_distribution(
            mean_action,
            std_action,
        )
    )

    return target_distribution


# ---------------------------------------------------------------------------
# Train target actor
# ---------------------------------------------------------------------------

def train_warm_start_actor(
    states: np.ndarray,
    adapter: TransferAdapter,
    source_policy: NeoRLSourcePolicy,
    config: PPOWarmStartConfig,
    device: torch.device,
) -> tuple[WarmStartActor, list[dict]]:

    torch.manual_seed(
        config.seed
    )

    np.random.seed(
        config.seed
    )

    state_tensor = torch.as_tensor(
        states,
        dtype=torch.float32,
    )

    target_distribution = build_transfer_targets(
        state_tensor.to(device),
        adapter,
        source_policy,
    ).cpu()

    dataset = TensorDataset(
        state_tensor,
        target_distribution,
    )

    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
    )

    actor = WarmStartActor(
        state_dim=config.state_dim,
        action_dim=config.action_dim,
        hidden_dims=config.hidden_dims,
    ).to(device)

    optimizer = torch.optim.AdamW(
        actor.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    history: list[dict] = []

    best_loss = float("inf")
    best_state = None

    for epoch in range(
        1,
        config.epochs + 1,
    ):

        actor.train()

        losses = []

        for batch_states, batch_targets in loader:

            batch_states = (
                batch_states.to(device)
            )

            batch_targets = (
                batch_targets.to(device)
            )

            logits = actor(
                batch_states
            )

            log_probabilities = F.log_softmax(
                logits / config.temperature,
                dim=1,
            )

            loss = F.kl_div(
                log_probabilities,
                batch_targets,
                reduction="batchmean",
            )

            optimizer.zero_grad()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                actor.parameters(),
                max_norm=1.0,
            )

            optimizer.step()

            losses.append(
                float(loss.item())
            )

        mean_loss = float(
            np.mean(losses)
        )

        with torch.no_grad():

            actor.eval()

            logits = actor(
                state_tensor.to(device)
            )

            probabilities = F.softmax(
                logits,
                dim=1,
            )

            entropy = -(
                probabilities
                * torch.log(
                    probabilities + 1e-12
                )
            ).sum(dim=1).mean()

            argmax_accuracy = (
                probabilities.argmax(dim=1)
                == target_distribution.argmax(dim=1).to(device)
            ).float().mean()

        row = {
            "epoch": epoch,
            "loss": mean_loss,
            "entropy": float(
                entropy.item()
            ),
            "argmax_match": float(
                argmax_accuracy.item()
            ),
        }

        history.append(
            row
        )

        print(
            f"Epoch {epoch:02d}/{config.epochs} | "
            f"KL {mean_loss:.6f} | "
            f"Entropy {row['entropy']:.6f} | "
            f"Argmax match "
            f"{row['argmax_match'] * 100:.2f}%"
        )

        if mean_loss < best_loss:

            best_loss = mean_loss

            best_state = {
                key: value.detach().cpu().clone()
                for key, value in actor.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError(
            "Warm-start training produced no checkpoint."
        )

    actor.load_state_dict(
        best_state
    )

    return actor, history


# ---------------------------------------------------------------------------
# Convert warm-start actor to the existing 9-action PPO actor
# ---------------------------------------------------------------------------

def expand_to_nine_actions(
    warm_actor: WarmStartActor,
    device: torch.device,
) -> PPOAgent:
    """
    Convert the 8-action transfer actor into the existing 9-action PPO
    architecture.

    Existing PPO action order:

        0,5,10,15,20,25,30,35,40

    Transfer actor order:

        5,10,15,20,25,30,35,40

    The new 0% action receives a deliberately neutral low initial logit
    rather than being copied from any source action because 0% has no
    NeoRL counterpart and is outside observed Amazon training support.
    """

    agent = PPOAgent(
        state_dim=111,
        action_dim=9,
        actor_lr=8e-4,
        critic_lr=8e-4,
    )

    source_state = warm_actor.state_dict()

    target_state = agent.actor.state_dict()

    # Copy first two hidden layers.
    target_state["network.0.weight"].copy_(
        source_state["network.0.weight"]
    )

    target_state["network.0.bias"].copy_(
        source_state["network.0.bias"]
    )

    target_state["network.2.weight"].copy_(
        source_state["network.2.weight"]
    )

    target_state["network.2.bias"].copy_(
        source_state["network.2.bias"]
    )

    # Copy the 8 transfer action logits into target actions 1..8.
    #
    # Target action index:
    #
    # 0 -> 0%
    # 1 -> 5%
    # ...
    # 8 -> 40%
    #
    # Transfer actor index:
    #
    # 0 -> 5%
    # ...
    # 7 -> 40%
    target_state[
        "network.4.weight"
    ][1:].copy_(
        source_state[
            "network.4.weight"
        ]
    )

    target_state[
        "network.4.bias"
    ][1:].copy_(
        source_state[
            "network.4.bias"
        ]
    )

    # Initialize the unsupported 0% action conservatively.
    #
    # We deliberately do not assign it an arbitrary source meaning.
    target_state[
        "network.4.weight"
    ][0].zero_()

    target_state[
        "network.4.bias"
    ][0] = -2.0

    agent.actor.load_state_dict(
        target_state
    )

    return agent