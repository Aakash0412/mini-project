from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from src.models.ppo_agent import PPOAgent


@dataclass
class PPOConfig:
    gamma: float = 0.9
    gae_lambda: float = 0.8
    clip_epsilon: float = 0.2
    entropy_coefficient: float = 0.01
    value_coefficient: float = 0.5
    batch_size: int = 64
    update_epochs: int = 4
    max_grad_norm: float = 0.5


class PPORolloutBuffer:
    """
    Stores one PPO rollout.
    """

    def __init__(self):
        self.states: List[np.ndarray] = []
        self.actions: List[int] = []
        self.rewards: List[float] = []
        self.log_probabilities: List[float] = []
        self.values: List[float] = []
        self.dones: List[bool] = []

    def add(
        self,
        state,
        action,
        reward,
        log_probability,
        value,
        done,
    ):
        self.states.append(np.asarray(state, dtype=np.float32))
        self.actions.append(int(action))
        self.rewards.append(float(reward))
        self.log_probabilities.append(float(log_probability))
        self.values.append(float(value))
        self.dones.append(bool(done))

    def __len__(self):
        return len(self.states)

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.log_probabilities.clear()
        self.values.clear()
        self.dones.clear()


class PPOTrainer:
    """
    PPO optimization logic.

    Implements:
    - Generalized Advantage Estimation (GAE)
    - Advantage normalization
    - PPO clipped policy objective
    - Value-function objective
    - Entropy regularization
    - Gradient clipping
    """

    def __init__(
        self,
        agent: PPOAgent,
        config: PPOConfig = None,
        device: str = "cpu",
    ):
        self.agent = agent
        self.config = config or PPOConfig()

        self.device = torch.device(device)

        self.agent.actor.to(self.device)
        self.agent.critic.to(self.device)

    def compute_gae(
        self,
        rewards,
        values,
        dones,
        next_value,
    ):
        """
        Compute GAE advantages and discounted returns.
        """

        rewards = np.asarray(rewards, dtype=np.float32)
        values = np.asarray(values, dtype=np.float32)
        dones = np.asarray(dones, dtype=np.float32)

        advantages = np.zeros_like(rewards)

        gae = 0.0

        for step in reversed(range(len(rewards))):
            if step == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[step]
                next_value_step = next_value
            else:
                next_non_terminal = 1.0 - dones[step]
                next_value_step = values[step + 1]

            delta = (
                rewards[step]
                + self.config.gamma
                * next_value_step
                * next_non_terminal
                - values[step]
            )

            gae = (
                delta
                + self.config.gamma
                * self.config.gae_lambda
                * next_non_terminal
                * gae
            )

            advantages[step] = gae

        returns = advantages + values

        return advantages, returns

    def update(
        self,
        buffer: PPORolloutBuffer,
        next_value: float = 0.0,
    ) -> Dict[str, float]:
        """
        Perform PPO optimization over a collected rollout.
        """

        if len(buffer) == 0:
            raise ValueError("Cannot update PPO with an empty buffer.")

        states = torch.tensor(
            np.asarray(buffer.states),
            dtype=torch.float32,
            device=self.device,
        )

        actions = torch.tensor(
            np.asarray(buffer.actions),
            dtype=torch.long,
            device=self.device,
        )

        old_log_probabilities = torch.tensor(
            np.asarray(buffer.log_probabilities),
            dtype=torch.float32,
            device=self.device,
        )

        values = np.asarray(buffer.values, dtype=np.float32)

        advantages, returns = self.compute_gae(
            rewards=buffer.rewards,
            values=values,
            dones=buffer.dones,
            next_value=float(next_value),
        )

        advantages = torch.tensor(
            advantages,
            dtype=torch.float32,
            device=self.device,
        )

        returns = torch.tensor(
            returns,
            dtype=torch.float32,
            device=self.device,
        )

        # Normalize advantages for stable PPO optimization.
        if len(advantages) > 1:
            advantages = (
                advantages - advantages.mean()
            ) / (
                advantages.std(unbiased=False) + 1e-8
            )

        dataset_size = len(states)

        actor_losses = []
        critic_losses = []
        entropy_values = []

        for _ in range(self.config.update_epochs):

            permutation = torch.randperm(
                dataset_size,
                device=self.device,
            )

            for start in range(
                0,
                dataset_size,
                self.config.batch_size,
            ):
                indices = permutation[
                    start:start + self.config.batch_size
                ]

                batch_states = states[indices]
                batch_actions = actions[indices]
                batch_old_log_probs = old_log_probabilities[indices]
                batch_advantages = advantages[indices]
                batch_returns = returns[indices]

                new_log_probs, new_values, entropy = (
                    self.agent.evaluate_actions(
                        batch_states,
                        batch_actions,
                    )
                )

                probability_ratio = torch.exp(
                    new_log_probs - batch_old_log_probs
                )

                unclipped_objective = (
                    probability_ratio * batch_advantages
                )

                clipped_ratio = torch.clamp(
                    probability_ratio,
                    1.0 - self.config.clip_epsilon,
                    1.0 + self.config.clip_epsilon,
                )

                clipped_objective = (
                    clipped_ratio * batch_advantages
                )

                actor_loss = -torch.min(
                    unclipped_objective,
                    clipped_objective,
                ).mean()

                critic_loss = F.mse_loss(
                    new_values,
                    batch_returns,
                )

                entropy_mean = entropy.mean()

                total_loss = (
                    actor_loss
                    + self.config.value_coefficient * critic_loss
                    - self.config.entropy_coefficient * entropy_mean
                )

                self.agent.actor_optimizer.zero_grad()
                self.agent.critic_optimizer.zero_grad()

                total_loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    self.agent.actor.parameters(),
                    self.config.max_grad_norm,
                )

                torch.nn.utils.clip_grad_norm_(
                    self.agent.critic.parameters(),
                    self.config.max_grad_norm,
                )

                self.agent.actor_optimizer.step()
                self.agent.critic_optimizer.step()

                actor_losses.append(float(actor_loss.item()))
                critic_losses.append(float(critic_loss.item()))
                entropy_values.append(float(entropy_mean.item()))

        return {
            "actor_loss": float(np.mean(actor_losses)),
            "critic_loss": float(np.mean(critic_losses)),
            "entropy": float(np.mean(entropy_values)),
            "mean_advantage": float(advantages.mean().item()),
            "mean_return": float(returns.mean().item()),
            "samples": float(dataset_size),
        }