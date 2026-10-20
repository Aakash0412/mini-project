from __future__ import annotations

from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.environment.ppo_dataset import PPODataset
from src.environment.pricing_environment import DynamicPricingEnvironment
from src.environment.reward import RewardConfig, SALES_REWARD


class PricingGymEnv(gym.Env):
    """Gymnasium adapter for the project's one-step pricing environment."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        dataset: PPODataset,
        indices: Optional[np.ndarray] = None,
        reward_config: RewardConfig = SALES_REWARD,
        seed: Optional[int] = None,
    ):
        super().__init__()

        if len(dataset.states) != len(dataset.products):
            raise ValueError("States and products must have equal lengths.")

        self.dataset = dataset
        self.indices = (
            np.arange(len(dataset.states), dtype=np.int64)
            if indices is None
            else np.asarray(indices, dtype=np.int64)
        )

        if self.indices.ndim != 1 or len(self.indices) == 0:
            raise ValueError("indices must be a non-empty 1-D array.")

        if np.any(self.indices < 0) or np.any(self.indices >= len(dataset.states)):
            raise ValueError("indices contain out-of-range dataset positions.")

        self.pricing_env = DynamicPricingEnvironment(
            reward_config=reward_config
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(111,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(9)
        self._episode_done = True

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        if options and "position" in options:
            position = int(options["position"])
            if position not in self.indices:
                raise ValueError(
                    "Requested position is not in this environment's indices."
                )
        else:
            position = int(self.np_random.choice(self.indices))

        self._episode_done = False
        state = self.pricing_env.reset(
            state=self.dataset.states[position],
            product=self.dataset.products[position],
        )
        return state.astype(np.float32), {
            "dataset_position": position,
            "source_row_index": int(
                self.dataset.source_row_indices[position]
            ),
        }

    def step(self, action):
        if self._episode_done:
            raise RuntimeError("Episode ended. Call reset() before step().")
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")

        next_state, reward, terminated, info = self.pricing_env.step(int(action))
        self._episode_done = bool(terminated)
        return (
            next_state.astype(np.float32),
            float(reward),
            bool(terminated),
            False,
            info,
        )
