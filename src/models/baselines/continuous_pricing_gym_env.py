from __future__ import annotations

from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.environment.ppo_dataset import PPODataset
from src.environment.pricing_environment import DynamicPricingEnvironment
from src.environment.reward import RewardConfig, SALES_REWARD
from src.environment.counterfactual_simulator import DISCOUNT_BINS


class ContinuousPricingGymEnv(gym.Env):
    """One-step pricing environment with continuous actions for DDPG/SAC.

    The agent emits a scalar in [-1, 1]. It is mapped to a discount in
    [0, 40] percent, then snapped to the nearest supported discount bin.
    """

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

        self.discount_bins = np.asarray(DISCOUNT_BINS, dtype=np.float32)
        if self.discount_bins.ndim != 1 or len(self.discount_bins) == 0:
            raise ValueError("DISCOUNT_BINS must be a non-empty 1-D array.")

        self.pricing_env = DynamicPricingEnvironment(
            reward_config=reward_config
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(111,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32,
        )
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

        action = np.asarray(action, dtype=np.float32)
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid continuous action: {action}")

        proposed_discount = float((action.reshape(-1)[0] + 1.0) * 20.0)
        discount_index = int(
            np.argmin(np.abs(self.discount_bins - proposed_discount))
        )
        selected_discount = float(self.discount_bins[discount_index])

        next_state, reward, terminated, info = self.pricing_env.step(
            discount_index
        )
        self._episode_done = bool(terminated)

        info = dict(info)
        info["proposed_discount"] = proposed_discount
        info["selected_discount"] = selected_discount
        info["discount_index"] = discount_index

        return (
            next_state.astype(np.float32),
            float(reward),
            bool(terminated),
            False,
            info,
        )
