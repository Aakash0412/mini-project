from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import DDPG
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.utils import set_random_seed

from src.environment.ppo_dataset import load_ppo_dataset
from src.environment.reward import SALES_REWARD
from src.models.baselines.continuous_pricing_gym_env import (
    ContinuousPricingGymEnv,
)


SEED = 42
TOTAL_TIMESTEPS = 5_000

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models" / "ddpg"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "ddpg_sales_seed42"
METADATA_PATH = MODEL_DIR / "ddpg_sales_seed42_metadata.json"


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    set_random_seed(SEED)

    dataset = load_ppo_dataset()
    env = Monitor(
        ContinuousPricingGymEnv(
            dataset=dataset,
            reward_config=SALES_REWARD,
            seed=SEED,
        )
    )

    action_noise = NormalActionNoise(
        mean=np.zeros(1, dtype=np.float32),
        sigma=0.2 * np.ones(1, dtype=np.float32),
    )

    model = DDPG(
        policy="MlpPolicy",
        env=env,
        learning_rate=1e-3,
        buffer_size=10_000,
        learning_starts=1_000,
        batch_size=64,
        tau=0.005,
        gamma=0.9,
        train_freq=(1, "step"),
        gradient_steps=1,
        action_noise=action_noise,
        policy_kwargs={"net_arch": [128, 128]},
        seed=SEED,
        verbose=1,
        device="cpu",
    )

    print(f"Dataset products: {len(dataset.states)}")
    print(f"State dimension: {env.observation_space.shape}")
    print(f"Action dimension: {env.action_space.shape}")
    print(f"Reward configuration: {SALES_REWARD}")
    print(f"Training timesteps: {TOTAL_TIMESTEPS}")

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        progress_bar=False,
    )

    model.save(str(MODEL_PATH))

    metadata = {
        "algorithm": "DDPG",
        "seed": SEED,
        "total_timesteps": TOTAL_TIMESTEPS,
        "dataset_products": int(len(dataset.states)),
        "state_dimension": 111,
        "action_space": "continuous [-1, 1], mapped to nearest supported discount",
        "discount_bins": [0, 5, 10, 15, 20, 25, 30, 35, 40],
        "reward_name": SALES_REWARD.name,
        "alpha": SALES_REWARD.alpha,
        "beta": SALES_REWARD.beta,
        "learning_rate": 1e-3,
        "buffer_size": 10_000,
        "learning_starts": 1_000,
        "batch_size": 64,
        "tau": 0.005,
        "gamma": 0.9,
        "policy_architecture": [128, 128],
        "seed_scope": "single training seed; repeat-seed robustness not yet evaluated",
        "environment_type": "one-step counterfactual pricing environment",
        "limitations": [
            "Continuous actions are quantized to nine discrete discount bins.",
            "0% discount is outside the observed 5%-40% training support.",
            "Rewards depend on a frozen sales predictor and proxy economics.",
            "This is not a sequential market simulator.",
        ],
    }
    METADATA_PATH.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print(f"DDPG model saved: {MODEL_PATH}.zip")
    print(f"Metadata saved: {METADATA_PATH}")
    env.close()


if __name__ == "__main__":
    main()


