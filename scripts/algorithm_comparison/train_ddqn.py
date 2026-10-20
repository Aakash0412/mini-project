from __future__ import annotations

import json
import random
from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.environment.ppo_dataset import load_ppo_dataset
from src.environment.reward import SALES_REWARD
from src.models.baselines.pricing_gym_env import PricingGymEnv


SEED = 42
TOTAL_TIMESTEPS = 5_000
BUFFER_SIZE = 10_000
LEARNING_STARTS = 1_000
BATCH_SIZE = 64
GAMMA = 0.9
LEARNING_RATE = 1e-3
TARGET_UPDATE_INTERVAL = 250
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY_STEPS = 4_000
STATE_DIM = 111
ACTION_DIM = 9
DISCOUNT_BINS = [0, 5, 10, 15, 20, 25, 30, 35, 40]

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models" / "ddqn"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "ddqn_sales_seed42.pth"
METADATA_PATH = MODEL_DIR / "ddqn_sales_seed42_metadata.json"


class QNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(STATE_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, ACTION_DIM),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.network(states)


def main() -> None:
    if MODEL_PATH.exists() or METADATA_PATH.exists():
        raise FileExistsError(
            "DDQN output already exists. Preserve it and choose a new "
            "run name before training again."
        )

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    dataset = load_ppo_dataset()
    env = PricingGymEnv(
        dataset=dataset,
        reward_config=SALES_REWARD,
        seed=SEED,
    )

    online = QNetwork().cpu()
    target = QNetwork().cpu()
    target.load_state_dict(online.state_dict())
    target.eval()

    optimizer = torch.optim.Adam(
        online.parameters(),
        lr=LEARNING_RATE,
    )
    replay = deque(maxlen=BUFFER_SIZE)

    episode_reward = 0.0
    completed_episodes = 0
    loss_values = []
    state, _ = env.reset(seed=SEED)

    print("Algorithm: Double DQN (custom implementation)")
    print("Dataset products:", len(dataset.states))
    print("State dimension:", STATE_DIM)
    print("Discrete actions:", list(enumerate(DISCOUNT_BINS)))
    print("Reward configuration:", SALES_REWARD)
    print("Training timesteps:", TOTAL_TIMESTEPS)

    for timestep in range(1, TOTAL_TIMESTEPS + 1):
        fraction = min(1.0, timestep / EPSILON_DECAY_STEPS)
        epsilon = EPSILON_START + fraction * (
            EPSILON_END - EPSILON_START
        )

        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                state_tensor = torch.as_tensor(
                    state, dtype=torch.float32
                ).unsqueeze(0)
                action = int(online(state_tensor).argmax(dim=1).item())

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = bool(terminated or truncated)

        replay.append(
            (
                state.copy(),
                int(action),
                float(reward),
                next_state.copy(),
                float(done),
            )
        )

        episode_reward += float(reward)

        if timestep >= LEARNING_STARTS and len(replay) >= BATCH_SIZE:
            batch = random.sample(replay, BATCH_SIZE)
            states, actions, rewards, next_states, dones = zip(*batch)

            states_t = torch.as_tensor(
                np.asarray(states), dtype=torch.float32
            )
            actions_t = torch.as_tensor(
                actions, dtype=torch.int64
            ).unsqueeze(1)
            rewards_t = torch.as_tensor(
                rewards, dtype=torch.float32
            ).unsqueeze(1)
            next_states_t = torch.as_tensor(
                np.asarray(next_states), dtype=torch.float32
            )
            dones_t = torch.as_tensor(
                dones, dtype=torch.float32
            ).unsqueeze(1)

            current_q = online(states_t).gather(1, actions_t)

            # Double DQN: online network selects; target network evaluates.
            with torch.no_grad():
                next_actions = online(next_states_t).argmax(
                    dim=1, keepdim=True
                )
                next_q = target(next_states_t).gather(1, next_actions)
                target_q = rewards_t + GAMMA * (1.0 - dones_t) * next_q

            loss = F.smooth_l1_loss(current_q, target_q)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(online.parameters(), max_norm=10.0)
            optimizer.step()
            loss_values.append(float(loss.item()))

        if timestep % TARGET_UPDATE_INTERVAL == 0:
            target.load_state_dict(online.state_dict())

        if done:
            completed_episodes += 1
            if completed_episodes % 500 == 0:
                print(
                    f"Step {timestep}/{TOTAL_TIMESTEPS} | "
                    f"Episodes {completed_episodes} | "
                    f"Last episode reward {episode_reward:.6f} | "
                    f"Epsilon {epsilon:.3f}"
                )
            episode_reward = 0.0
            state, _ = env.reset()
        else:
            state = next_state

    torch.save(
        {
            "algorithm": "DDQN",
            "seed": SEED,
            "total_timesteps": TOTAL_TIMESTEPS,
            "state_dim": STATE_DIM,
            "action_dim": ACTION_DIM,
            "discount_bins": DISCOUNT_BINS,
            "online_state_dict": online.state_dict(),
            "target_state_dict": target.state_dict(),
        },
        MODEL_PATH,
    )

    metadata = {
        "algorithm": "Double DQN",
        "implementation": "custom PyTorch online-selection/target-evaluation",
        "seed": SEED,
        "total_timesteps": TOTAL_TIMESTEPS,
        "dataset_products": int(len(dataset.states)),
        "state_dimension": STATE_DIM,
        "action_dimension": ACTION_DIM,
        "discount_bins": DISCOUNT_BINS,
        "reward_name": SALES_REWARD.name,
        "alpha": SALES_REWARD.alpha,
        "beta": SALES_REWARD.beta,
        "learning_rate": LEARNING_RATE,
        "buffer_size": BUFFER_SIZE,
        "learning_starts": LEARNING_STARTS,
        "batch_size": BATCH_SIZE,
        "gamma": GAMMA,
        "target_update_interval": TARGET_UPDATE_INTERVAL,
        "epsilon_start": EPSILON_START,
        "epsilon_end": EPSILON_END,
        "epsilon_decay_steps": EPSILON_DECAY_STEPS,
        "policy_architecture": [128, 128],
        "loss": "smooth_l1",
        "gradient_clipping_norm": 10.0,
        "completed_one_step_episodes": completed_episodes,
        "mean_training_loss": (
            float(np.mean(loss_values)) if loss_values else None
        ),
        "seed_scope": "single training seed; repeat-seed robustness not evaluated",
        "environment_type": "one-step counterfactual pricing environment",
        "limitations": [
            "The 0% discount action is outside observed 5%-40% training support.",
            "Rewards depend on a frozen sales predictor and proxy economics.",
            "This is not a sequential market simulator.",
            "Training uses random product sampling and one-step episodes.",
        ],
    }
    METADATA_PATH.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("DDQN checkpoint saved:", MODEL_PATH)
    print("Metadata saved:", METADATA_PATH)
    print("Mean training loss:", metadata["mean_training_loss"])
    env.close()


if __name__ == "__main__":
    main()
