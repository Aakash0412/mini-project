import csv
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


HISTORY_PATH = (
    PROJECT_ROOT
    / "models"
    / "ppo"
    / "training_history.csv"
)


def load_history():
    with open(
        HISTORY_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        return list(csv.DictReader(file))


def main():
    print("=" * 65)
    print("PHASE 11E - PPO TRAINING DIAGNOSTICS")
    print("=" * 65)

    rows = load_history()

    if not rows:
        raise ValueError("Training history is empty.")

    rewards = np.array(
        [float(row["total_reward"]) for row in rows],
        dtype=np.float64,
    )

    mean_rewards = np.array(
        [float(row["mean_step_reward"]) for row in rows],
        dtype=np.float64,
    )

    actor_losses = np.array(
        [float(row["actor_loss"]) for row in rows],
        dtype=np.float64,
    )

    critic_losses = np.array(
        [float(row["critic_loss"]) for row in rows],
        dtype=np.float64,
    )

    entropy = np.array(
        [float(row["entropy"]) for row in rows],
        dtype=np.float64,
    )

    state_indices = np.array(
        [int(row["state_index"]) for row in rows],
        dtype=np.int64,
    )

    print("\n[1] Training history")
    print("Episodes:", len(rows))

    print("\n[2] Reward statistics")
    print("Mean total reward :", rewards.mean())
    print("Std total reward  :", rewards.std())
    print("Minimum reward    :", rewards.min())
    print("Maximum reward    :", rewards.max())
    print("Median reward     :", np.median(rewards))

    print("\n[3] Reward by episode")

    for i, reward in enumerate(rewards, start=1):
        print(
            f"Episode {i:3d}: "
            f"reward={reward:10.4f} "
            f"mean_step={mean_rewards[i - 1]:10.4f}"
        )

    print("\n[4] Early vs late reward")

    midpoint = len(rewards) // 2

    early = rewards[:midpoint]
    late = rewards[midpoint:]

    print("Early-half mean:", early.mean())
    print("Late-half mean :", late.mean())

    print("\n[5] PPO loss statistics")

    print("Actor loss:")
    print("  Mean:", actor_losses.mean())
    print("  Min :", actor_losses.min())
    print("  Max :", actor_losses.max())

    print("Critic loss:")
    print("  Mean:", critic_losses.mean())
    print("  Min :", critic_losses.min())
    print("  Max :", critic_losses.max())

    print("\n[6] Entropy")

    print("Initial entropy:", entropy[0])
    print("Final entropy  :", entropy[-1])
    print("Maximum entropy for 9 actions:", np.log(9))

    print("\n[7] State sampling")

    print("Unique states sampled:", len(np.unique(state_indices)))
    print("Minimum state index :", state_indices.min())
    print("Maximum state index :", state_indices.max())

    print("\n[8] Finite-value validation")

    arrays = {
        "rewards": rewards,
        "mean_rewards": mean_rewards,
        "actor_losses": actor_losses,
        "critic_losses": critic_losses,
        "entropy": entropy,
    }

    for name, values in arrays.items():
        print(
            f"{name:16s}: "
            f"{'PASS' if np.isfinite(values).all() else 'FAIL'}"
        )

    print("\n" + "=" * 65)
    print("PPO DIAGNOSTICS COMPLETE")
    print("=" * 65)


if __name__ == "__main__":
    main()