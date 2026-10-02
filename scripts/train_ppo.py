from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

# ---------------------------------------------------------------------
# Project-root import fix
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch

from src.environment.ppo_dataset import load_ppo_dataset
from src.environment.pricing_environment import DynamicPricingEnvironment
from src.models.ppo_agent import PPOAgent
from src.models.ppo_trainer import (
    PPOConfig,
    PPOTrainer,
    PPORolloutBuffer,
)


# ---------------------------------------------------------------------
# 8-action observed-support pricing space
#
# The original dataset contains observed discounts from 5% to 40%.
# Therefore this experiment excludes 0%, which is outside observed
# training support.
# ---------------------------------------------------------------------
DISCOUNT_ACTIONS = np.array(
    [5, 10, 15, 20, 25, 30, 35, 40],
    dtype=np.int64,
)

ACTION_DIM = len(DISCOUNT_ACTIONS)


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Batched PPO training for dynamic pricing "
                    "using the 8-action observed-support space."
    )

    parser.add_argument(
        "--updates",
        type=int,
        default=500,
        help="Number of PPO updates.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Number of real products per PPO rollout.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/ppo",
        help="Directory for PPO artifacts.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    args = parse_args()

    set_seed(args.seed)

    print("=" * 70)
    print("BATCHED ECONOMIC PPO TRAINING")
    print("8-ACTION OBSERVED-SUPPORT EXPERIMENT")
    print("=" * 70)

    print()
    print("Action space:")
    print(
        "  "
        + ", ".join(f"{discount}%" for discount in DISCOUNT_ACTIONS)
    )
    print(f"Action dimension: {ACTION_DIM}")
    print()

    # -----------------------------------------------------------------
    # 1. Load states + aligned economics
    # -----------------------------------------------------------------
    dataset = load_ppo_dataset()

    # -----------------------------------------------------------------
    # 2. Recover chronological training split
    # -----------------------------------------------------------------
    products_df = pd.read_parquet(
        "data/processed/products_clean.parquet"
    )

    aligned = products_df.iloc[
        dataset.source_row_indices
    ].copy()

    train_mask = aligned["MonthNum"].between(1, 6)

    train_positions = np.flatnonzero(
        train_mask.to_numpy()
    )

    if len(train_positions) == 0:
        raise RuntimeError(
            "No training products found in Jan-Jun split."
        )

    print(
        f"Total usable PPO products: "
        f"{len(dataset.states)}"
    )

    print(
        f"Training products: "
        f"{len(train_positions)}"
    )

    print(
        f"Products per PPO update: "
        f"{args.batch_size}"
    )

    print(
        f"Number of PPO updates: "
        f"{args.updates}"
    )

    print(
        f"Total pricing decisions: "
        f"{args.updates * args.batch_size}"
    )

    print()

    # -----------------------------------------------------------------
    # 3. PPO agent
    # -----------------------------------------------------------------
    agent = PPOAgent(
        state_dim=111,
        action_dim=ACTION_DIM,
        actor_lr=8e-4,
        critic_lr=8e-4,
    )

    config = PPOConfig(
        gamma=0.9,
        gae_lambda=0.8,
        clip_epsilon=0.2,
        entropy_coefficient=0.01,
        value_coefficient=0.5,
        batch_size=args.batch_size,
        update_epochs=4,
        max_grad_norm=0.5,
    )

    trainer = PPOTrainer(
        agent=agent,
        config=config,
    )

    # -----------------------------------------------------------------
    # 4. Economic environment
    #
    # IMPORTANT:
    # The existing DynamicPricingEnvironment still uses the project's
    # original 9-action mapping:
    #
    #   0 -> 0%
    #   1 -> 5%
    #   ...
    #   8 -> 40%
    #
    # Our PPO has 8 actions:
    #
    #   0 -> 5%
    #   1 -> 10%
    #   ...
    #   7 -> 40%
    #
    # Therefore we translate the PPO action into the corresponding
    # original environment action index.
    #
    # PPO action:
    #   0,1,2,3,4,5,6,7
    #
    # Environment action:
    #   1,2,3,4,5,6,7,8
    # -----------------------------------------------------------------
    environment = DynamicPricingEnvironment()

    # PPO action -> original 9-action environment action
    environment_action_map = np.arange(
        1,
        9,
        dtype=np.int64,
    )

    # -----------------------------------------------------------------
    # 5. Output
    # -----------------------------------------------------------------
    output_dir = Path(args.output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    history_path = (
        output_dir
        / "batched_supported_8action_training_history.csv"
    )

    checkpoint_path = (
        output_dir
        / "ppo_batched_supported_8action_500_updates.pth"
    )

    history = []

    rng = np.random.default_rng(args.seed)

    # -----------------------------------------------------------------
    # 6. Batched PPO training
    # -----------------------------------------------------------------
    for update_number in range(
        1,
        args.updates + 1,
    ):

        # -------------------------------------------------------------
        # Sample real training products.
        #
        # Sampling with replacement across updates is intentional.
        # Each update receives independent real products.
        # -------------------------------------------------------------
        positions = rng.choice(
            train_positions,
            size=args.batch_size,
            replace=False
            if args.batch_size <= len(train_positions)
            else True,
        )

        buffer = PPORolloutBuffer()

        update_rewards = []
        update_actions = []

        update_growth = []
        update_sales = []
        update_revenue = []
        update_margin = []

        # -------------------------------------------------------------
        # Collect one rollout
        # -------------------------------------------------------------
        for position in positions:

            position = int(position)

            state = dataset.states[position]
            product = dataset.products[position]

            environment.reset(
                state=state,
                product=product,
            )

            state_tensor = torch.as_tensor(
                state,
                dtype=torch.float32,
            )

            # ---------------------------------------------------------
            # Stochastic PPO action during training
            #
            # PPO action range:
            #   0..7
            #
            # corresponding discounts:
            #   5,10,15,20,25,30,35,40
            # ---------------------------------------------------------
            ppo_action, log_probability, value = (
                agent.select_action(state_tensor)
            )

            ppo_action = int(ppo_action)

            if not 0 <= ppo_action < ACTION_DIM:
                raise RuntimeError(
                    f"Invalid PPO action: {ppo_action}. "
                    f"Expected 0..{ACTION_DIM - 1}."
                )

            # ---------------------------------------------------------
            # Translate 8-action PPO index into original environment
            # action index.
            #
            # Example:
            # PPO action 0 -> environment action 1 -> 5%
            # PPO action 7 -> environment action 8 -> 40%
            # ---------------------------------------------------------
            environment_action = int(
                environment_action_map[ppo_action]
            )

            # ---------------------------------------------------------
            # Economic environment
            # ---------------------------------------------------------
            _, reward, terminated, info = (
                environment.step(environment_action)
            )

            # ---------------------------------------------------------
            # IMPORTANT:
            # Store the PPO action (0..7), NOT the environment action
            # (1..8), because the PPO actor has 8 outputs.
            # ---------------------------------------------------------
            buffer.add(
                state=state_tensor,
                action=ppo_action,
                reward=reward,
                log_probability=log_probability,
                value=value,
                done=terminated,
            )

            update_rewards.append(
                float(reward)
            )

            update_actions.append(
                ppo_action
            )

            update_growth.append(
                float(info["predicted_growth_rate"])
            )

            update_sales.append(
                float(info["estimated_sales"])
            )

            update_revenue.append(
                float(info["estimated_revenue"])
            )

            update_margin.append(
                float(info["estimated_margin"])
            )

        # -------------------------------------------------------------
        # Every product is a terminal one-step episode.
        # Therefore no bootstrap value is required.
        # -------------------------------------------------------------
        next_value = 0.0

        # -------------------------------------------------------------
        # PPO update
        # -------------------------------------------------------------
        metrics = trainer.update(
            buffer,
            next_value=next_value,
        )

        # -------------------------------------------------------------
        # Batch statistics
        # -------------------------------------------------------------
        mean_reward = float(
            np.mean(update_rewards)
        )

        mean_growth = float(
            np.mean(update_growth)
        )

        mean_sales = float(
            np.mean(update_sales)
        )

        mean_revenue = float(
            np.mean(update_revenue)
        )

        mean_margin = float(
            np.mean(update_margin)
        )

        action_counts = np.bincount(
            update_actions,
            minlength=ACTION_DIM,
        )

        # -------------------------------------------------------------
        # Save update-level history
        # -------------------------------------------------------------
        row = {
            "update": update_number,
            "samples": args.batch_size,
            "mean_reward": mean_reward,
            "mean_predicted_growth": mean_growth,
            "mean_estimated_sales": mean_sales,
            "mean_estimated_revenue": mean_revenue,
            "mean_estimated_margin": mean_margin,
            "actor_loss": float(
                metrics.get("actor_loss", 0.0)
            ),
            "critic_loss": float(
                metrics.get("critic_loss", 0.0)
            ),
            "entropy": float(
                metrics.get("entropy", 0.0)
            ),
        }

        # Store action counts using actual discount names.
        for action_index, discount in enumerate(
            DISCOUNT_ACTIONS
        ):
            row[
                f"action_{int(discount)}"
            ] = int(
                action_counts[action_index]
            )

        history.append(row)

        # -------------------------------------------------------------
        # Console output
        # -------------------------------------------------------------
        action_text = ", ".join(
            f"{int(DISCOUNT_ACTIONS[i])}%={action_counts[i]}"
            for i in range(ACTION_DIM)
            if action_counts[i] > 0
        )

        print(
            f"Update {update_number:>3}/{args.updates} | "
            f"Reward {mean_reward:>10.4f} | "
            f"Growth {mean_growth:>9.4f} | "
            f"Entropy {metrics.get('entropy', 0.0):>8.4f} | "
            f"Actions [{action_text}]"
        )

    # -----------------------------------------------------------------
    # 7. Save history
    # -----------------------------------------------------------------
    if history:
        fieldnames = list(
            history[0].keys()
        )

        with open(
            history_path,
            "w",
            newline="",
            encoding="utf-8",
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames,
            )

            writer.writeheader()
            writer.writerows(history)

    # -----------------------------------------------------------------
    # 8. Save checkpoint
    # -----------------------------------------------------------------
    torch.save(
        {
            "actor_state_dict": (
                agent.actor.state_dict()
            ),
            "critic_state_dict": (
                agent.critic.state_dict()
            ),
            "actor_optimizer_state_dict": (
                agent.actor_optimizer.state_dict()
            ),
            "critic_optimizer_state_dict": (
                agent.critic_optimizer.state_dict()
            ),
            "config": vars(config),
            "updates": args.updates,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "training_samples": len(train_positions),
            "total_decisions": (
                args.updates * args.batch_size
            ),
            "state_dim": 111,
            "action_dim": ACTION_DIM,
            "discount_actions": (
                DISCOUNT_ACTIONS.tolist()
            ),
            "reward": (
                "paper-style delta margin + delta quantity"
            ),
            "economic_environment": True,
            "chronological_training_split": (
                "Jan-Jun 2023"
            ),
            "experiment": (
                "8-action observed-support PPO"
            ),
            "excluded_action": (
                "0% - outside observed training support"
            ),
        },
        checkpoint_path,
    )

    # -----------------------------------------------------------------
    # 9. Final summary
    # -----------------------------------------------------------------
    print()
    print("=" * 70)
    print(
        "8-ACTION OBSERVED-SUPPORT PPO TRAINING COMPLETE"
    )
    print("=" * 70)

    print(
        f"Training products: "
        f"{len(train_positions)}"
    )

    print(
        f"PPO updates: "
        f"{args.updates}"
    )

    print(
        f"Decisions processed: "
        f"{args.updates * args.batch_size}"
    )

    print(
        "Action space: "
        + ", ".join(
            f"{int(x)}%"
            for x in DISCOUNT_ACTIONS
        )
    )

    print(
        f"History: "
        f"{history_path}"
    )

    print(
        f"Checkpoint: "
        f"{checkpoint_path}"
    )


if __name__ == "__main__":
    main()