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
# Original observed discounts:
#     5, 10, 15, 20, 25, 30, 35, 40%
#
# 0% is excluded because it is outside the observed training support.
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
        description=(
            "Transfer-warm-start PPO training for dynamic pricing "
            "using the 8-action observed-support space."
        )
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
        default="models/ppo_transfer",
        help="Directory for transfer-PPO artifacts.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------
# Transfer warm-start loading
# ---------------------------------------------------------------------
def load_transfer_warmstart(
    agent: PPOAgent,
    warmstart_path: Path,
) -> None:
    """
    Load the actor from the transfer warm-start checkpoint.

    Warm-start checkpoint:
        9 actions

        0 -> 0%
        1 -> 5%
        2 -> 10%
        ...
        8 -> 40%

    Current PPO experiment:
        8 actions

        0 -> 5%
        1 -> 10%
        ...
        7 -> 40%

    Therefore the warm-start actor output rows 1..8 are copied
    into the 8-action PPO actor.

    The critic is intentionally NOT loaded.
    """

    if not warmstart_path.exists():
        raise FileNotFoundError(
            "Transfer warm-start checkpoint not found:\n"
            f"    {warmstart_path}"
        )

    checkpoint = torch.load(
        warmstart_path,
        map_location="cpu",
        weights_only=False,
    )

    if "actor_state_dict" not in checkpoint:
        raise KeyError(
            "Transfer checkpoint does not contain "
            "'actor_state_dict'."
        )

    warmstart_actor = checkpoint["actor_state_dict"]

    required_keys = {
        "network.0.weight",
        "network.0.bias",
        "network.2.weight",
        "network.2.bias",
        "network.4.weight",
        "network.4.bias",
    }

    missing_keys = required_keys - set(
        warmstart_actor.keys()
    )

    if missing_keys:
        raise KeyError(
            "Warm-start actor is missing keys: "
            f"{sorted(missing_keys)}"
        )

    current_actor = agent.actor.state_dict()

    # -------------------------------------------------------------
    # Check first hidden layer.
    # -------------------------------------------------------------
    if (
        current_actor["network.0.weight"].shape
        != warmstart_actor["network.0.weight"].shape
    ):
        raise RuntimeError(
            "First actor layer shape mismatch:\n"
            f"    PPO:       "
            f"{current_actor['network.0.weight'].shape}\n"
            f"    Warmstart: "
            f"{warmstart_actor['network.0.weight'].shape}"
        )

    # -------------------------------------------------------------
    # Check second hidden layer.
    # -------------------------------------------------------------
    if (
        current_actor["network.2.weight"].shape
        != warmstart_actor["network.2.weight"].shape
    ):
        raise RuntimeError(
            "Second actor layer shape mismatch:\n"
            f"    PPO:       "
            f"{current_actor['network.2.weight'].shape}\n"
            f"    Warmstart: "
            f"{warmstart_actor['network.2.weight'].shape}"
        )

    # -------------------------------------------------------------
    # Check warm-start output dimension.
    # -------------------------------------------------------------
    warmstart_output_dim = (
        warmstart_actor["network.4.weight"].shape[0]
    )

    if warmstart_output_dim != 9:
        raise RuntimeError(
            "Expected warm-start actor to have 9 outputs, "
            f"but found {warmstart_output_dim}."
        )

    # -------------------------------------------------------------
    # Check target PPO output dimension.
    # -------------------------------------------------------------
    current_output_dim = (
        current_actor["network.4.weight"].shape[0]
    )

    if current_output_dim != ACTION_DIM:
        raise RuntimeError(
            "Expected PPO actor to have "
            f"{ACTION_DIM} outputs, "
            f"but found {current_output_dim}."
        )

    # -------------------------------------------------------------
    # Copy transfer actor.
    # -------------------------------------------------------------
    with torch.no_grad():

        # Shared first hidden layer.
        current_actor["network.0.weight"].copy_(
            warmstart_actor["network.0.weight"]
        )

        current_actor["network.0.bias"].copy_(
            warmstart_actor["network.0.bias"]
        )

        # Shared second hidden layer.
        current_actor["network.2.weight"].copy_(
            warmstart_actor["network.2.weight"]
        )

        current_actor["network.2.bias"].copy_(
            warmstart_actor["network.2.bias"]
        )

        # ---------------------------------------------------------
        # Warm-start output:
        #
        # row 0 -> 0%
        # row 1 -> 5%
        # row 2 -> 10%
        # ...
        # row 8 -> 40%
        #
        # Current 8-action PPO:
        #
        # row 0 -> 5%
        # row 1 -> 10%
        # ...
        # row 7 -> 40%
        #
        # Therefore copy rows 1..8.
        # ---------------------------------------------------------
        current_actor["network.4.weight"].copy_(
            warmstart_actor["network.4.weight"][1:9]
        )

        current_actor["network.4.bias"].copy_(
            warmstart_actor["network.4.bias"][1:9]
        )

    agent.actor.load_state_dict(
        current_actor,
        strict=True,
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():

    args = parse_args()

    set_seed(args.seed)

    print("=" * 70)
    print("TRANSFER-WARM-START ECONOMIC PPO TRAINING")
    print("8-ACTION OBSERVED-SUPPORT EXPERIMENT")
    print("=" * 70)

    # -----------------------------------------------------------------
    # Configuration
    # -----------------------------------------------------------------
    print()
    print("Transfer configuration:")
    print(
        "  Source: NeoRL source policy"
    )
    print(
        "  Adapter: 111-D Amazon state -> 4-D pseudo-state"
    )
    print(
        "  Initialization: transfer-warm-start actor"
    )
    print(
        "  Critic: fresh initialization"
    )
    print(
        "  Action space: "
        + ", ".join(
            f"{int(x)}%"
            for x in DISCOUNT_ACTIONS
        )
    )
    print(
        f"  Action dimension: {ACTION_DIM}"
    )
    print(
        f"  PPO updates: {args.updates}"
    )
    print(
        f"  Batch size: {args.batch_size}"
    )
    print(
        f"  Seed: {args.seed}"
    )

    # -----------------------------------------------------------------
    # 1. Load states + aligned economics
    # -----------------------------------------------------------------
    print()
    print("[1] Loading Amazon target dataset...")

    dataset = load_ppo_dataset()

    print(
        f"    Total usable PPO products: "
        f"{len(dataset.states)}"
    )

    # -----------------------------------------------------------------
    # 2. Recover chronological training split
    # -----------------------------------------------------------------
    print()
    print("[2] Recovering chronological training split...")

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
        f"    Training products: "
        f"{len(train_positions)}"
    )

    print(
        f"    Products per PPO update: "
        f"{args.batch_size}"
    )

    print(
        f"    Number of PPO updates: "
        f"{args.updates}"
    )

    print(
        f"    Total pricing decisions: "
        f"{args.updates * args.batch_size}"
    )

    # -----------------------------------------------------------------
    # 3. PPO agent
    # -----------------------------------------------------------------
    print()
    print("[3] Creating PPO agent...")

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

    print(
        "    state dimension: 111"
    )

    print(
        f"    action dimension: {ACTION_DIM}"
    )

    print(
        "    actor learning rate: 8e-4"
    )

    print(
        "    critic learning rate: 8e-4"
    )

    # -----------------------------------------------------------------
    # 3A. Load transfer warm-start
    # -----------------------------------------------------------------
    print()
    print("[3A] Loading transfer warm-start actor...")

    warmstart_path = Path(
        "models/transfer/ppo_warmstart/"
        "ppo_warmstart_initialization.pth"
    )

    load_transfer_warmstart(
        agent=agent,
        warmstart_path=warmstart_path,
    )

    print(
        "    warm-start actor: loaded"
    )

    print(
        "    critic: freshly initialized"
    )

    # -----------------------------------------------------------------
    # 3B. Verify warm-start actor
    # -----------------------------------------------------------------
    print()
    print("[3B] Verifying warm-start actor...")

    verification_count = min(
        5000,
        len(train_positions),
    )

    verification_positions = train_positions[
        :verification_count
    ]

    verification_states = dataset.states[
        verification_positions
    ]

    with torch.no_grad():

        verification_tensor = torch.as_tensor(
            verification_states,
            dtype=torch.float32,
        )

        logits = agent.actor(
            verification_tensor
        )

        probabilities = torch.softmax(
            logits,
            dim=-1,
        )

        deterministic_actions = torch.argmax(
            probabilities,
            dim=-1,
        ).cpu().numpy()

        entropy = -(
            probabilities
            * torch.log(
                probabilities.clamp_min(1e-8)
            )
        ).sum(dim=-1)

    logits_finite = bool(
        torch.isfinite(logits).all()
    )

    probabilities_finite = bool(
        torch.isfinite(probabilities).all()
    )

    probability_sums_valid = torch.allclose(
        probabilities.sum(dim=-1),
        torch.ones(
            probabilities.shape[0],
            dtype=probabilities.dtype,
            device=probabilities.device,
        ),
        atol=1e-5,
    )

    print(
        f"    finite logits: "
        f"{logits_finite}"
    )

    print(
        f"    finite probabilities: "
        f"{probabilities_finite}"
    )

    print(
        f"    probability sums valid: "
        f"{bool(probability_sums_valid)}"
    )

    print(
        f"    mean entropy: "
        f"{entropy.mean().item():.6f}"
    )

    print()
    print(
        "    Deterministic warm-start distribution:"
    )

    for action_index, discount in enumerate(
        DISCOUNT_ACTIONS
    ):

        count = int(
            np.sum(
                deterministic_actions
                == action_index
            )
        )

        percentage = (
            count
            / len(deterministic_actions)
            * 100.0
        )

        print(
            f"       {int(discount):2d}% : "
            f"{count:5d} "
            f"({percentage:6.2f}%)"
        )

    # -----------------------------------------------------------------
    # 4. Economic environment
    # -----------------------------------------------------------------
    print()
    print("[4] Creating economic environment...")

    environment = DynamicPricingEnvironment()

    # -----------------------------------------------------------------
    # Existing environment mapping:
    #
    # Environment:
    #   0 -> 0%
    #   1 -> 5%
    #   2 -> 10%
    #   ...
    #   8 -> 40%
    #
    # PPO:
    #   0 -> 5%
    #   1 -> 10%
    #   ...
    #   7 -> 40%
    #
    # Therefore:
    #   PPO action + 1 = environment action
    # -----------------------------------------------------------------
    environment_action_map = np.arange(
        1,
        9,
        dtype=np.int64,
    )

    # -----------------------------------------------------------------
    # 5. Output
    # -----------------------------------------------------------------
    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    history_path = (
        output_dir
        / "transfer_supported_8action_training_history.csv"
    )

    checkpoint_path = (
        output_dir
        / (
            "ppo_transfer_supported_8action_"
            f"{args.updates}_updates.pth"
        )
    )

    history = []

    rng = np.random.default_rng(
        args.seed
    )

    # -----------------------------------------------------------------
    # 6. Batched PPO training
    # -----------------------------------------------------------------
    print()
    print("[5] Starting PPO fine-tuning...")

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
            replace=(
                args.batch_size
                > len(train_positions)
            ),
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

            state = dataset.states[
                position
            ]

            product = dataset.products[
                position
            ]

            environment.reset(
                state=state,
                product=product,
            )

            state_tensor = torch.as_tensor(
                state,
                dtype=torch.float32,
            )

            # ---------------------------------------------------------
            # Stochastic PPO action during training.
            #
            # PPO action:
            #   0..7
            #
            # Discount:
            #   5,10,15,20,25,30,35,40
            # ---------------------------------------------------------
            (
                ppo_action,
                log_probability,
                value,
            ) = agent.select_action(
                state_tensor
            )

            ppo_action = int(
                ppo_action
            )

            if not 0 <= ppo_action < ACTION_DIM:
                raise RuntimeError(
                    f"Invalid PPO action: "
                    f"{ppo_action}. "
                    f"Expected 0.."
                    f"{ACTION_DIM - 1}."
                )

            # ---------------------------------------------------------
            # Translate PPO action into environment action.
            # ---------------------------------------------------------
            environment_action = int(
                environment_action_map[
                    ppo_action
                ]
            )

            # ---------------------------------------------------------
            # Economic environment step.
            # ---------------------------------------------------------
            (
                _,
                reward,
                terminated,
                info,
            ) = environment.step(
                environment_action
            )

            # ---------------------------------------------------------
            # Store PPO action, NOT environment action.
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
                float(
                    info[
                        "predicted_growth_rate"
                    ]
                )
            )

            update_sales.append(
                float(
                    info[
                        "estimated_sales"
                    ]
                )
            )

            update_revenue.append(
                float(
                    info[
                        "estimated_revenue"
                    ]
                )
            )

            update_margin.append(
                float(
                    info[
                        "estimated_margin"
                    ]
                )
            )

        # -------------------------------------------------------------
        # Every product is a terminal one-step episode.
        # -------------------------------------------------------------
        next_value = 0.0

        # -------------------------------------------------------------
        # PPO update.
        # -------------------------------------------------------------
        metrics = trainer.update(
            buffer,
            next_value=next_value,
        )

        # -------------------------------------------------------------
        # Batch statistics.
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
        # Save update-level history.
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
                metrics.get(
                    "actor_loss",
                    0.0,
                )
            ),
            "critic_loss": float(
                metrics.get(
                    "critic_loss",
                    0.0,
                )
            ),
            "entropy": float(
                metrics.get(
                    "entropy",
                    0.0,
                )
            ),
        }

        for action_index, discount in enumerate(
            DISCOUNT_ACTIONS
        ):

            row[
                f"action_{int(discount)}"
            ] = int(
                action_counts[
                    action_index
                ]
            )

        history.append(row)

        # -------------------------------------------------------------
        # Console output.
        # -------------------------------------------------------------
        action_text = ", ".join(
            f"{int(DISCOUNT_ACTIONS[i])}%="
            f"{action_counts[i]}"
            for i in range(
                ACTION_DIM
            )
            if action_counts[i] > 0
        )

        print(
            f"Update {update_number:>4}/"
            f"{args.updates} | "
            f"Reward "
            f"{mean_reward:>10.4f} | "
            f"Growth "
            f"{mean_growth:>9.4f} | "
            f"Entropy "
            f"{metrics.get('entropy', 0.0):>8.4f} | "
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
            writer.writerows(
                history
            )

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
            "training_samples": len(
                train_positions
            ),
            "total_decisions": (
                args.updates
                * args.batch_size
            ),
            "state_dim": 111,
            "action_dim": ACTION_DIM,
            "discount_actions": (
                DISCOUNT_ACTIONS.tolist()
            ),
            "reward": (
                "paper-style delta margin "
                "+ delta quantity"
            ),
            "economic_environment": True,
            "chronological_training_split": (
                "Jan-Jun 2023"
            ),
            "experiment": (
                "8-action observed-support PPO "
                "with transfer warm-start"
            ),
            "transfer_warmstart": True,
            "transfer_source": (
                "NeoRL source policy"
            ),
            "transfer_adapter": True,
            "critic_initialization": (
                "fresh"
            ),
            "excluded_action": (
                "0% - outside observed "
                "training support"
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
        "TRANSFER-WARM-START 8-ACTION "
        "PPO TRAINING COMPLETE"
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