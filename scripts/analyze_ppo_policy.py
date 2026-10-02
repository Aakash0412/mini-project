from __future__ import annotations

import sys
from pathlib import Path

# ---------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch

from src.environment.ppo_dataset import load_ppo_dataset
from src.models.ppo_agent import PPOAgent


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

CHECKPOINT = Path(
    "models/ppo/ppo_batched_economic_50_updates.pth"
)

OUTPUT_PATH = Path(
    "models/ppo/ppo_policy_analysis_test.csv"
)

DISCOUNT_BINS = [
    0, 5, 10, 15, 20,
    25, 30, 35, 40
]

STATE_DIM = 111
ACTION_DIM = 9


# ---------------------------------------------------------------------
# Load agent
# ---------------------------------------------------------------------

def load_agent(checkpoint_path: Path) -> PPOAgent:

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    agent = PPOAgent(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        actor_lr=8e-4,
        critic_lr=8e-4,
    )

    agent.actor.load_state_dict(
        checkpoint["actor_state_dict"]
    )

    agent.critic.load_state_dict(
        checkpoint["critic_state_dict"]
    )

    agent.actor.eval()
    agent.critic.eval()

    return agent


# ---------------------------------------------------------------------
# Get actor probabilities
# ---------------------------------------------------------------------

def get_action_probabilities(
    agent: PPOAgent,
    state: np.ndarray,
) -> np.ndarray:

    state_tensor = torch.as_tensor(
        state,
        dtype=torch.float32,
    ).unsqueeze(0)

    with torch.no_grad():

        output = agent.actor(
            state_tensor
        )

        if hasattr(output, "logits"):
            logits = output.logits
        else:
            logits = output

        probabilities = torch.softmax(
            logits,
            dim=-1,
        )

    return probabilities.squeeze(0).cpu().numpy()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    print("=" * 70)
    print("PPO POLICY STATE-SENSITIVITY ANALYSIS")
    print("=" * 70)

    print()
    print(
        f"Checkpoint: {CHECKPOINT}"
    )

    # ---------------------------------------------------------------
    # Load dataset
    # ---------------------------------------------------------------

    dataset = load_ppo_dataset()

    print(
        f"Loaded products: "
        f"{len(dataset.states)}"
    )

    # ---------------------------------------------------------------
    # Load original dataframe
    # ---------------------------------------------------------------

    df = pd.read_parquet(
        "data/processed/products_clean.parquet"
    )

    source_indices = dataset.source_row_indices

    aligned = df.iloc[source_indices].copy()

    # ---------------------------------------------------------------
    # Test split
    # ---------------------------------------------------------------

    test_mask = aligned["MonthNum"].between(
        9,
        10,
    )

    test_positions = np.flatnonzero(
        test_mask.to_numpy()
    )

    print(
        f"Test products: "
        f"{len(test_positions)}"
    )

    # ---------------------------------------------------------------
    # Load PPO
    # ---------------------------------------------------------------

    agent = load_agent(
        CHECKPOINT
    )

    # ---------------------------------------------------------------
    # Collect probabilities
    # ---------------------------------------------------------------

    probability_rows = []

    for counter, position in enumerate(
        test_positions
    ):

        state = dataset.states[
            position
        ]

        probabilities = (
            get_action_probabilities(
                agent,
                state,
            )
        )

        top_action = int(
            np.argmax(probabilities)
        )

        row = {
            "source_row": int(
                source_indices[position]
            ),
            "top_action": top_action,
            "top_discount": float(
                DISCOUNT_BINS[top_action]
            ),
        }

        for action, discount in enumerate(
            DISCOUNT_BINS
        ):
            row[
                f"prob_{discount}pct"
            ] = float(
                probabilities[action]
            )

        # -----------------------------------------------------------
        # Entropy
        # -----------------------------------------------------------

        entropy = -np.sum(
            probabilities
            * np.log(
                np.clip(
                    probabilities,
                    1e-12,
                    1.0,
                )
            )
        )

        row["policy_entropy"] = float(
            entropy
        )

        row["max_probability"] = float(
            probabilities.max()
        )

        probability_rows.append(row)

        if (counter + 1) % 1000 == 0:
            print(
                f"Processed "
                f"{counter + 1}/"
                f"{len(test_positions)}"
            )

    results_df = pd.DataFrame(
        probability_rows
    )

    # ---------------------------------------------------------------
    # Probability summary
    # ---------------------------------------------------------------

    probability_columns = [
        f"prob_{discount}pct"
        for discount in DISCOUNT_BINS
    ]

    print()
    print("=" * 70)
    print("AVERAGE ACTION PROBABILITIES")
    print("=" * 70)

    for discount in DISCOUNT_BINS:

        column = (
            f"prob_{discount}pct"
        )

        mean_probability = (
            results_df[column].mean()
        )

        median_probability = (
            results_df[column].median()
        )

        std_probability = (
            results_df[column].std()
        )

        print(
            f"{discount:>2}% : "
            f"mean={mean_probability:.6f}  "
            f"median={median_probability:.6f}  "
            f"std={std_probability:.6f}"
        )

    # ---------------------------------------------------------------
    # Selected action distribution
    # ---------------------------------------------------------------

    print()
    print("=" * 70)
    print("DETERMINISTIC POLICY DISTRIBUTION")
    print("=" * 70)

    top_distribution = (
        results_df["top_discount"]
        .value_counts()
        .sort_index()
    )

    for discount in DISCOUNT_BINS:

        count = int(
            top_distribution.get(
                discount,
                0,
            )
        )

        percentage = (
            count
            / len(results_df)
            * 100
        )

        print(
            f"{discount:>2}% : "
            f"{count:>5} "
            f"({percentage:>6.2f}%)"
        )

    # ---------------------------------------------------------------
    # Entropy statistics
    # ---------------------------------------------------------------

    print()
    print("=" * 70)
    print("POLICY ENTROPY")
    print("=" * 70)

    print(
        f"Mean:       "
        f"{results_df['policy_entropy'].mean():.6f}"
    )

    print(
        f"Median:     "
        f"{results_df['policy_entropy'].median():.6f}"
    )

    print(
        f"Std:        "
        f"{results_df['policy_entropy'].std():.6f}"
    )

    print(
        f"Min:        "
        f"{results_df['policy_entropy'].min():.6f}"
    )

    print(
        f"Max:        "
        f"{results_df['policy_entropy'].max():.6f}"
    )

    # ---------------------------------------------------------------
    # Maximum probability statistics
    # ---------------------------------------------------------------

    print()
    print("=" * 70)
    print("MAXIMUM ACTION PROBABILITY")
    print("=" * 70)

    print(
        f"Mean:       "
        f"{results_df['max_probability'].mean():.6f}"
    )

    print(
        f"Median:     "
        f"{results_df['max_probability'].median():.6f}"
    )

    print(
        f"Std:        "
        f"{results_df['max_probability'].std():.6f}"
    )

    print(
        f"Min:        "
        f"{results_df['max_probability'].min():.6f}"
    )

    print(
        f"Max:        "
        f"{results_df['max_probability'].max():.6f}"
    )

    # ---------------------------------------------------------------
    # State sensitivity
    # ---------------------------------------------------------------

    probability_matrix = (
        results_df[
            probability_columns
        ].to_numpy()
    )

    # Standard deviation of each action's probability across products.
    # Larger values indicate stronger variation in the actor's
    # response to different product states.

    print()
    print("=" * 70)
    print("STATE SENSITIVITY")
    print("=" * 70)

    state_sensitivity_values = []

    for action, discount in enumerate(
        DISCOUNT_BINS
    ):

        values = probability_matrix[
            :, action
        ]

        std = float(
            np.std(values)
        )

        state_sensitivity_values.append(
            std
        )

        print(
            f"{discount:>2}% probability "
            f"std across products: "
            f"{std:.6f}"
        )

    overall_state_sensitivity = float(
        np.mean(
            state_sensitivity_values
        )
    )

    print()
    print(
        "Mean action-probability "
        "variation across products: "
        f"{overall_state_sensitivity:.6f}"
    )

    # ---------------------------------------------------------------
    # Pairwise policy variation
    # ---------------------------------------------------------------

    # Compare each product's probability vector to the average
    # probability vector.

    mean_policy = (
        probability_matrix.mean(
            axis=0
        )
    )

    distances = np.linalg.norm(
        probability_matrix
        - mean_policy,
        axis=1,
    )

    print()
    print("=" * 70)
    print("POLICY VECTOR VARIATION")
    print("=" * 70)

    print(
        f"Mean distance from "
        f"average policy: "
        f"{distances.mean():.6f}"
    )

    print(
        f"Median distance: "
        f"{np.median(distances):.6f}"
    )

    print(
        f"Std distance: "
        f"{distances.std():.6f}"
    )

    print(
        f"Maximum distance: "
        f"{distances.max():.6f}"
    )

    # ---------------------------------------------------------------
    # Save
    # ---------------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print()
    print("=" * 70)
    print("COMPLETED")
    print("=" * 70)

    print(
        f"Saved analysis: "
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()