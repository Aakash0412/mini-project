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

from src.environment.ppo_dataset import load_ppo_dataset
from src.environment.pricing_environment import DynamicPricingEnvironment


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

FULL_ACTIONS = [0, 5, 10, 15, 20, 25, 30, 35, 40]

SUPPORTED_ACTIONS = [
    5, 10, 15, 20, 25, 30, 35, 40
]

PPO_ACTIONS = [
    10, 20, 30
]

OUTPUT_PATH = Path(
    "models/ppo/action_space_analysis_test.csv"
)

EPSILON = 1e-8


# ---------------------------------------------------------------------
# Relative reward
# ---------------------------------------------------------------------

def calculate_relative_reward(
    delta_margin: float,
    delta_quantity: float,
    baseline_margin: float,
    baseline_quantity: float,
    alpha: float,
    beta: float,
) -> float:

    relative_delta_quantity = (
        delta_quantity
        / max(abs(baseline_quantity), EPSILON)
    )

    relative_delta_margin = (
        delta_margin
        / max(abs(baseline_margin), EPSILON)
    )

    return float(
        alpha * relative_delta_margin
        + beta * relative_delta_quantity
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    print("=" * 70)
    print("ACTION-SPACE ORACLE ANALYSIS")
    print("=" * 70)

    # -----------------------------------------------------------------
    # Load dataset
    # -----------------------------------------------------------------

    dataset = load_ppo_dataset()

    print(
        f"Loaded economic PPO samples: "
        f"{len(dataset.states)}"
    )

    # -----------------------------------------------------------------
    # Load source dataframe
    # -----------------------------------------------------------------

    df = pd.read_parquet(
        "data/processed/products_clean.parquet"
    )

    source_indices = dataset.source_row_indices

    aligned = df.iloc[source_indices].copy()

    # -----------------------------------------------------------------
    # Test split
    # -----------------------------------------------------------------

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

    # -----------------------------------------------------------------
    # Environment
    # -----------------------------------------------------------------

    environment = DynamicPricingEnvironment()

    alpha = environment.reward_config.alpha
    beta = environment.reward_config.beta

    print()
    print("Reward configuration:")
    print(f"  Alpha = {alpha}")
    print(f"  Beta  = {beta}")
    print("  Reward = normalized relative reward")

    # -----------------------------------------------------------------
    # Storage
    # -----------------------------------------------------------------

    rows = []

    # -----------------------------------------------------------------
    # Evaluate each product
    # -----------------------------------------------------------------

    for counter, position in enumerate(
        test_positions
    ):

        state = dataset.states[position]
        product = dataset.products[position]

        environment.reset(
            state=state,
            product=product,
        )

        baseline_sales = (
            product.last_monthly_sales
        )

        baseline_margin = (
            environment.observed_outcome.estimated_margin
        )

        if baseline_margin is None:
            continue

        # -------------------------------------------------------------
        # Evaluate all actions
        # -------------------------------------------------------------

        all_outcomes = (
            environment.predict_all_actions()
        )

        rewards = {}

        for action, outcome in all_outcomes.items():

            discount = float(
                outcome.discount_percent
            )

            if outcome.estimated_margin is None:
                continue

            delta_quantity = (
                outcome.estimated_sales
                - baseline_sales
            )

            delta_margin = (
                outcome.estimated_margin
                - baseline_margin
            )

            reward = calculate_relative_reward(
                delta_margin=delta_margin,
                delta_quantity=delta_quantity,
                baseline_margin=baseline_margin,
                baseline_quantity=baseline_sales,
                alpha=alpha,
                beta=beta,
            )

            rewards[discount] = reward

        # -------------------------------------------------------------
        # Ensure all required actions exist
        # -------------------------------------------------------------

        if not all(
            discount in rewards
            for discount in FULL_ACTIONS
        ):
            continue

        # -------------------------------------------------------------
        # Three oracle definitions
        # -------------------------------------------------------------

        full_oracle_discount = max(
            FULL_ACTIONS,
            key=lambda d: rewards[d],
        )

        supported_oracle_discount = max(
            SUPPORTED_ACTIONS,
            key=lambda d: rewards[d],
        )

        ppo_space_oracle_discount = max(
            PPO_ACTIONS,
            key=lambda d: rewards[d],
        )

        # -------------------------------------------------------------
        # Rewards
        # -------------------------------------------------------------

        full_oracle_reward = (
            rewards[full_oracle_discount]
        )

        supported_oracle_reward = (
            rewards[supported_oracle_discount]
        )

        ppo_space_oracle_reward = (
            rewards[ppo_space_oracle_discount]
        )

        # -------------------------------------------------------------
        # Regret caused by action-space restrictions
        # -------------------------------------------------------------

        support_restriction_gap = (
            full_oracle_reward
            - supported_oracle_reward
        )

        ppo_space_gap = (
            supported_oracle_reward
            - ppo_space_oracle_reward
        )

        # -------------------------------------------------------------
        # Save
        # -------------------------------------------------------------

        row = {
            "source_row": int(
                source_indices[position]
            ),

            "full_oracle_discount": float(
                full_oracle_discount
            ),

            "full_oracle_reward": float(
                full_oracle_reward
            ),

            "supported_oracle_discount": float(
                supported_oracle_discount
            ),

            "supported_oracle_reward": float(
                supported_oracle_reward
            ),

            "ppo_space_oracle_discount": float(
                ppo_space_oracle_discount
            ),

            "ppo_space_oracle_reward": float(
                ppo_space_oracle_reward
            ),

            "support_restriction_gap": float(
                support_restriction_gap
            ),

            "ppo_space_gap": float(
                ppo_space_gap
            ),
        }

        # -------------------------------------------------------------
        # Store reward for every action
        # -------------------------------------------------------------

        for discount in FULL_ACTIONS:

            row[
                f"reward_{discount}pct"
            ] = float(
                rewards[discount]
            )

        rows.append(row)

        if (counter + 1) % 500 == 0:

            print(
                f"Processed "
                f"{counter + 1}/"
                f"{len(test_positions)}"
            )

    # -----------------------------------------------------------------
    # Dataframe
    # -----------------------------------------------------------------

    results_df = pd.DataFrame(
        rows
    )

    # -----------------------------------------------------------------
    # Summary helper
    # -----------------------------------------------------------------

    def print_distribution(
        column: str,
        title: str,
    ):

        print()
        print("-" * 70)
        print(title)
        print("-" * 70)

        distribution = (
            results_df[column]
            .value_counts()
            .sort_index()
        )

        for discount in FULL_ACTIONS:

            count = int(
                distribution.get(
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

    # -----------------------------------------------------------------
    # Full oracle
    # -----------------------------------------------------------------

    print()
    print("=" * 70)
    print("FULL ACTION-SPACE ORACLE")
    print("=" * 70)

    print(
        f"Mean reward: "
        f"{results_df['full_oracle_reward'].mean():.6f}"
    )

    print(
        f"Median reward: "
        f"{results_df['full_oracle_reward'].median():.6f}"
    )

    print_distribution(
        "full_oracle_discount",
        "Full Oracle Discount Distribution",
    )

    # -----------------------------------------------------------------
    # Observed-support oracle
    # -----------------------------------------------------------------

    print()
    print("=" * 70)
    print("OBSERVED-SUPPORT ORACLE")
    print("=" * 70)

    print(
        f"Allowed actions: "
        f"{SUPPORTED_ACTIONS}"
    )

    print(
        f"Mean reward: "
        f"{results_df['supported_oracle_reward'].mean():.6f}"
    )

    print(
        f"Median reward: "
        f"{results_df['supported_oracle_reward'].median():.6f}"
    )

    print_distribution(
        "supported_oracle_discount",
        "Observed-Support Oracle Discount Distribution",
    )

    # -----------------------------------------------------------------
    # PPO action-space oracle
    # -----------------------------------------------------------------

    print()
    print("=" * 70)
    print("PPO-ACTION-SPACE ORACLE")
    print("=" * 70)

    print(
        f"Allowed actions: "
        f"{PPO_ACTIONS}"
    )

    print(
        f"Mean reward: "
        f"{results_df['ppo_space_oracle_reward'].mean():.6f}"
    )

    print(
        f"Median reward: "
        f"{results_df['ppo_space_oracle_reward'].median():.6f}"
    )

    print_distribution(
        "ppo_space_oracle_discount",
        "PPO-Action-Space Oracle Discount Distribution",
    )

    # -----------------------------------------------------------------
    # Action-space restriction effect
    # -----------------------------------------------------------------

    print()
    print("=" * 70)
    print("ACTION-SPACE RESTRICTION ANALYSIS")
    print("=" * 70)

    mean_full = (
        results_df["full_oracle_reward"]
        .mean()
    )

    mean_supported = (
        results_df["supported_oracle_reward"]
        .mean()
    )

    mean_ppo_space = (
        results_df["ppo_space_oracle_reward"]
        .mean()
    )

    mean_support_gap = (
        results_df["support_restriction_gap"]
        .mean()
    )

    mean_ppo_gap = (
        results_df["ppo_space_gap"]
        .mean()
    )

    print(
        f"Full action-space oracle:       "
        f"{mean_full:.6f}"
    )

    print(
        f"Observed-support oracle:        "
        f"{mean_supported:.6f}"
    )

    print(
        f"PPO-action-space oracle:        "
        f"{mean_ppo_space:.6f}"
    )

    print()
    print(
        f"Penalty from excluding 0%:      "
        f"{mean_support_gap:.6f}"
    )

    print(
        f"Penalty inside 10/20/30 space:  "
        f"{mean_ppo_gap:.6f}"
    )

    # -----------------------------------------------------------------
    # Relative performance
    # -----------------------------------------------------------------

    print()
    print("=" * 70)
    print("RELATIVE PERFORMANCE")
    print("=" * 70)

    if mean_full != 0:

        print(
            f"Observed-support / full oracle: "
            f"{mean_supported / mean_full * 100:.2f}%"
        )

        print(
            f"PPO-space oracle / full oracle: "
            f"{mean_ppo_space / mean_full * 100:.2f}%"
        )

    if mean_supported != 0:

        print(
            f"PPO-space oracle / "
            f"observed-support oracle: "
            f"{mean_ppo_space / mean_supported * 100:.2f}%"
        )

    # -----------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------

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
        f"Evaluated products: "
        f"{len(results_df)}"
    )

    print(
        f"Saved analysis: "
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()