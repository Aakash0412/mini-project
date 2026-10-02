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

OUTPUT_PATH = Path(
    "models/ppo/reward_landscape_test.csv"
)

DISCOUNT_BINS = [0, 5, 10, 15, 20, 25, 30, 35, 40]

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
    print("PPO REWARD LANDSCAPE ANALYSIS")
    print("=" * 70)

    # ---------------------------------------------------------------
    # Load dataset
    # ---------------------------------------------------------------

    dataset = load_ppo_dataset()

    print(
        f"Loaded economic PPO samples: "
        f"{len(dataset.states)}"
    )

    # ---------------------------------------------------------------
    # Load source dataframe for chronological split
    # ---------------------------------------------------------------

    df = pd.read_parquet(
        "data/processed/products_clean.parquet"
    )

    source_indices = dataset.source_row_indices

    aligned = df.iloc[source_indices].copy()

    # ---------------------------------------------------------------
    # Test split = September–October
    # ---------------------------------------------------------------

    test_mask = aligned["MonthNum"].between(9, 10)

    test_positions = np.flatnonzero(
        test_mask.to_numpy()
    )

    print(
        f"Test products: {len(test_positions)}"
    )

    # ---------------------------------------------------------------
    # Environment
    # ---------------------------------------------------------------

    environment = DynamicPricingEnvironment()

    alpha = environment.reward_config.alpha
    beta = environment.reward_config.beta

    print()
    print("Reward configuration:")
    print(f"  Alpha = {alpha}")
    print(f"  Beta  = {beta}")
    print("  Reward = normalized relative reward")

    # ---------------------------------------------------------------
    # Storage
    # ---------------------------------------------------------------

    rows = []

    # ---------------------------------------------------------------
    # Evaluate every product × every discount
    # ---------------------------------------------------------------

    for counter, position in enumerate(test_positions):

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

        # Margin is required for the current reward.
        if baseline_margin is None:
            continue

        all_outcomes = (
            environment.predict_all_actions()
        )

        product_rewards = {}

        for action, outcome in all_outcomes.items():

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

            product_rewards[action] = reward

            rows.append(
                {
                    "source_row": int(
                        source_indices[position]
                    ),

                    "action": int(action),

                    "discount": float(
                        DISCOUNT_BINS[action]
                    ),

                    "reward": float(reward),

                    "predicted_growth_rate": float(
                        environment.simulator.predict(
                            state,
                            DISCOUNT_BINS[action],
                        )
                    ),

                    "estimated_sales": float(
                        outcome.estimated_sales
                    ),

                    "estimated_revenue": float(
                        outcome.estimated_revenue
                    ),

                    "estimated_margin": float(
                        outcome.estimated_margin
                    ),

                    "baseline_sales": float(
                        baseline_sales
                    ),

                    "baseline_margin": float(
                        baseline_margin
                    ),
                }
            )

        if (counter + 1) % 500 == 0:
            print(
                f"Processed "
                f"{counter + 1}/{len(test_positions)}"
            )

    # ---------------------------------------------------------------
    # Convert to dataframe
    # ---------------------------------------------------------------

    results_df = pd.DataFrame(rows)

    # ---------------------------------------------------------------
    # Determine best action per product
    # ---------------------------------------------------------------

    best_rows = (
        results_df
        .loc[
            results_df.groupby("source_row")[
                "reward"
            ].idxmax()
        ]
        .copy()
    )

    # ---------------------------------------------------------------
    # Aggregate reward by discount
    # ---------------------------------------------------------------

    summary = (
        results_df
        .groupby("discount")
        .agg(
            mean_reward=("reward", "mean"),
            median_reward=("reward", "median"),
            std_reward=("reward", "std"),
            min_reward=("reward", "min"),
            max_reward=("reward", "max"),
            mean_growth=("predicted_growth_rate", "mean"),
            mean_sales=("estimated_sales", "mean"),
            mean_revenue=("estimated_revenue", "mean"),
            mean_margin=("estimated_margin", "mean"),
            samples=("reward", "count"),
        )
        .reset_index()
    )

    # ---------------------------------------------------------------
    # Percentage of products for which each discount is optimal
    # ---------------------------------------------------------------

    best_distribution = (
        best_rows["discount"]
        .value_counts(normalize=True)
        .sort_index()
        * 100
    )

    summary["best_action_percentage"] = (
        summary["discount"]
        .map(best_distribution)
        .fillna(0.0)
    )

    # ---------------------------------------------------------------
    # Print summary
    # ---------------------------------------------------------------

    print()
    print("=" * 70)
    print("REWARD LANDSCAPE BY DISCOUNT")
    print("=" * 70)

    print(
        summary[
            [
                "discount",
                "mean_reward",
                "median_reward",
                "std_reward",
                "best_action_percentage",
                "mean_growth",
                "mean_sales",
                "mean_revenue",
                "mean_margin",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    # ---------------------------------------------------------------
    # Oracle distribution
    # ---------------------------------------------------------------

    print()
    print("=" * 70)
    print("ORACLE DISCOUNT DISTRIBUTION")
    print("=" * 70)

    for discount in DISCOUNT_BINS:

        percentage = (
            best_rows["discount"]
            .eq(discount)
            .mean()
            * 100
        )

        count = (
            best_rows["discount"]
            .eq(discount)
            .sum()
        )

        print(
            f"{discount:>2}% : "
            f"{count:>5} "
            f"({percentage:>6.2f}%)"
        )

    # ---------------------------------------------------------------
    # Overall reward statistics
    # ---------------------------------------------------------------

    oracle_rewards = best_rows["reward"]

    print()
    print("=" * 70)
    print("ORACLE REWARD STATISTICS")
    print("=" * 70)

    print(
        f"Mean:     {oracle_rewards.mean():.6f}"
    )

    print(
        f"Median:   {oracle_rewards.median():.6f}"
    )

    print(
        f"Std:      {oracle_rewards.std():.6f}"
    )

    print(
        f"Min:      {oracle_rewards.min():.6f}"
    )

    print(
        f"Max:      {oracle_rewards.max():.6f}"
    )

    # ---------------------------------------------------------------
    # Save detailed results
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
    print(
        f"Saved detailed reward landscape: "
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()