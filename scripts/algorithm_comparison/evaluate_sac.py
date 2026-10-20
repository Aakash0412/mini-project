from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from stable_baselines3 import SAC

from src.environment.ppo_dataset import load_ppo_dataset
from src.environment.pricing_environment import DynamicPricingEnvironment
from src.environment.reward import SALES_REWARD


CHECKPOINT = PROJECT_ROOT / "models" / "sac" / "sac_sales_seed42.zip"
OUTPUT = (
    PROJECT_ROOT
    / "reports"
    / "algorithm_comparison"
    / "sac_test_evaluation.csv"
)
DISCOUNT_BINS = [5, 10, 15, 20, 25, 30, 35, 40]
EPSILON = 1e-8


def relative_reward(
    delta_margin: float,
    delta_sales: float,
    baseline_margin: float,
    baseline_sales: float,
) -> float:
    relative_margin = delta_margin / max(abs(baseline_margin), EPSILON)
    relative_sales = delta_sales / max(abs(baseline_sales), EPSILON)
    return float(
        SALES_REWARD.alpha * relative_margin
        + SALES_REWARD.beta * relative_sales
    )


def main() -> None:
    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"SAC checkpoint not found: {CHECKPOINT}")

    if OUTPUT.exists():
        raise FileExistsError(
            f"Evaluation output already exists: {OUTPUT}. "
            "Preserve it and choose a new output name before rerunning."
        )

    dataset = load_ppo_dataset()
    products_df = pd.read_parquet(
        PROJECT_ROOT / "data" / "processed" / "products_clean.parquet"
    )
    aligned = products_df.iloc[dataset.source_row_indices].copy()

    test_mask = aligned["MonthNum"].between(9, 10).to_numpy()
    positions = np.flatnonzero(test_mask)

    if len(positions) == 0:
        raise RuntimeError("No test products found for MonthNum 9-10.")

    model = SAC.load(str(CHECKPOINT), device="cpu")
    environment = DynamicPricingEnvironment(reward_config=SALES_REWARD)
    results = []

    print("SAC held-out test evaluation")
    print("Checkpoint:", CHECKPOINT)
    print("Test products before margin validation:", len(positions))
    print("Discount support:", DISCOUNT_BINS)

    for position in positions:
        state = dataset.states[position]
        product = dataset.products[position]

        environment.reset(state=state, product=product)
        baseline_sales = float(product.last_monthly_sales)
        baseline_margin = environment.observed_outcome.estimated_margin

        if baseline_margin is None or not np.isfinite(baseline_margin):
            continue

        continuous_action, _ = model.predict(state, deterministic=True)
        continuous_action = np.asarray(continuous_action).reshape(-1)

        if continuous_action.size != 1 or not np.isfinite(continuous_action[0]):
            raise ValueError(
                f"Invalid SAC action for dataset position {position}"
            )

        proposed_discount = float(
            (np.clip(continuous_action[0], -1.0, 1.0) + 1.0) * 20.0
        )
        selected_discount = min(
            DISCOUNT_BINS, key=lambda value: abs(value - proposed_discount)
        )
        environment_action = DISCOUNT_BINS.index(selected_discount) + 1

        selected_outcome = environment.predict_action(environment_action)
        delta_sales = float(selected_outcome.estimated_sales - baseline_sales)
        delta_margin = float(
            selected_outcome.estimated_margin - baseline_margin
        )
        reward = relative_reward(
            delta_margin, delta_sales, baseline_margin, baseline_sales
        )

        all_outcomes = environment.predict_all_actions()
        oracle_candidates = []

        for action, outcome in all_outcomes.items():
            if action == 0 or outcome.estimated_margin is None:
                continue

            candidate_sales_delta = float(
                outcome.estimated_sales - baseline_sales
            )
            candidate_margin_delta = float(
                outcome.estimated_margin - baseline_margin
            )
            candidate_reward = relative_reward(
                candidate_margin_delta,
                candidate_sales_delta,
                baseline_margin,
                baseline_sales,
            )
            oracle_candidates.append(
                {
                    "action": action,
                    "discount": float(outcome.discount_percent),
                    "reward": candidate_reward,
                }
            )

        if not oracle_candidates:
            continue

        oracle = max(oracle_candidates, key=lambda item: item["reward"])

        results.append(
            {
                "split": "test",
                "source_row": int(dataset.source_row_indices[position]),
                "continuous_action": float(continuous_action[0]),
                "proposed_discount": proposed_discount,
                "sac_discount": float(selected_discount),
                "sac_reward": reward,
                "sac_sales": float(selected_outcome.estimated_sales),
                "sac_revenue": float(selected_outcome.estimated_revenue),
                "sac_margin": float(selected_outcome.estimated_margin),
                "baseline_sales": baseline_sales,
                "baseline_margin": float(baseline_margin),
                "oracle_discount": oracle["discount"],
                "oracle_reward": oracle["reward"],
                "oracle_gap": oracle["reward"] - reward,
                "oracle_action_match": (
                    float(selected_discount) == oracle["discount"]
                ),
            }
        )

    results_df = pd.DataFrame(results)
    if results_df.empty:
        raise RuntimeError("No valid SAC test results were produced.")

    numeric_columns = results_df.select_dtypes(include=[np.number]).columns
    if not np.isfinite(results_df[numeric_columns].to_numpy()).all():
        raise ValueError("Non-finite values detected in evaluation results.")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(OUTPUT, index=False)

    print("\nTEST SUMMARY")
    print("Evaluated products:", len(results_df))
    print(f"SAC mean reward: {results_df['sac_reward'].mean():.6f}")
    print(f"Oracle mean reward: {results_df['oracle_reward'].mean():.6f}")
    print(f"Mean oracle gap: {results_df['oracle_gap'].mean():.6f}")
    print(
        "Oracle action match:",
        f"{100 * results_df['oracle_action_match'].mean():.2f}%",
    )
    print("\nSAC discount distribution:")
    print(
        results_df["sac_discount"]
        .value_counts(normalize=True)
        .reindex(DISCOUNT_BINS, fill_value=0)
        .mul(100)
        .round(2)
        .to_string()
    )
    print("\nSaved:", OUTPUT)


if __name__ == "__main__":
    main()
