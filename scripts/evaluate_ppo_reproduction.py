from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT)
)

from src.models.ppo_agent import PPOActor
from src.environment.ppo_dataset import load_ppo_dataset
from src.environment.counterfactual_simulator import CounterfactualSimulator
from src.environment.pricing_economics import calculate_pricing_outcome
from src.environment.reward import SALES_REWARD, calculate_reward


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "models/ppo/ppo_batched_economic_50_updates.pth"
PRODUCTS = ROOT / "data/processed/products_clean.parquet"
OUTPUT_DIR = ROOT / "reports/base_paper_reproduction"
DISCOUNTS = [0, 5, 10, 15, 20, 25, 30, 35, 40]


def main():
    # Preserve every existing artifact.
    outputs = [
        OUTPUT_DIR / "ppo_test_predictions.csv",
        OUTPUT_DIR / "ppo_test_action_distribution.csv",
        OUTPUT_DIR / "ppo_test_summary.json",
        OUTPUT_DIR / "all_action_counterfactuals.csv",
    ]
    existing = [str(path) for path in outputs if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing files:\n" + "\n".join(existing)
        )

    for path in (CHECKPOINT, PRODUCTS):
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    device = torch.device("cpu")
    checkpoint = torch.load(
        CHECKPOINT, map_location=device, weights_only=False
    )

    if checkpoint.get("state_dim") != 111 or checkpoint.get("action_dim") != 9:
        raise ValueError(
            "Checkpoint dimensions do not match the expected 111-state, "
            "9-action configuration."
        )

    actor = PPOActor(state_dim=111, action_dim=9).to(device)
    actor.load_state_dict(checkpoint["actor_state_dict"])
    actor.eval()

    dataset = load_ppo_dataset(
        state_path=ROOT / "data/processed/state/state.npy",
        index_path=ROOT / "data/processed/state/state_row_indices.npy",
        products_path=PRODUCTS,
    )

    products_df = pd.read_parquet(PRODUCTS)
    aligned = products_df.iloc[dataset.source_row_indices].reset_index(drop=True)

    if len(dataset.states) != len(aligned):
        raise RuntimeError("State/product alignment check failed.")

    if "MonthNum" not in aligned.columns:
        raise ValueError("MonthNum is required for the chronological test split.")

    test_mask = aligned["MonthNum"].isin([9, 10]).to_numpy()
    states = dataset.states[test_mask]
    test_products = aligned.loc[test_mask].reset_index(drop=True)
    contexts = [
        context for context, keep in zip(dataset.products, test_mask) if keep
    ]

    if len(states) == 0:
        raise RuntimeError("No test observations found for MonthNum 9–10.")

    # Deterministic policy: choose the highest-probability action.
    with torch.no_grad():
        logits = actor(torch.as_tensor(states, dtype=torch.float32, device=device))
        probabilities = torch.softmax(logits, dim=-1).cpu().numpy()

    chosen_actions = probabilities.argmax(axis=1)
    chosen_discounts = [DISCOUNTS[index] for index in chosen_actions]

    simulator = CounterfactualSimulator(
        checkpoint_dir=str(ROOT / "models/sales_predictor"),
        device=device,
    )

    prediction_rows = []
    counterfactual_rows = []

    for i, (state, context, discount, action, probs) in enumerate(
        zip(states, contexts, chosen_discounts, chosen_actions, probabilities)
    ):
        # Baseline at the product's observed discount and zero growth.
        baseline = calculate_pricing_outcome(
            observed_price=context.observed_price,
            observed_discount=context.observed_discount,
            last_monthly_sales=context.last_monthly_sales,
            predicted_growth_rate=0.0,
            candidate_discount=context.observed_discount,
            gross_margin_rate=context.gross_margin_rate,
        )

        action_results = []

        for candidate_discount in DISCOUNTS:
            growth = simulator.predict(state, candidate_discount)
            outcome = calculate_pricing_outcome(
                observed_price=context.observed_price,
                observed_discount=context.observed_discount,
                last_monthly_sales=context.last_monthly_sales,
                predicted_growth_rate=growth,
                candidate_discount=candidate_discount,
                gross_margin_rate=context.gross_margin_rate,
            )

            # Match the project's normalized, paper-style reward convention.
            delta_quantity = (
                (outcome.estimated_sales - baseline.estimated_sales)
                / baseline.estimated_sales
                if baseline.estimated_sales > 0
                else np.nan
            )

            delta_margin = (
                (outcome.estimated_margin - baseline.estimated_margin)
                / abs(baseline.estimated_margin)
                if baseline.estimated_margin is not None
                and abs(baseline.estimated_margin) > 1e-12
                else np.nan
            )

            reward = (
                calculate_reward(delta_margin, delta_quantity, SALES_REWARD)
                if np.isfinite(delta_margin) and np.isfinite(delta_quantity)
                else np.nan
            )

            row = {
                "test_position": i,
                "source_row_index": int(dataset.source_row_indices[np.flatnonzero(test_mask)[i]]),
                "month_num": int(test_products.iloc[i]["MonthNum"]),
                "discount_percent": candidate_discount,
                "predicted_growth_percent": growth,
                "estimated_sales": outcome.estimated_sales,
                "estimated_revenue": outcome.estimated_revenue,
                "estimated_margin_proxy": outcome.estimated_margin,
                "delta_quantity": delta_quantity,
                "delta_margin_proxy": delta_margin,
                "sales_weighted_reward": reward,
            }
            counterfactual_rows.append(row)
            action_results.append(row)

        chosen = action_results[int(action)]

        prediction = {
            "test_position": i,
            "source_row_index": int(
                dataset.source_row_indices[np.flatnonzero(test_mask)[i]]
            ),
            "month_num": int(test_products.iloc[i]["MonthNum"]),
            "observed_discount_percent": context.observed_discount,
            "recommended_action_index": int(action),
            "recommended_discount_percent": int(discount),
            "predicted_growth_percent": chosen["predicted_growth_percent"],
            "estimated_sales": chosen["estimated_sales"],
            "estimated_revenue": chosen["estimated_revenue"],
            "estimated_margin_proxy": chosen["estimated_margin_proxy"],
            "simulated_reward": chosen["sales_weighted_reward"],
            "policy_entropy": float(
                -(probs * np.log(np.clip(probs, 1e-12, 1))).sum()
            ),
        }
        prediction.update(
            {f"prob_discount_{d}": float(p) for d, p in zip(DISCOUNTS, probs)}
        )
        prediction_rows.append(prediction)

    predictions = pd.DataFrame(prediction_rows)
    counterfactuals = pd.DataFrame(counterfactual_rows)

    distribution = (
        predictions["recommended_discount_percent"]
        .value_counts()
        .reindex(DISCOUNTS, fill_value=0)
        .rename_axis("discount_percent")
        .reset_index(name="product_count")
    )
    distribution["product_share_percent"] = (
        distribution["product_count"] / len(predictions) * 100
    )

    valid_rewards = predictions["simulated_reward"].dropna()
    summary = {
        "evaluation": "deterministic held-out PPO policy evaluation",
        "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
        "checkpoint_updates": checkpoint.get("updates"),
        "seed": checkpoint.get("seed"),
        "state_dimension": 111,
        "action_dimension": 9,
        "discount_bins_percent": DISCOUNTS,
        "test_months": [9, 10],
        "test_products": int(len(predictions)),
        "reward_definition": "alpha=0.2, beta=0.8",
        "mean_simulated_reward": (
            float(valid_rewards.mean()) if len(valid_rewards) else None
        ),
        "median_simulated_reward": (
            float(valid_rewards.median()) if len(valid_rewards) else None
        ),
        "valid_reward_count": int(len(valid_rewards)),
        "invalid_reward_count": int(predictions["simulated_reward"].isna().sum()),
        "mean_predicted_growth_percent": float(
            predictions["predicted_growth_percent"].mean()
        ),
        "mean_estimated_sales": float(predictions["estimated_sales"].mean()),
        "mean_estimated_revenue": float(predictions["estimated_revenue"].mean()),
        "mean_margin_proxy": float(predictions["estimated_margin_proxy"].mean()),
        "mean_policy_entropy": float(predictions["policy_entropy"].mean()),
        "limitations": [
            "111-D state omits unavailable raw-review sentiment features.",
            "Margin is a dataset-derived proxy, not verified accounting profit.",
            "Counterfactual outcomes depend on the frozen sales predictor.",
            "0% discount is outside the observed training-discount support.",
            "Offline simulated results do not establish real-world causal impact.",
        ],
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(outputs[0], index=False)
    distribution.to_csv(outputs[1], index=False)
    outputs[2].write_text(json.dumps(summary, indent=2), encoding="utf-8")
    counterfactuals.to_csv(outputs[3], index=False)

    print("Evaluation complete.")
    print("Test products:", len(predictions))
    print("Mean simulated reward:", summary["mean_simulated_reward"])
    print("\nRecommended discount distribution:")
    print(distribution.to_string(index=False))
    print("\nNew files:")
    for path in outputs:
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()

