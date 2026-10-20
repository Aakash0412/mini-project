from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.environment.counterfactual_simulator import CounterfactualSimulator
from src.environment.monotone_simulator import MonotoneCounterfactualSimulator
from src.environment.ppo_dataset import load_ppo_dataset
from src.environment.pricing_environment import DynamicPricingEnvironment
from src.environment.pricing_economics import calculate_pricing_outcome
from src.environment.reward import SALES_REWARD
from src.models.ppo_agent import PPOActor


STATE_DIM = 111
ACTION_DIM = 8
DISCOUNTS = [5, 10, 15, 20, 25, 30, 35, 40]
ENV_ACTIONS = list(range(1, 9))
EPSILON = 1e-8


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare original and monotone-trained 8-action PPO policies "
            "on the same held-out test rows under both response simulators."
        )
    )
    parser.add_argument(
        "--original-checkpoint",
        type=Path,
        default=ROOT / "models/ppo_transfer/ppo_transfer_supported_8action_1000_updates.pth",
    )
    parser.add_argument(
        "--monotone-checkpoint",
        type=Path,
        default=ROOT / "models/ppo_monotone_response/ppo_batched_supported_8action_500_updates.pth",
    )
    parser.add_argument(
        "--products",
        type=Path,
        default=ROOT / "data/processed/products_clean.parquet",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "reports/ppo_monotone_comparison_full",
    )
    parser.add_argument(
        "--max-test-products",
        type=int,
        default=None,
        help="Optional limit for a smoke test; omit for the complete test split.",
    )
    return parser.parse_args()


def load_actor(checkpoint_path: Path):
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    if int(checkpoint.get("state_dim", -1)) != STATE_DIM:
        raise ValueError(
            f"{checkpoint_path}: expected state_dim={STATE_DIM}, "
            f"found {checkpoint.get('state_dim')!r}"
        )
    if int(checkpoint.get("action_dim", -1)) != ACTION_DIM:
        raise ValueError(
            f"{checkpoint_path}: expected action_dim={ACTION_DIM}, "
            f"found {checkpoint.get('action_dim')!r}"
        )

    checkpoint_discounts = checkpoint.get("discount_actions")
    if checkpoint_discounts is not None and list(checkpoint_discounts) != DISCOUNTS:
        raise ValueError(
            f"{checkpoint_path}: unexpected discount_actions "
            f"{checkpoint_discounts!r}; expected {DISCOUNTS}"
        )

    actor = PPOActor(state_dim=STATE_DIM, action_dim=ACTION_DIM)
    actor.load_state_dict(checkpoint["actor_state_dict"], strict=True)
    actor.eval()
    return actor, checkpoint


def choose_action(actor, state: np.ndarray) -> tuple[int, list[float]]:
    state_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        logits = actor(state_tensor)
        probabilities = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

    if probabilities.shape != (ACTION_DIM,) or not np.all(np.isfinite(probabilities)):
        raise RuntimeError("Actor produced invalid action probabilities.")

    action = int(np.argmax(probabilities))
    return action, probabilities.astype(float).tolist()


def relative_reward(outcome, baseline_sales: float, baseline_margin: float) -> float:
    if not np.isfinite(baseline_sales) or abs(baseline_sales) <= EPSILON:
        raise ValueError("Invalid baseline sales.")
    if not np.isfinite(baseline_margin) or abs(baseline_margin) <= EPSILON:
        raise ValueError("Invalid baseline margin.")
    if outcome.estimated_margin is None:
        raise ValueError("Counterfactual outcome has no estimated margin.")

    delta_quantity = float(outcome.estimated_sales) - baseline_sales
    delta_margin = float(outcome.estimated_margin) - baseline_margin
    relative_quantity = delta_quantity / max(abs(baseline_sales), EPSILON)
    relative_margin = delta_margin / max(abs(baseline_margin), EPSILON)

    return float(
        SALES_REWARD.alpha * relative_margin
        + SALES_REWARD.beta * relative_quantity
    )


def main():
    args = parse_args()

    if args.max_test_products is not None and args.max_test_products < 1:
        raise ValueError("--max-test-products must be positive.")

    output_dir = args.output_dir
    output_files = [
        output_dir / "per_product_comparison.csv",
        output_dir / "comparison_summary.json",
    ]
    existing = [str(path) for path in output_files if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing evaluation outputs:\n"
            + "\n".join(existing)
        )

    for required in (args.products, args.original_checkpoint, args.monotone_checkpoint):
        if not required.is_file():
            raise FileNotFoundError(f"Required file not found: {required}")

    original_actor, original_meta = load_actor(args.original_checkpoint)
    monotone_actor, monotone_meta = load_actor(args.monotone_checkpoint)
    actors = {
        "original_ppo": original_actor,
        "monotone_trained_ppo": monotone_actor,
    }

    dataset = load_ppo_dataset()
    products_df = pd.read_parquet(args.products)

    source_indices = np.asarray(dataset.source_row_indices)
    states = np.asarray(dataset.states)

    if states.ndim != 2 or states.shape[1] != STATE_DIM:
        raise ValueError(f"Expected states shaped (N, {STATE_DIM}); got {states.shape}")
    if len(states) != len(source_indices) or len(states) != len(dataset.products):
        raise RuntimeError(
            "State, source-row-index, and product-context lengths do not match."
        )
    if not np.all(np.isfinite(states)):
        raise ValueError("Loaded states contain NaN or Inf.")
    if len(source_indices) == 0:
        raise RuntimeError("The PPO dataset contains no rows.")
    if source_indices.min() < 0 or source_indices.max() >= len(products_df):
        raise IndexError("Source row indices fall outside products dataframe.")
    if "MonthNum" not in products_df.columns:
        raise ValueError("Products dataframe is missing MonthNum.")

    aligned = products_df.iloc[source_indices].reset_index(drop=True)
    test_positions = np.flatnonzero(aligned["MonthNum"].isin([9, 10]).to_numpy())
    if args.max_test_products is not None:
        test_positions = test_positions[: args.max_test_products]
    if len(test_positions) == 0:
        raise RuntimeError("No held-out test rows found for MonthNum 9–10.")

    # Use the same frozen base predictor in both simulator variants.
    base_simulator = CounterfactualSimulator()
    simulators = {
        "original_response": base_simulator,
        "monotone_response": MonotoneCounterfactualSimulator(
            base_simulator=base_simulator
        ),
    }

    results = []
    skipped = 0

    for simulator_name, simulator in simulators.items():
        environment = DynamicPricingEnvironment(
            simulator=simulator,
            reward_config=SALES_REWARD,
        )

        for position in test_positions:
            state = states[position]
            product = dataset.products[position]
            source_row = int(source_indices[position])
            month_num = int(aligned.iloc[position]["MonthNum"])

            environment.reset(state=state, product=product)
            baseline_sales = float(product.last_monthly_sales)
            baseline_margin = environment.observed_outcome.estimated_margin

            if (
                not np.isfinite(baseline_sales)
                or abs(baseline_sales) <= EPSILON
                or baseline_margin is None
                or not np.isfinite(baseline_margin)
                or abs(float(baseline_margin)) <= EPSILON
            ):
                skipped += 1
                continue

            baseline_margin = float(baseline_margin)

            # Predict the full response curve once. This matters because the
            # monotone wrapper projects all discount predictions together.
            outcomes = {}
            rewards = {}
            try:
                growth_by_discount = simulator.predict_all_discounts(state)
                for action_index, discount in enumerate(DISCOUNTS):
                    outcome = calculate_pricing_outcome(
                        observed_price=product.observed_price,
                        observed_discount=product.observed_discount,
                        last_monthly_sales=product.last_monthly_sales,
                        predicted_growth_rate=float(growth_by_discount[discount]),
                        candidate_discount=discount,
                        gross_margin_rate=product.gross_margin_rate,
                    )
                    reward = relative_reward(
                        outcome,
                        baseline_sales=baseline_sales,
                        baseline_margin=baseline_margin,
                    )
                    outcomes[action_index] = outcome
                    rewards[action_index] = reward
            except (ValueError, FloatingPointError, OverflowError, KeyError):
                skipped += 1
                continue

            oracle_action = max(rewards, key=rewards.get)
            oracle_reward = rewards[oracle_action]

            for policy_name, actor in actors.items():
                action, probabilities = choose_action(actor, state)
                outcome = outcomes[action]
                reward = rewards[action]

                row = {
                    "simulator": simulator_name,
                    "policy": policy_name,
                    "source_row_index": source_row,
                    "month_num": month_num,
                    "action_index": action,
                    "recommended_discount_percent": DISCOUNTS[action],
                    "simulated_reward": reward,
                    "oracle_action_index": int(oracle_action),
                    "oracle_discount_percent": DISCOUNTS[oracle_action],
                    "oracle_reward": float(oracle_reward),
                    "oracle_gap": float(oracle_reward - reward),
                    "oracle_action_match": bool(action == oracle_action),
                    "baseline_sales": baseline_sales,
                    "baseline_margin_proxy": baseline_margin,
                    "estimated_sales": float(outcome.estimated_sales),
                    "estimated_revenue": float(outcome.estimated_revenue),
                    "estimated_margin_proxy": float(outcome.estimated_margin),
                    "predicted_growth_percent": float(
                        growth_by_discount[DISCOUNTS[action]]
                    ),
                    "policy_entropy": float(
                        -sum(
                            p * np.log(max(p, 1e-12))
                            for p in probabilities
                        )
                    ),
                }
                row.update(
                    {
                        f"prob_discount_{discount}": probabilities[index]
                        for index, discount in enumerate(DISCOUNTS)
                    }
                )
                results.append(row)

    results_df = pd.DataFrame(results)
    if results_df.empty:
        raise RuntimeError("No valid evaluation rows were produced.")

    # Keep only source rows with all four policy/simulator combinations.
    # If one simulator fails for a row, do not compare that row asymmetrically.
    record_counts = results_df.groupby("source_row_index").size()
    complete_source_rows = record_counts[record_counts == 4].index
    incomplete_source_rows = int((record_counts != 4).sum())
    results_df = results_df[
        results_df["source_row_index"].isin(complete_source_rows)
    ].copy()
    if results_df.empty:
        raise RuntimeError(
            "No source rows had complete results for both policies under both simulators."
        )

    summary = {
        "experiment": "original versus monotone-trained PPO; held-out evaluation",
        "test_months": [9, 10],
        "test_rows_selected_before_validity_filter": int(len(test_positions)),
        "valid_source_rows": int(results_df["source_row_index"].nunique()),
        "policy_simulator_records": int(len(results_df)),
        "skipped_row_simulator_cases": int(skipped),
        "incomplete_source_rows_excluded_from_comparison": incomplete_source_rows,
        "state_dimension": STATE_DIM,
        "action_dimension": ACTION_DIM,
        "discount_actions_percent": DISCOUNTS,
        "reward": {
            "name": "SALES_REWARD",
            "alpha_margin": float(SALES_REWARD.alpha),
            "beta_quantity": float(SALES_REWARD.beta),
        },
        "original_checkpoint": str(args.original_checkpoint),
        "original_checkpoint_updates": original_meta.get("updates"),
        "monotone_checkpoint": str(args.monotone_checkpoint),
        "monotone_checkpoint_updates": monotone_meta.get("updates"),
        "sales_predictor_checkpoint_dir": str(
            getattr(base_simulator, "checkpoint_dir", "default CounterfactualSimulator")
        ),
        "simulator_definitions": {
            "original_response": "Frozen sales predictor without monotonic projection.",
            "monotone_response": (
                "Same frozen sales predictor with per-state isotonic non-decreasing "
                "projection over supported discount bins."
            ),
        },
        "limitations": [
            "The monotonic response constraint is an experimental assumption, not a verified causal demand relationship.",
            "Both simulator variants rely on the same frozen sales predictor.",
            "Estimated margin is a dataset-derived proxy, not verified accounting profit.",
            "Offline simulated outcomes do not establish real-world causal impact.",
            "The comparison is limited to the supported 5%-40% discount actions.",
        ],
        "results_by_simulator_and_policy": {},
    }

    for (simulator_name, policy_name), subset in results_df.groupby(
        ["simulator", "policy"], sort=True
    ):
        distribution = {
            str(int(discount)): int(count)
            for discount, count in subset[
                "recommended_discount_percent"
            ].value_counts().sort_index().items()
        }
        summary["results_by_simulator_and_policy"][
            f"{simulator_name}__{policy_name}"
        ] = {
            "n": int(len(subset)),
            "mean_simulated_reward": float(subset["simulated_reward"].mean()),
            "median_simulated_reward": float(subset["simulated_reward"].median()),
            "mean_oracle_reward": float(subset["oracle_reward"].mean()),
            "mean_oracle_gap": float(subset["oracle_gap"].mean()),
            "oracle_action_match_percent": float(
                subset["oracle_action_match"].mean() * 100.0
            ),
            "mean_policy_entropy": float(subset["policy_entropy"].mean()),
            "discount_distribution_count": distribution,
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_files[0], index=False)
    output_files[1].write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Evaluation complete.")
    print(f"Selected held-out rows: {len(test_positions)}")
    print(f"Valid source rows: {results_df['source_row_index'].nunique()}")
    print(f"Skipped row/simulator cases: {skipped}")
    print(f"Results: {output_files[0]}")
    print(f"Summary: {output_files[1]}")
    print("\nSummary by simulator and policy:")
    for key, metrics in summary["results_by_simulator_and_policy"].items():
        print(
            f"{key}: n={metrics['n']}, "
            f"mean_reward={metrics['mean_simulated_reward']:.6f}, "
            f"oracle_gap={metrics['mean_oracle_gap']:.6f}, "
            f"oracle_match={metrics['oracle_action_match_percent']:.2f}%"
        )


if __name__ == "__main__":
    main()
