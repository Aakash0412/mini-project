
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.ppo_agent import PPOActor
from src.environment.ppo_dataset import load_ppo_dataset
from src.environment.counterfactual_simulator import CounterfactualSimulator
from src.environment.pricing_economics import calculate_pricing_outcome
from src.environment.reward import SALES_REWARD, calculate_reward

CHECKPOINT = ROOT / "models/ppo/ppo_batched_economic_50_updates.pth"
PRODUCTS = ROOT / "data/processed/products_clean.parquet"
OUTPUT_DIR = ROOT / "reports/base_paper_reproduction/economic_robustness"

DISCOUNTS = np.array([0, 5, 10, 15, 20, 25, 30, 35, 40])
NOISE_LEVELS = [0.00, 0.01, 0.05, 0.10]
SEED = 42
BATCH_SIZE = 1024


def policy_probabilities(actor, states):
    chunks = []
    with torch.no_grad():
        for start in range(0, len(states), BATCH_SIZE):
            batch = torch.as_tensor(
                states[start:start + BATCH_SIZE], dtype=torch.float32
            )
            chunks.append(torch.softmax(actor(batch), dim=-1).cpu().numpy())
    return np.concatenate(chunks, axis=0)


def evaluate_outcome(context, simulator, state, discount, baseline):
    growth = simulator.predict(state, int(discount))
    outcome = calculate_pricing_outcome(
        observed_price=context.observed_price,
        observed_discount=context.observed_discount,
        last_monthly_sales=context.last_monthly_sales,
        predicted_growth_rate=growth,
        candidate_discount=int(discount),
        gross_margin_rate=context.gross_margin_rate,
    )

    delta_quantity = (
        (outcome.estimated_sales - baseline.estimated_sales)
        / baseline.estimated_sales
        if baseline.estimated_sales > 0 else np.nan
    )
    delta_margin = (
        (outcome.estimated_margin - baseline.estimated_margin)
        / abs(baseline.estimated_margin)
        if outcome.estimated_margin is not None
        and baseline.estimated_margin is not None
        and abs(baseline.estimated_margin) > 1e-12
        else np.nan
    )
    reward = (
        calculate_reward(delta_margin, delta_quantity, SALES_REWARD)
        if np.isfinite(delta_margin) and np.isfinite(delta_quantity)
        else np.nan
    )

    return {
        "discount_percent": int(discount),
        "predicted_growth_percent": growth,
        "estimated_sales": outcome.estimated_sales,
        "estimated_revenue": outcome.estimated_revenue,
        "estimated_margin_proxy": outcome.estimated_margin,
        "delta_quantity": delta_quantity,
        "delta_margin_proxy": delta_margin,
        "simulated_reward": reward,
    }


def main():
    for path in (CHECKPOINT, PRODUCTS):
        if not path.exists():
            raise FileNotFoundError(path)

    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty; refusing to overwrite: {OUTPUT_DIR}"
        )

    checkpoint = torch.load(
        CHECKPOINT, map_location="cpu", weights_only=False
    )
    if checkpoint.get("state_dim") != 111 or checkpoint.get("action_dim") != 9:
        raise ValueError("Expected a 111-state, 9-action checkpoint.")

    dataset = load_ppo_dataset(
        state_path=ROOT / "data/processed/state/state.npy",
        index_path=ROOT / "data/processed/state/state_row_indices.npy",
        products_path=PRODUCTS,
    )
    products_df = pd.read_parquet(PRODUCTS)
    aligned = products_df.iloc[dataset.source_row_indices].reset_index(drop=True)

    test_mask = aligned["MonthNum"].isin([9, 10]).to_numpy()
    test_positions = np.flatnonzero(test_mask)
    states = np.asarray(dataset.states[test_mask], dtype=np.float32)
    contexts = [
        context for context, keep in zip(dataset.products, test_mask) if keep
    ]
    source_rows = np.asarray(dataset.source_row_indices)[test_mask]

    if len(states) != 7139 or len(contexts) != len(states):
        raise RuntimeError(
            f"Unexpected test alignment: states={len(states)}, contexts={len(contexts)}"
        )

    train_mask = aligned["MonthNum"].between(1, 6).to_numpy()
    train_states = np.asarray(dataset.states[train_mask], dtype=np.float64)
    feature_std = np.maximum(train_states.std(axis=0), 1e-8)

    actor = PPOActor(state_dim=111, action_dim=9)
    actor.load_state_dict(checkpoint["actor_state_dict"])
    actor.eval()

    simulator = CounterfactualSimulator(
        checkpoint_dir=str(ROOT / "models/sales_predictor"),
        device=torch.device("cpu"),
    )

    # Keep the same synthetic noise realization at every nonzero level.
    rng = np.random.default_rng(SEED)
    shared_noise = rng.standard_normal(states.shape).astype(np.float32)

    # Product-specific baseline: observed discount, zero predicted growth.
    baselines = []
    for context in contexts:
        baselines.append(calculate_pricing_outcome(
            observed_price=context.observed_price,
            observed_discount=context.observed_discount,
            last_monthly_sales=context.last_monthly_sales,
            predicted_growth_rate=0.0,
            candidate_discount=context.observed_discount,
            gross_margin_rate=context.gross_margin_rate,
        ))

    # Cache counterfactual predictions for every state/noise/action combination.
    rows = []
    for level in NOISE_LEVELS:
        perturbed_states = states + (
            level * feature_std.astype(np.float32) * shared_noise
        )
        probs = policy_probabilities(actor, perturbed_states)
        actions = probs.argmax(axis=1)
        chosen_discounts = DISCOUNTS[actions]

        for i, (state, context, baseline) in enumerate(
            zip(perturbed_states, contexts, baselines)
        ):
            for discount in sorted(set([int(chosen_discounts[i]), 20])):
                result = evaluate_outcome(
                    context, simulator, state, discount, baseline
                )
                rows.append({
                    "noise_fraction_of_train_std": level,
                    "source_row": int(source_rows[i]),
                    "test_position": i,
                    "policy_discount": int(chosen_discounts[i]),
                    "comparison_discount": discount,
                    "is_policy_action": discount == int(chosen_discounts[i]),
                    **result,
                })

    outcomes = pd.DataFrame(rows)
    # Each product has one policy action and one fixed-20 action per level.
    # When PPO itself selects 20%, the shared outcome is represented once;
    # mark it as both the policy and comparison action.
    outcomes = outcomes.drop_duplicates(
        subset=["noise_fraction_of_train_std", "source_row",
                "comparison_discount"],
        keep="first",
    )

    # Rebuild unambiguous product-level records, retaining policy and fixed-20
    # outcomes separately even when their discounts coincide.
    detail_rows = []
    clean_actions = None

    for level in NOISE_LEVELS:
        perturbed_states = states + (
            level * feature_std.astype(np.float32) * shared_noise
        )
        probs = policy_probabilities(actor, perturbed_states)
        policy_discounts = DISCOUNTS[probs.argmax(axis=1)]

        if level == 0:
            clean_actions = policy_discounts.copy()

        for i, (state, context, baseline) in enumerate(
            zip(perturbed_states, contexts, baselines)
        ):
            policy_outcome = evaluate_outcome(
                context, simulator, state, policy_discounts[i], baseline
            )
            fixed_outcome = evaluate_outcome(
                context, simulator, state, 20, baseline
            )

            for method, discount, result in (
                ("ppo", int(policy_discounts[i]), policy_outcome),
                ("fixed_20", 20, fixed_outcome),
            ):
                detail_rows.append({
                    "noise_fraction_of_train_std": level,
                    "source_row": int(source_rows[i]),
                    "method": method,
                    **result,
                })

    detail = pd.DataFrame(detail_rows)
    detail.to_csv(OUTPUT_DIR / "economic_robustness_by_product.csv", index=False)

    summaries = []
    for level in NOISE_LEVELS:
        level_df = detail[detail["noise_fraction_of_train_std"] == level]
        for method in ("ppo", "fixed_20"):
            frame = level_df[level_df["method"] == method]
            summaries.append({
                "noise_fraction_of_train_std": level,
                "method": method,
                "test_products": len(frame),
                "mean_reward": frame["simulated_reward"].mean(),
                "median_reward": frame["simulated_reward"].median(),
                "mean_revenue": frame["estimated_revenue"].mean(),
                "mean_margin_proxy": frame["estimated_margin_proxy"].mean(),
                "mean_estimated_sales": frame["estimated_sales"].mean(),
                "valid_reward_count": int(frame["simulated_reward"].notna().sum()),
                "adverse_reward_share_percent": (
                    frame["simulated_reward"].lt(0).mean() * 100
                ),
            })

    summary = pd.DataFrame(summaries)
    summary.to_csv(OUTPUT_DIR / "economic_robustness_summary.csv", index=False)

    # Paired PPO-minus-fixed-20 comparison at each perturbation level.
    paired_rows = []
    for level in NOISE_LEVELS:
        frame = detail[detail["noise_fraction_of_train_std"] == level]
        ppo = frame[frame["method"] == "ppo"].set_index("source_row")
        fixed = frame[frame["method"] == "fixed_20"].set_index("source_row")
        paired = ppo.join(
            fixed, lsuffix="_ppo", rsuffix="_fixed", how="inner"
        )

        for metric in (
            "simulated_reward", "estimated_revenue",
            "estimated_margin_proxy", "estimated_sales",
        ):
            difference = (
                paired[f"{metric}_ppo"] - paired[f"{metric}_fixed"]
            ).dropna().to_numpy()
            paired_rows.append({
                "noise_fraction_of_train_std": level,
                "metric": metric,
                "paired_products": len(difference),
                "mean_ppo_minus_fixed20": (
                    float(difference.mean()) if len(difference) else None
                ),
                "median_ppo_minus_fixed20": (
                    float(np.median(difference)) if len(difference) else None
                ),
                "ppo_win_share_percent": (
                    float((difference > 0).mean() * 100)
                    if len(difference) else None
                ),
                "tie_share_percent": (
                    float((difference == 0).mean() * 100)
                    if len(difference) else None
                ),
            })

    paired_df = pd.DataFrame(paired_rows)
    paired_df.to_csv(OUTPUT_DIR / "ppo_vs_fixed20_paired.csv", index=False)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metadata = {
        "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
        "checkpoint_updates": checkpoint.get("updates"),
        "test_months": [9, 10],
        "test_products": len(states),
        "discount_actions": DISCOUNTS.tolist(),
        "noise_levels": NOISE_LEVELS,
        "seed": SEED,
        "reward": {"alpha": SALES_REWARD.alpha, "beta": SALES_REWARD.beta},
        "baseline": "product-specific observed discount with zero predicted growth",
        "fixed_policy": "20% discount",
        "limitations": [
            "Synthetic Gaussian state perturbations are not calibrated measurement errors.",
            "Economic results are conditional on the frozen sales predictor.",
            "Margin is a revenue-times-gross-margin-rate proxy, not verified accounting profit.",
            "Offline simulated results do not establish real-world causal impact.",
            "The 0% discount action is outside observed training discount support.",
        ],
    }
    (OUTPUT_DIR / "economic_robustness_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print("Economic robustness evaluation complete.")
    print("Output directory:", OUTPUT_DIR)
    print("\nSummary:")
    print(summary.to_string(index=False))
    print("\nPPO vs fixed-20 paired comparison:")
    print(paired_df.to_string(index=False))


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    main()
