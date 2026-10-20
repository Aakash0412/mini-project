from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path.cwd().resolve()
sys.path.insert(0, str(ROOT))

from src.environment.ppo_dataset import load_ppo_dataset
from src.environment.pricing_environment import DynamicPricingEnvironment
from src.environment.reward import get_reward_config
from src.models.ppo_agent import PPOAgent

CHECKPOINT = ROOT / "models/ppo_transfer/ppo_transfer_supported_8action_1000_updates.pth"
PRODUCTS = ROOT / "data/processed/products_clean.parquet"
OUT = ROOT / "reports/robustness/ppo_transfer_economic_robustness"
DISCOUNTS = np.array([5, 10, 15, 20, 25, 30, 35, 40])
LEVELS = [0.00, 0.01, 0.05, 0.10]
SEED = 42
BATCH_SIZE = 1024
EPS = 1e-8

def main():
    if not CHECKPOINT.is_file() or not PRODUCTS.is_file():
        raise FileNotFoundError("Required checkpoint or product data is missing.")
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing results: {OUT}")

    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    if ckpt.get("state_dim") != 111 or ckpt.get("action_dim") != 8:
        raise ValueError("Expected final checkpoint with state_dim=111, action_dim=8.")

    dataset = load_ppo_dataset()
    df = pd.read_parquet(PRODUCTS)
    source_indices = np.asarray(dataset.source_row_indices, dtype=int)
    aligned = df.iloc[source_indices].reset_index(drop=True)
    states = np.asarray(dataset.states, dtype=np.float32)

    if states.shape != (len(dataset.products), 111) or len(states) != len(aligned):
        raise ValueError("Dataset state/product alignment check failed.")
    if not np.isfinite(states).all():
        raise ValueError("States contain NaN or infinity.")

    train_mask = aligned["MonthNum"].between(1, 6).to_numpy()
    test_positions = np.flatnonzero(aligned["MonthNum"].between(9, 10).to_numpy())
    if not train_mask.any() or not len(test_positions):
        raise ValueError("Training or held-out test split is empty.")

    feature_std = np.maximum(states[train_mask].astype(np.float64).std(axis=0), 1e-8).astype(np.float32)
    test_states = states[test_positions]
    rng = np.random.default_rng(SEED)
    noise = rng.standard_normal(test_states.shape).astype(np.float32)

    agent = PPOAgent(state_dim=111, action_dim=8)
    agent.actor.load_state_dict(ckpt["actor_state_dict"])
    agent.actor.eval()

    def policy_actions(x):
        actions = []
        with torch.no_grad():
            for start in range(0, len(x), BATCH_SIZE):
                batch = torch.as_tensor(x[start:start + BATCH_SIZE])
                actions.extend(torch.argmax(agent.actor(batch), dim=-1).numpy().tolist())
        return np.asarray(actions, dtype=int)

    baseline_actions = policy_actions(test_states)
    action_by_level = {
        level: policy_actions(test_states + level * feature_std * noise)
        for level in LEVELS
    }

    # Evaluate all selected actions against original, unperturbed product states.
    # This isolates action-selection sensitivity from economic-model input changes.
    environment = DynamicPricingEnvironment(reward_config=get_reward_config("sales"))
    alpha = environment.reward_config.alpha
    beta = environment.reward_config.beta

    detail = []
    for j, position in enumerate(test_positions):
        state = states[position]
        product = dataset.products[position]
        environment.reset(state=state, product=product)

        baseline_sales = float(product.last_monthly_sales)
        baseline_margin = environment.observed_outcome.estimated_margin
        if baseline_margin is None or not np.isfinite(baseline_margin):
            continue

        def score(outcome):
            delta_q = outcome.estimated_sales - baseline_sales
            delta_m = outcome.estimated_margin - baseline_margin
            reward = (
                alpha * delta_m / max(abs(baseline_margin), EPS)
                + beta * delta_q / max(abs(baseline_sales), EPS)
            )
            return float(reward), float(outcome.estimated_sales), float(outcome.estimated_margin)

        # Offline oracle: best modeled reward across the same supported 5%-40% actions.
        oracle_candidates = []
        for env_action, outcome in environment.predict_all_actions().items():
            if env_action == 0 or outcome.estimated_margin is None:
                continue
            r, q, m = score(outcome)
            if np.isfinite(r) and np.isfinite(q) and np.isfinite(m):
                oracle_candidates.append((r, env_action, q, m))
        if not oracle_candidates:
            continue
        oracle_reward, oracle_env_action, oracle_sales, oracle_margin = max(
            oracle_candidates, key=lambda x: x[0]
        )

        # Baseline and noisy policies use the same economic state and simulator.
        row = {
            "source_row": int(source_indices[position]),
            "baseline_sales": baseline_sales,
            "baseline_margin_proxy": float(baseline_margin),
            "oracle_discount": int(oracle_env_action * 5),
            "oracle_reward": oracle_reward,
        }

        for level in LEVELS:
            action = int(action_by_level[level][j])
            discount = int(DISCOUNTS[action])
            # Environment action 1..8 corresponds to PPO discount 5..40%.
            outcome = environment.predict_action(action + 1)
            reward, sales, margin = score(outcome)
            prefix = f"noise_{int(level * 100):02d}pct"
            row.update({
                f"{prefix}_discount": discount,
                f"{prefix}_reward": reward,
                f"{prefix}_estimated_sales": sales,
                f"{prefix}_estimated_margin_proxy": margin,
                f"{prefix}_sales_change_pct": (
                    100 * (sales - baseline_sales) / max(abs(baseline_sales), EPS)
                ),
                f"{prefix}_margin_change_pct": (
                    100 * (margin - baseline_margin) / max(abs(baseline_margin), EPS)
                ),
                f"{prefix}_oracle_reward_gap": oracle_reward - reward,
                f"{prefix}_oracle_reward_ratio_pct": (
                    100 * reward / oracle_reward if abs(oracle_reward) > EPS else np.nan
                ),
            })
        detail.append(row)

    if not detail:
        raise RuntimeError("No test products had valid margin and oracle outcomes.")

    detail_df = pd.DataFrame(detail)
    summary = []
    for level in LEVELS:
        p = f"noise_{int(level * 100):02d}pct"
        rewards = detail_df[f"{p}_reward"]
        ratios = detail_df[f"{p}_oracle_reward_ratio_pct"]
        summary.append({
            "noise_fraction_of_train_std": level,
            "evaluated_products": len(detail_df),
            "mean_reward": float(rewards.mean()),
            "median_reward": float(rewards.median()),
            "mean_estimated_sales": float(detail_df[f"{p}_estimated_sales"].mean()),
            "mean_estimated_margin_proxy": float(detail_df[f"{p}_estimated_margin_proxy"].mean()),
            "mean_sales_change_pct": float(detail_df[f"{p}_sales_change_pct"].mean()),
            "mean_margin_change_pct": float(detail_df[f"{p}_margin_change_pct"].mean()),
            "mean_oracle_reward": float(detail_df["oracle_reward"].mean()),
            "mean_oracle_reward_gap": float(detail_df[f"{p}_oracle_reward_gap"].mean()),
            "mean_productwise_oracle_reward_ratio_pct": float(ratios.mean()),
            "action_flip_rate_vs_baseline": (
                float((detail_df[f"{p}_discount"] != detail_df["noise_00pct_discount"]).mean())
                if "noise_00pct_discount" in detail_df else 0.0
            ),
            "mean_discount": float(detail_df[f"{p}_discount"].mean()),
        })

    OUT.mkdir(parents=True)
    detail_df.to_csv(OUT / "by_product.csv", index=False)
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(OUT / "summary.csv", index=False)

    metadata = {
        "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
        "checkpoint_updates": ckpt.get("updates"),
        "state_dim": 111,
        "action_dim": 8,
        "discount_actions": DISCOUNTS.tolist(),
        "train_months": [1, 2, 3, 4, 5, 6],
        "test_months": [9, 10],
        "noise_levels": LEVELS,
        "noise_seed": SEED,
        "reward_configuration": {"name": "sales", "alpha_margin": alpha, "beta_quantity": beta},
        "reward_configuration_note": (
            "Evaluation configuration from current code/reports; original checkpoint "
            "training weights are not independently verified from checkpoint metadata."
        ),
        "oracle_definition": "Best modeled reward across supported 5%-40% actions, excluding 0%.",
        "economic_evaluation": (
            "Policy action is selected from the perturbed state; economic outcomes are "
            "simulated using the original unperturbed state and product data."
        ),
        "limitations": [
            "Oracle is an offline counterfactual under the same learned simulator, not observed ground truth.",
            "Margin is a dataset-derived proxy, not full accounting profit.",
            "Synthetic Gaussian noise is not calibrated to real-world measurement error."
        ],
        "checkpoint_metadata": {
            "transfer_warmstart": ckpt.get("transfer_warmstart"),
            "transfer_source": ckpt.get("transfer_source"),
        },
    }
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Saved results: {OUT.relative_to(ROOT)}")
    print(summary_df.to_string(index=False))

if __name__ == "__main__":
    main()
