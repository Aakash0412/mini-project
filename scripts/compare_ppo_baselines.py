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
OUT = ROOT / "reports/evaluation_audit/final_baseline_comparison"
DISCOUNTS = np.array([5, 10, 15, 20, 25, 30, 35, 40])
EPS = 1e-8
BATCH_SIZE = 1024

def main():
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(CHECKPOINT)
    if not (ROOT / "data/processed/products_clean.parquet").is_file():
        raise FileNotFoundError("Processed products parquet is missing.")
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite: {OUT}")

    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    if checkpoint.get("state_dim") != 111 or checkpoint.get("action_dim") != 8:
        raise ValueError("Checkpoint does not match the expected 111-state/8-action policy.")

    dataset = load_ppo_dataset()
    df = pd.read_parquet(ROOT / "data/processed/products_clean.parquet")
    source_indices = np.asarray(dataset.source_row_indices, dtype=int)
    aligned = df.iloc[source_indices].reset_index(drop=True)
    states = np.asarray(dataset.states, dtype=np.float32)

    if states.shape != (len(aligned), 111) or not np.isfinite(states).all():
        raise ValueError("State shape, alignment, or finite-value validation failed.")

    test_positions = np.flatnonzero(aligned["MonthNum"].between(9, 10).to_numpy())
    train_positions = np.flatnonzero(aligned["MonthNum"].between(1, 6).to_numpy())
    if len(test_positions) != 7139:
        raise ValueError(f"Expected 7,139 held-out rows; found {len(test_positions)}.")

    agent = PPOAgent(state_dim=111, action_dim=8)
    agent.actor.load_state_dict(checkpoint["actor_state_dict"])
    agent.actor.eval()

    test_states = states[test_positions]
    ppo_actions = []
    with torch.no_grad():
        for start in range(0, len(test_states), BATCH_SIZE):
            batch = torch.as_tensor(test_states[start:start + BATCH_SIZE])
            ppo_actions.extend(torch.argmax(agent.actor(batch), dim=-1).numpy().tolist())
    ppo_actions = np.asarray(ppo_actions, dtype=int)

    # Current evaluation configuration: alpha=0.2 margin, beta=0.8 quantity.
    reward_config = get_reward_config("sales")
    alpha, beta = reward_config.alpha, reward_config.beta
    env = DynamicPricingEnvironment(reward_config=reward_config)

    policies = [
        "PPO",
        "Observed discount (nearest grid)",
        "Fixed 10%",
        "Fixed 20%",
        "Fixed 30%",
        "Offline oracle (5%-40%)",
    ]
    rows = []

    for j, position in enumerate(test_positions):
        product = dataset.products[position]
        state = states[position]
        env.reset(state=state, product=product)

        baseline_sales = float(product.last_monthly_sales)
        baseline_margin = env.observed_outcome.estimated_margin
        if baseline_margin is None or not np.isfinite(baseline_margin):
            continue

        def metrics(env_action):
            outcome = env.predict_action(int(env_action))
            if outcome.estimated_margin is None:
                return None
            sales = float(outcome.estimated_sales)
            margin = float(outcome.estimated_margin)
            revenue = float(outcome.estimated_revenue)
            reward = (
                alpha * (margin - baseline_margin) / max(abs(baseline_margin), EPS)
                + beta * (sales - baseline_sales) / max(abs(baseline_sales), EPS)
            )
            return {
                "discount": float(env.action_to_discount(int(env_action))),
                "reward": float(reward),
                "estimated_sales": sales,
                "estimated_revenue": revenue,
                "estimated_margin_proxy": margin,
            }

        # PPO actions map from 0..7 to environment actions 1..8.
        ppo_result = metrics(int(ppo_actions[j]) + 1)

        observed_discount = float(product.observed_discount)
        observed_grid = int(np.argmin(np.abs(DISCOUNTS - observed_discount))) + 1
        observed_result = metrics(observed_grid)
        fixed10 = metrics(2)
        fixed20 = metrics(4)
        fixed30 = metrics(6)

        candidates = [(d, metrics(i + 1)) for i, d in enumerate(DISCOUNTS)]
        candidates = [(d, m) for d, m in candidates if m is not None and np.isfinite(m["reward"])]
        if not candidates:
            continue
        oracle_discount, oracle_result = max(candidates, key=lambda pair: pair[1]["reward"])

        results = {
            "PPO": ppo_result,
            "Observed discount (nearest grid)": observed_result,
            "Fixed 10%": fixed10,
            "Fixed 20%": fixed20,
            "Fixed 30%": fixed30,
            "Offline oracle (5%-40%)": oracle_result,
        }

        record = {
            "source_row": int(source_indices[position]),
            "month_num": int(aligned.iloc[position]["MonthNum"]),
            "observed_discount": observed_discount,
            "oracle_discount": float(oracle_discount),
            "baseline_sales": baseline_sales,
            "baseline_margin_proxy": float(baseline_margin),
        }
        for name, result in results.items():
            prefix = name.lower().replace(" ", "_").replace("%", "pct").replace("(", "").replace(")", "").replace("-", "_")
            for key, value in result.items():
                record[f"{prefix}_{key}"] = value
        rows.append(record)

    if not rows:
        raise RuntimeError("No eligible test products could be evaluated.")

    detail = pd.DataFrame(rows)
    policy_specs = [
        ("PPO", "ppo"),
        ("Observed discount (nearest grid)", "observed_discount_nearest_grid"),
        ("Fixed 10%", "fixed_10pct"),
        ("Fixed 20%", "fixed_20pct"),
        ("Fixed 30%", "fixed_30pct"),
        ("Offline oracle (5%-40%)", "offline_oracle_5pct_40pct"),
    ]
    summaries = []
    oracle_mean_reward = detail["offline_oracle_5pct_40pct_reward"].mean()

    for name, prefix in policy_specs:
        rewards = detail[f"{prefix}_reward"]
        summaries.append({
            "policy": name,
            "evaluated_products": len(detail),
            "mean_reward": float(rewards.mean()),
            "median_reward": float(rewards.median()),
            "mean_estimated_sales": float(detail[f"{prefix}_estimated_sales"].mean()),
            "mean_estimated_revenue": float(detail[f"{prefix}_estimated_revenue"].mean()),
            "mean_estimated_margin_proxy": float(detail[f"{prefix}_estimated_margin_proxy"].mean()),
            "mean_discount_pct": float(detail[f"{prefix}_discount"].mean()),
            "reward_ratio_to_oracle_mean_pct": (
                float(100 * rewards.mean() / oracle_mean_reward)
                if abs(oracle_mean_reward) > EPS else np.nan
            ),
            "mean_oracle_reward_gap": float((detail["offline_oracle_5pct_40pct_reward"] - rewards).mean()),
        })

    OUT.mkdir(parents=True)
    detail.to_csv(OUT / "by_product.csv", index=False)
    summary = pd.DataFrame(summaries)
    summary.to_csv(OUT / "summary.csv", index=False)

    metadata = {
        "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
        "checkpoint_updates": checkpoint.get("updates"),
        "state_dim": 111,
        "action_dim": 8,
        "test_months": [9, 10],
        "training_months_for_context": [1, 2, 3, 4, 5, 6],
        "test_products": int(len(detail)),
        "reward_config": {"name": "sales", "alpha_margin": alpha, "beta_quantity": beta},
        "training_reward_weights_verified": False,
        "policies": policies,
        "observed_discount_policy": "Nearest supported action in [5,10,15,20,25,30,35,40] percent.",
        "oracle": "Highest modeled reward among the eight supported discounts; offline comparator, not deployable ground truth.",
        "economics_note": "Revenue and margin are simulator estimates; margin is a proxy, not full accounting profit.",
        "split_note": "All policies evaluated on the same chronological held-out products.",
    }
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Saved results: {OUT.relative_to(ROOT)}")
    print(summary.to_string(index=False))

if __name__ == "__main__":
    main()
