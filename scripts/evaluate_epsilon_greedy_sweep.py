"""Epsilon-greedy exploration analysis on the frozen 8-action PPO policy.

For each of 7,139 test products and each epsilon in the sweep, simulate
an epsilon-greedy policy over the frozen actor's action distribution:

    with probability epsilon: choose a uniformly random action from {0..7}
    with probability 1-epsilon: choose the actor's argmax

Then compute:
    - mean simulated reward (under the frozen predictor)
    - exploitation rate (fraction of products where action == greedy)
    - action-change rate vs epsilon=0 baseline

The actor is NOT retrained. This is a policy wrapper, not a training run.
"""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch

import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.ppo_agent import PPOAgent
from src.environment.ppo_dataset import load_ppo_dataset
from src.environment.pricing_economics import calculate_pricing_outcome

DISCOUNTS = np.array([5, 10, 15, 20, 25, 30, 35, 40], dtype=np.float64)
SUPPORTED = set(int(x) for x in DISCOUNTS)
CHECKPOINT = ROOT / "models/ppo_transfer/ppo_transfer_supported_8action_1000_updates.pth"
OUTPUT_DIR = ROOT / "reports/epsilon_greedy_sweep"
STATE_DIM = 111
ACTION_DIM = 8

EPSILONS = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0]
SEEDS = [0, 1, 2, 3, 4]   # repeat each epsilon with different seeds to average randomness


def load_actor() -> torch.nn.Module:
    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    agent = PPOAgent(state_dim=STATE_DIM, action_dim=ACTION_DIM,
                     actor_lr=8e-4, critic_lr=8e-4)
    agent.actor.load_state_dict(ckpt["actor_state_dict"])
    agent.actor.eval()
    return agent.actor


def outcome_reward(state, ctx, discount, predictor):
    growth = predictor.predict(
        torch.as_tensor(state, dtype=torch.float32).unsqueeze(0),
        torch.as_tensor([[float(discount)]], dtype=torch.float32),
    ).item()
    out = calculate_pricing_outcome(
        observed_price=ctx.observed_price,
        observed_discount=ctx.observed_discount,
        last_monthly_sales=ctx.last_monthly_sales,
        predicted_growth_rate=growth,
        candidate_discount=float(discount),
        gross_margin_rate=ctx.gross_margin_rate,
    )
    if out.estimated_margin is None:
        return float("nan")
    baseline = calculate_pricing_outcome(
        observed_price=ctx.observed_price,
        observed_discount=ctx.observed_discount,
        last_monthly_sales=ctx.last_monthly_sales,
        predicted_growth_rate=0.0,
        candidate_discount=ctx.observed_discount,
        gross_margin_rate=ctx.gross_margin_rate,
    )
    if baseline.estimated_margin is None:
        return float("nan")
    dq = (out.estimated_sales - ctx.last_monthly_sales) / max(abs(ctx.last_monthly_sales), 1e-8)
    dm = (out.estimated_margin - baseline.estimated_margin) / max(abs(baseline.estimated_margin), 1e-8)
    return 0.5 * dm + 0.5 * dq


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    actor = load_actor()
    from src.models.sales_predictor import load_frozen_predictor
    predictor = load_frozen_predictor("models/sales_predictor", device=torch.device("cpu"))
    for p in predictor.parameters():
        p.requires_grad_(False)

    dataset = load_ppo_dataset(
        state_path=ROOT / "data/processed/state/state.npy",
        index_path=ROOT / "data/processed/state/state_row_indices.npy",
        products_path=ROOT / "data/processed/products_clean.parquet",
    )
    df = pd.read_parquet(ROOT / "data/processed/products_clean.parquet")
    aligned = df.iloc[dataset.source_row_indices].reset_index(drop=True)
    test_mask = aligned["MonthNum"].isin([9, 10]).to_numpy()
    states = dataset.states[test_mask]
    contexts = [c for c, k in zip(dataset.products, test_mask) if k]
    n = len(states)

    # Precompute argmax action and per-action mean reward for each product
    with torch.no_grad():
        logits = actor(torch.as_tensor(states, dtype=torch.float32))
        greedy_action = logits.argmax(dim=1).cpu().numpy()

    reward_matrix = np.full((n, ACTION_DIM), np.nan, dtype=np.float64)
    for i in range(n):
        for a, d in enumerate(DISCOUNTS):
            reward_matrix[i, a] = outcome_reward(states[i], contexts[i], d, predictor)

    # Oracle = argmax of mean reward per product
    oracle_action = np.nanargmax(reward_matrix, axis=1)

    rows = []
    for eps in EPSILONS:
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            # Epsilon-greedy: with prob eps random; else greedy
            rand_mask = rng.random(n) < eps
            random_actions = rng.integers(0, ACTION_DIM, size=n)
            actions = np.where(rand_mask, random_actions, greedy_action)
            chosen_rewards = reward_matrix[np.arange(n), actions]
            mean_reward = float(np.nanmean(chosen_rewards))
            exploit_rate = float((actions == oracle_action).mean())
            rows.append({
                "epsilon": eps,
                "seed": seed,
                "mean_simulated_reward": mean_reward,
                "exploitation_rate": exploit_rate,
                "n_products": n,
            })

    df_out = pd.DataFrame(rows)
    agg = df_out.groupby("epsilon").agg(
        mean_reward_mean=("mean_simulated_reward", "mean"),
        mean_reward_std=("mean_simulated_reward", "std"),
        exploitation_rate_mean=("exploitation_rate", "mean"),
        exploitation_rate_std=("exploitation_rate", "std"),
    ).reset_index()

    df_out.to_csv(OUTPUT_DIR / "epsilon_greedy_raw.csv", index=False)
    agg.to_csv(OUTPUT_DIR / "epsilon_greedy_summary.csv", index=False)

    summary = {
        "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
        "n_products": n,
        "epsilons": EPSILONS,
        "seeds": SEEDS,
        "noise_protocol": "uniform random action over 8 bins when rand < epsilon; argmax otherwise",
        "oracle": "argmax over actions of mean simulated reward per product",
        "greedy_baseline_exploitation_rate": float((greedy_action == oracle_action).mean()),
        "note": "Actor is frozen; this is a policy wrapper, not retraining",
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(agg.to_string(index=False))
    print(f"\nGreedy baseline (epsilon=0) exploitation rate: "
          f"{summary['greedy_baseline_exploitation_rate']*100:.2f}%")
    print(f"Outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()