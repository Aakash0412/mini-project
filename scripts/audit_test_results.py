
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

CSV_PATH = (
    ROOT
    / "models"
    / "ppo_transfer"
    / "transfer_supported_8action_policy_evaluation.csv"
)

df = pd.read_csv(CSV_PATH)

print("Available splits:")
print(df["split"].value_counts().to_string())

test = df.loc[df["split"] == "test"].copy()

if test.empty:
    raise ValueError("No test rows found in the evaluation CSV.")

numeric_cols = [
    "ppo_reward",
    "oracle_reward",
    "ppo_sales",
    "oracle_sales",
    "ppo_revenue",
    "oracle_revenue",
    "ppo_margin",
    "oracle_margin",
    "baseline_sales",
    "baseline_margin",
]

for col in numeric_cols:
    test[col] = pd.to_numeric(test[col], errors="raise")

if not np.isfinite(test[numeric_cols].to_numpy()).all():
    raise ValueError("Non-finite numerical values found in test rows.")

ppo_reward = test["ppo_reward"].mean()
oracle_reward = test["oracle_reward"].mean()

print("\n========== HELD-OUT TEST RESULTS ==========")
print(f"Test products: {len(test)}")
print(f"PPO mean reward: {ppo_reward:.6f}")
print(f"Oracle mean reward: {oracle_reward:.6f}")
print(f"Mean oracle gap: {(oracle_reward - ppo_reward):.6f}")

if abs(oracle_reward) > 1e-8:
    print(f"PPO/oracle reward ratio: {100 * ppo_reward / oracle_reward:.2f}%")
else:
    print("PPO/oracle reward ratio: undefined (oracle mean near zero)")

print(
    "Exact oracle action match: "
    f"{100 * (test['ppo_action'] == test['oracle_action']).mean():.2f}%"
)

print("\nPPO discount distribution:")
print(
    test["ppo_discount"]
    .value_counts(normalize=True)
    .sort_index()
    .mul(100)
    .round(2)
    .to_string()
)

print("\nMean changes relative to baseline:")
print(f"Quantity change: {test['ppo_delta_quantity'].mean():.4f}")
print(f"Margin change: {test['ppo_delta_margin'].mean():.4f}")

print("\nRelative changes:")
print(
    "Quantity: "
    f"{100 * test['ppo_relative_delta_quantity'].mean():.4f}% "
    "(mean of per-product relative changes)"
)
print(
    "Margin: "
    f"{100 * test['ppo_relative_delta_margin'].mean():.4f}% "
    "(mean of per-product relative changes)"
)

print("\nPositive-change proportions:")
print(
    "Products with positive quantity change: "
    f"{100 * (test['ppo_delta_quantity'] > 0).mean():.2f}%"
)
print(
    "Products with positive margin change: "
    f"{100 * (test['ppo_delta_margin'] > 0).mean():.2f}%"
)
