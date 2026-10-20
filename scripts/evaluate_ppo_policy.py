
import numpy as np
import pandas as pd

PATH = (
    "models/ppo_transfer/"
    "transfer_supported_8action_policy_evaluation.csv"
)

df = pd.read_csv(PATH)
test = df.loc[df["split"].eq("test")].copy()

configs = {
    "Sales-oriented (alpha=0.2, beta=0.8)": (0.2, 0.8),
    "Balanced (alpha=0.5, beta=0.5)": (0.5, 0.5),
    "Profit-oriented (alpha=0.8, beta=0.2)": (0.8, 0.2),
}

print("=" * 76)
print("HELD-OUT TEST POLICY EVALUATION")
print("=" * 76)
print(f"Test products: {len(test)}")
print(f"Unique source rows: {test['source_row'].nunique()}")
print(f"Missing values in test data: {int(test.isna().sum().sum())}")

# 1. BUSINESS OUTCOMES
print("\n" + "=" * 76)
print("1. BUSINESS OUTCOMES")
print("=" * 76)

outcomes = [
    ("Sales", "ppo_sales", "baseline_sales",
     "ppo_relative_delta_quantity"),
    ("Revenue", "ppo_revenue", None, None),
    ("Margin proxy", "ppo_margin", "baseline_margin",
     "ppo_relative_delta_margin"),
]

for label, ppo_col, base_col, relative_col in outcomes:
    ppo_mean = test[ppo_col].mean()
    print(f"\n{label}")
    print(f"  PPO mean: {ppo_mean:,.4f}")
    print(f"  PPO median: {test[ppo_col].median():,.4f}")

    if base_col is not None:
        base_mean = test[base_col].mean()
        print(f"  Baseline mean: {base_mean:,.4f}")
        print(
            "  Ratio-of-means change: "
            f"{(ppo_mean / base_mean - 1) * 100:.4f}%"
        )
        print(
            "  Mean per-product relative change: "
            f"{test[relative_col].mean() * 100:.4f}%"
        )
        print(
            "  Median per-product relative change: "
            f"{test[relative_col].median() * 100:.4f}%"
        )
        print(
            "  Products with positive relative change: "
            f"{(test[relative_col] > 0).mean() * 100:.2f}%"
        )

# 2. REWARD WEIGHTING ON FIXED PPO ACTIONS
print("\n" + "=" * 76)
print("2. REWARD WEIGHTING ON FIXED PPO ACTIONS")
print("=" * 76)
print(
    "Note: these are recomputed scores for existing actions, "
    "not separately trained policies."
)

for name, (alpha, beta) in configs.items():
    reward = (
        alpha * test["ppo_relative_delta_margin"]
        + beta * test["ppo_relative_delta_quantity"]
    )

    print(f"\n{name}")
    print(f"  Mean score: {reward.mean():.6f}")
    print(f"  Median score: {reward.median():.6f}")
    print(f"  Standard deviation: {reward.std():.6f}")

# 3. PPO VERSUS OFFLINE ORACLE
print("\n" + "=" * 76)
print("3. PPO VERSUS OFFLINE ORACLE")
print("=" * 76)

for metric in ["reward", "sales", "revenue", "margin"]:
    ppo_col = f"ppo_{metric}"
    oracle_col = f"oracle_{metric}"
    diff = test[ppo_col] - test[oracle_col]

    print(f"\n{metric.capitalize()}")
    print(f"  PPO mean: {test[ppo_col].mean():,.6f}")
    print(f"  Oracle mean: {test[oracle_col].mean():,.6f}")
    print(f"  Mean PPO-minus-oracle gap: {diff.mean():,.6f}")
    print(
        "  PPO higher than oracle: "
        f"{(diff > 1e-12).mean() * 100:.2f}%"
    )
    print(
        "  Exact ties: "
        f"{np.isclose(diff, 0).mean() * 100:.2f}%"
    )

# 4. DISCOUNT RECOMMENDATIONS
print("\n" + "=" * 76)
print("4. DISCOUNT RECOMMENDATIONS")
print("=" * 76)

print("\nRecommended discount distribution (%):")
print(
    (test["ppo_discount"].value_counts(normalize=True)
     .sort_index() * 100).round(3).to_string()
)

print("\nObserved discount distribution (%):")
print(
    (test["observed_discount"].value_counts(normalize=True)
     .sort_index() * 100).round(3).to_string()
)

changed = ~np.isclose(
    test["ppo_discount"], test["observed_discount"]
)

print(
    f"\nDiscount changed: {changed.sum()} / {len(test)} "
    f"({changed.mean() * 100:.2f}%)"
)
print(f"Mean recommended discount: {test['ppo_discount'].mean():.3f}%")
print(f"Mean observed discount: {test['observed_discount'].mean():.3f}%")
print(
    "Mean discount difference: "
    f"{(test['ppo_discount'] - test['observed_discount']).mean():.3f} "
    "percentage points"
)

# 5. OUTCOMES BY RECOMMENDED DISCOUNT
print("\n" + "=" * 76)
print("5. OUTCOMES BY RECOMMENDED DISCOUNT")
print("=" * 76)

action_summary = test.groupby("ppo_discount").agg(
    products=("source_row", "size"),
    mean_reward=("ppo_reward", "mean"),
    mean_sales=("ppo_sales", "mean"),
    mean_revenue=("ppo_revenue", "mean"),
    mean_margin_proxy=("ppo_margin", "mean"),
)

action_summary["share_pct"] = (
    action_summary["products"] / len(test) * 100
)

print(action_summary.round(4).to_string())

print("\nEvaluation completed. No files were modified.")
