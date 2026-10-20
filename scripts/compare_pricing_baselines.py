from pathlib import Path
import json
import pandas as pd

root = Path(".")
report_dir = root / "reports/base_paper_reproduction"
counter_path = report_dir / "all_action_counterfactuals.csv"
policy_path = report_dir / "ppo_test_predictions.csv"

script_out = root / "scripts/compare_pricing_baselines.py"
csv_out = report_dir / "pricing_baseline_comparison.csv"
json_out = report_dir / "pricing_baseline_comparison.json"

for path in (csv_out, json_out):
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")

cf = pd.read_csv(counter_path)
ppo = pd.read_csv(policy_path)

required_cf = {
    "test_position", "discount_percent", "sales_weighted_reward",
    "estimated_sales", "estimated_revenue", "estimated_margin_proxy"
}
required_ppo = {
    "test_position", "observed_discount_percent",
    "recommended_discount_percent", "simulated_reward",
    "estimated_sales", "estimated_revenue", "estimated_margin_proxy"
}
if required_cf - set(cf.columns):
    raise ValueError(f"Counterfactual file missing: {required_cf - set(cf.columns)}")
if required_ppo - set(ppo.columns):
    raise ValueError(f"Policy file missing: {required_ppo - set(ppo.columns)}")

if ppo["test_position"].duplicated().any():
    raise ValueError("Duplicate PPO test positions.")
if cf.duplicated(["test_position", "discount_percent"]).any():
    raise ValueError("Duplicate product/discount combinations.")

positions = set(ppo["test_position"])
if positions != set(cf["test_position"]):
    raise ValueError("PPO and counterfactual test-product sets differ.")

rows = []

def add_result(name, frame, reward_col, discount_col):
    rows.append({
        "strategy": name,
        "test_products": int(frame["test_position"].nunique()),
        "mean_reward": float(frame[reward_col].mean()),
        "median_reward": float(frame[reward_col].median()),
        "mean_estimated_sales": float(frame["estimated_sales"].mean()),
        "mean_estimated_revenue": float(frame["estimated_revenue"].mean()),
        "mean_margin_proxy": float(frame["estimated_margin_proxy"].mean()),
        "mean_discount_percent": float(frame[discount_col].mean()),
    })

# Fixed-discount strategies: identical products and simulator.
for discount in [10, 20, 30]:
    part = cf[cf["discount_percent"] == discount].copy()
    if part["test_position"].nunique() != len(positions):
        raise ValueError(f"Discount {discount}% lacks some test products.")
    add_result(
        f"Fixed {discount}%",
        part,
        "sales_weighted_reward",
        "discount_percent",
    )

# Existing PPO policy: use its selected action's stored outcome.
ppo_part = ppo.rename(columns={
    "simulated_reward": "baseline_reward",
    "recommended_discount_percent": "baseline_discount",
})
add_result(
    "Existing PPO",
    ppo_part,
    "baseline_reward",
    "baseline_discount",
)

# Observed-discount reference: use the closest evaluated grid action.
# This is explicitly an approximation when observed discounts are off-grid.
observed = ppo[["test_position", "observed_discount_percent"]].copy()
observed["observed_discount_percent"] = pd.to_numeric(
    observed["observed_discount_percent"], errors="coerce"
)
nearest = cf.merge(observed, on="test_position", validate="many_to_one")
nearest["distance"] = (
    nearest["discount_percent"] - nearest["observed_discount_percent"]
).abs()
nearest = nearest.sort_values(
    ["test_position", "distance", "discount_percent"]
).drop_duplicates("test_position")

add_result(
    "Observed discount (nearest evaluated action)",
    nearest,
    "sales_weighted_reward",
    "discount_percent",
)

result = pd.DataFrame(rows)
result["reward_rank"] = result["mean_reward"].rank(
    ascending=False, method="min"
).astype(int)
result = result.sort_values("reward_rank")

result.to_csv(csv_out, index=False)
json_out.write_text(
    json.dumps({
        "evaluation": "Offline model-based baseline comparison",
        "test_products": int(len(positions)),
        "reward_weights": {"alpha_margin": 0.2, "beta_quantity": 0.8},
        "observed_discount_note": (
            "Uses the nearest of the nine evaluated discount actions; "
            "not necessarily the exact observed discount."
        ),
        "limitations": [
            "All outcomes are simulated using the existing frozen sales predictor.",
            "Margin is a proxy, not verified accounting profit.",
            "These are not real-world causal effects.",
            "Existing PPO and fixed strategies use the same test product set.",
        ],
        "results": result.to_dict(orient="records"),
    }, indent=2),
    encoding="utf-8",
)

print("Baseline comparison complete.")
print(result.to_string(index=False))
print("\nCreated:")
print(csv_out)
print(json_out)

