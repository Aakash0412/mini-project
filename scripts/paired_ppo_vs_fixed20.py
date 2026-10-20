from pathlib import Path
import numpy as np
import pandas as pd
import json

root = Path(".")
report_dir = root / "reports/base_paper_reproduction"
pred_path = report_dir / "ppo_test_predictions.csv"
cf_path = report_dir / "all_action_counterfactuals.csv"
csv_out = report_dir / "ppo_vs_fixed20_paired_bootstrap.csv"
json_out = report_dir / "ppo_vs_fixed20_paired_bootstrap.json"

for p in (csv_out, json_out):
    if p.exists():
        raise FileExistsError(f"Refusing to overwrite: {p}")

pred = pd.read_csv(pred_path)
cf = pd.read_csv(cf_path)

ppo = pred[["test_position", "simulated_reward"]].rename(
    columns={"simulated_reward": "ppo_reward"}
)
fixed = cf.loc[
    cf["discount_percent"] == 20,
    ["test_position", "sales_weighted_reward"]
].rename(columns={"sales_weighted_reward": "fixed20_reward"})

paired = ppo.merge(fixed, on="test_position", validate="one_to_one")
paired = paired.dropna(subset=["ppo_reward", "fixed20_reward"]).copy()

if len(paired) < 100:
    raise ValueError(f"Unexpectedly few paired rows: {len(paired)}")

paired["reward_difference"] = (
    paired["ppo_reward"] - paired["fixed20_reward"]
)

differences = paired["reward_difference"].to_numpy()
rng = np.random.default_rng(42)
n = len(differences)
bootstrap_means = np.empty(10000)

for i in range(len(bootstrap_means)):
    sample = rng.choice(differences, size=n, replace=True)
    bootstrap_means[i] = sample.mean()

ci_low, ci_high = np.quantile(bootstrap_means, [0.025, 0.975])

summary = {
    "comparison": "Existing PPO versus fixed 20% discount",
    "paired_test_products": int(n),
    "mean_ppo_reward": float(paired["ppo_reward"].mean()),
    "mean_fixed20_reward": float(paired["fixed20_reward"].mean()),
    "mean_paired_reward_difference": float(differences.mean()),
    "median_paired_reward_difference": float(np.median(differences)),
    "bootstrap_95_percent_ci": [float(ci_low), float(ci_high)],
    "ppo_win_share_percent": float((differences > 0).mean() * 100),
    "tie_share_percent": float((differences == 0).mean() * 100),
    "fixed20_win_share_percent": float((differences < 0).mean() * 100),
    "bootstrap_resamples": len(bootstrap_means),
    "bootstrap_seed": 42,
    "limitations": [
        "Paired bootstrap quantifies uncertainty across held-out products, conditional on this simulator.",
        "It does not account for sales-predictor model error.",
        "It does not establish real-world causal impact or statistical independence between products."
    ]
}

paired.to_csv(csv_out, index=False)
json_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

print(json.dumps(summary, indent=2))
print("\nCreated:")
print(csv_out)
print(json_out)
