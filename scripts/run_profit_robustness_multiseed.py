from pathlib import Path
import importlib.util
import json
import sys
import numpy as np
import pandas as pd

ROOT = Path.cwd()
SOURCE = ROOT / "scripts" / "run_robustness.py"
CHECKPOINT = ROOT / "models" / "ppo_reward_experiments" / "profit_entropy_005" / "ppo_transfer_supported_8action_200_updates.pth"
OUTPUT = ROOT / "reports" / "robustness" / "profit_entropy_005_multiseed"

if not SOURCE.is_file():
    raise FileNotFoundError(SOURCE)
if not CHECKPOINT.is_file():
    raise FileNotFoundError(CHECKPOINT)
if OUTPUT.exists() and any(OUTPUT.iterdir()):
    raise FileExistsError(f"Refusing to overwrite existing results: {OUTPUT}")

spec = importlib.util.spec_from_file_location("robustness_profit", SOURCE)
robustness = importlib.util.module_from_spec(spec)
spec.loader.exec_module(robustness)

from src.environment.reward import get_reward_config
from src.environment.pricing_environment import DynamicPricingEnvironment

robustness.CHECKPOINT = CHECKPOINT
robustness.DynamicPricingEnvironment = lambda: DynamicPricingEnvironment(
    reward_config=get_reward_config("profit")
)

seeds = list(range(101, 111))
all_results = []
OUTPUT.mkdir(parents=True, exist_ok=True)

for seed in seeds:
    run_dir = OUTPUT / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=False)

    robustness.SEED = seed
    robustness.OUTPUT_DIR = run_dir
    robustness.RESULTS_PATH = run_dir / "robustness_results.csv"
    robustness.SUMMARY_PATH = run_dir / "robustness_summary.json"

    print(f"\n######## PROFIT ROBUSTNESS SEED {seed} ########", flush=True)
    robustness.main()

    result = pd.read_csv(robustness.RESULTS_PATH)
    result["seed"] = seed
    all_results.append(result)

runs = pd.concat(all_results, ignore_index=True)
per_seed_path = OUTPUT / "all_seed_results.csv"
runs.to_csv(per_seed_path, index=False)

metrics = [
    c for c in [
        "mean_reward", "reward_std", "reward_retention_percent",
        "mean_predicted_growth", "mean_selected_discount"
    ] if c in runs.columns
]
summary_rows = []
for noise, group in runs.groupby("noise_fraction", sort=True):
    row = {"noise_fraction": float(noise), "seeds": int(group["seed"].nunique())}
    for metric in metrics:
        values = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy()
        row[f"{metric}_mean"] = float(values.mean()) if len(values) else None
        row[f"{metric}_std_across_seeds"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        if len(values) > 1:
            from scipy.stats import t
            half = float(t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values)))
            row[f"{metric}_ci95_low"] = float(values.mean() - half)
            row[f"{metric}_ci95_high"] = float(values.mean() + half)
    summary_rows.append(row)

summary_path = OUTPUT / "aggregate_summary.csv"
pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

metadata = {
    "checkpoint": str(CHECKPOINT),
    "reward_config": "profit",
    "reward_alpha_margin": 0.8,
    "reward_beta_quantity": 0.2,
    "seeds": seeds,
    "noise_levels": [0.0, 0.05, 0.10, 0.15],
    "policy_retrained": False,
    "experiment_type": "frozen-policy sales-growth prediction-noise robustness",
    "per_seed_results": str(per_seed_path),
    "aggregate_results": str(summary_path),
    "confidence_interval": "t-based 95% CI across noise seeds where applicable"
}
(OUTPUT / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

print("\nRobustness experiment complete.")
print(f"Per-seed results: {per_seed_path}")
print(f"Aggregate summary: {summary_path}")
print(f"Metadata: {OUTPUT / 'metadata.json'}")
