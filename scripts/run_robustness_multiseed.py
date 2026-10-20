from pathlib import Path
import importlib.util
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path.cwd()
SOURCE = ROOT / "scripts" / "run_robustness.py"
OUTPUT = ROOT / "reports" / "robustness" / "multiseed_runs"

sys.path.insert(0, str(ROOT))

spec = importlib.util.spec_from_file_location(
    "original_robustness", SOURCE
)
robustness = importlib.util.module_from_spec(spec)
spec.loader.exec_module(robustness)

seeds = list(range(101, 111))
all_results = []

print("MULTI-SEED ROBUSTNESS EVALUATION")
print("Independent noise seeds:", seeds)
print("Original results will not be overwritten.")
print("Each run evaluates the existing frozen PPO checkpoint.")

for seed in seeds:
    run_dir = OUTPUT / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    robustness.SEED = seed
    robustness.OUTPUT_DIR = run_dir
    robustness.RESULTS_PATH = run_dir / "robustness_results.csv"
    robustness.SUMMARY_PATH = run_dir / "robustness_summary.json"

    print(f"\n\n######## SEED {seed} ########", flush=True)
    robustness.main()

    result = pd.read_csv(robustness.RESULTS_PATH)
    result["seed"] = seed
    all_results.append(result)

    print(f"Saved separate results for seed {seed}.", flush=True)

runs = pd.concat(all_results, ignore_index=True)

per_seed_path = ROOT / "reports" / "robustness" / "multiseed_all_runs.csv"
runs.to_csv(per_seed_path, index=False)

metrics = [
    "mean_reward",
    "reward_retention_percent",
    "reward_change_percent",
    "mean_estimated_sales",
    "mean_estimated_revenue",
    "mean_estimated_margin",
]

summary_rows = []
t_critical_9df = 2.262  # two-sided 95% CI for 10 independent runs

for noise, group in runs.groupby("noise_fraction", sort=True):
    row = {
        "noise_fraction": float(noise),
        "independent_seeds": int(group["seed"].nunique()),
        "evaluated_products_per_seed": int(group["evaluated_products"].min()),
    }

    for metric in metrics:
        values = group[metric].to_numpy(dtype=float)
        mean = float(values.mean())
        sd = float(values.std(ddof=1))
        half_width = t_critical_9df * sd / np.sqrt(len(values))

        row[f"{metric}_mean"] = mean
        row[f"{metric}_between_seed_sd"] = sd
        row[f"{metric}_ci95_low"] = mean - half_width
        row[f"{metric}_ci95_high"] = mean + half_width

    summary_rows.append(row)

summary = pd.DataFrame(summary_rows)
summary_path = ROOT / "reports" / "robustness" / "multiseed_aggregate.csv"
summary.to_csv(summary_path, index=False)

metadata = {
    "seeds": seeds,
    "noise_levels": robustness.NOISE_LEVELS,
    "checkpoint": str(robustness.CHECKPOINT),
    "policy_retrained": False,
    "existing_results_overwritten": False,
    "per_seed_results": str(per_seed_path),
    "aggregate_results": str(summary_path),
    "confidence_interval": "t-based 95% CI across 10 noise seeds",
    "interpretation": (
        "Intervals quantify variation across noise realizations for "
        "this fixed policy and fixed test set; they do not represent "
        "uncertainty across independently trained policies or products."
    ),
}

metadata_path = ROOT / "reports" / "robustness" / "multiseed_metadata.json"
metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

print("\n\n===== MULTI-SEED AGGREGATE RESULTS =====")
print(summary[
    [
        "noise_fraction",
        "independent_seeds",
        "mean_reward_mean",
        "mean_reward_between_seed_sd",
        "mean_reward_ci95_low",
        "mean_reward_ci95_high",
        "reward_retention_percent_mean",
    ]
].round(6).to_string(index=False))

print("\nSaved:")
print(per_seed_path)
print(summary_path)
print(metadata_path)
