"""
Read-only paired PPO vs baseline/oracle audit.

Reads the existing evaluation CSV only. Does not load models, run inference,
train, or modify checkpoints/input data. Outputs go into:
reports/evaluation_audit/paired_baseline_audit/

Run from the project root:
  .\.venv\Scripts\python.exe scripts\audit_ppo_paired_comparisons.py
"""
from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "models" / "ppo_transfer" / "transfer_supported_8action_policy_evaluation.csv"
OUT = ROOT / "reports" / "evaluation_audit" / "paired_baseline_audit"
SEED = 42
BOOTSTRAP_REPS = 5000

def bootstrap_mean_ci(values, rng, reps=BOOTSTRAP_REPS):
    x = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
    if len(x) == 0:
        return (np.nan, np.nan, np.nan, 0)
    mean = float(np.mean(x))
    if len(x) == 1:
        return (mean, np.nan, np.nan, 1)
    # Batch bootstrap to keep memory reasonable.
    boot_means = np.empty(reps, dtype=float)
    batch = 250
    for start in range(0, reps, batch):
        n = min(batch, reps - start)
        idx = rng.integers(0, len(x), size=(n, len(x)))
        boot_means[start:start+n] = x[idx].mean(axis=1)
    low, high = np.quantile(boot_means, [0.025, 0.975])
    return mean, float(low), float(high), int(len(x))

def find_col(df, candidates, required=True):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise ValueError(f"Required column not found. Tried {candidates}. Available: {list(df.columns)}")
    return None

def main():
    if not INPUT.exists():
        raise FileNotFoundError(f"Evaluation CSV not found: {INPUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(INPUT, low_memory=False)

    split = find_col(df, ["split", "dataset_split"])
    discount = find_col(df, ["ppo_discount", "discount"], required=False)
    ppo_reward = find_col(df, ["ppo_reward", "reward"], required=False)
    oracle_reward = find_col(df, ["oracle_reward"], required=False)

    pairs = [
        ("ppo_sales", ["ppo_sales", "sales"], "baseline_sales", ["baseline_sales"], "sales"),
        ("ppo_margin", ["ppo_margin", "margin"], "baseline_margin", ["baseline_margin"], "margin"),
        ("ppo_sales", ["ppo_sales", "sales"], "oracle_sales", ["oracle_sales"], "sales"),
        ("ppo_margin", ["ppo_margin", "margin"], "oracle_margin", ["oracle_margin"], "margin"),
        ("ppo_revenue", ["ppo_revenue", "revenue"], "oracle_revenue", ["oracle_revenue"], "revenue"),
    ]
    comparison_specs = []
    for left_label, left_candidates, right_label, right_candidates, metric in pairs:
        left = find_col(df, left_candidates, required=False)
        right = find_col(df, right_candidates, required=False)
        if left and right:
            comparison_specs.append((left, right, metric, f"{left_label}_minus_{right_label}"))
    if ppo_reward and oracle_reward:
        comparison_specs.append((ppo_reward, oracle_reward, "reward", "ppo_reward_minus_oracle_reward"))

    rng = np.random.default_rng(SEED)
    rows = []
    for split_value, sub in df.groupby(split, dropna=False):
        for left, right, metric, delta_name in comparison_specs:
            a = pd.to_numeric(sub[left], errors="coerce")
            b = pd.to_numeric(sub[right], errors="coerce")
            delta = a - b
            mean, low, high, n = bootstrap_mean_ci(delta, rng)
            rows.append({
                "split": split_value,
                "comparison": delta_name,
                "metric": metric,
                "left_column": left,
                "right_column": right,
                "n_paired": n,
                "mean_left": float(a.mean()) if a.notna().any() else np.nan,
                "mean_right": float(b.mean()) if b.notna().any() else np.nan,
                "mean_paired_difference": mean,
                "bootstrap_95ci_low": low,
                "bootstrap_95ci_high": high,
                "fraction_left_greater": float((delta.dropna() > 0).mean()) if delta.notna().any() else np.nan,
                "fraction_equal": float((delta.dropna() == 0).mean()) if delta.notna().any() else np.nan,
            })
    overall = pd.DataFrame(rows)
    overall.to_csv(OUT / "paired_comparison_summary.csv", index=False)

    by_discount_rows = []
    if discount:
        for (split_value, discount_value), sub in df.groupby([split, discount], dropna=False):
            for left, right, metric, delta_name in comparison_specs:
                a = pd.to_numeric(sub[left], errors="coerce")
                b = pd.to_numeric(sub[right], errors="coerce")
                delta = a - b
                mean, low, high, n = bootstrap_mean_ci(delta, rng)
                by_discount_rows.append({
                    "split": split_value,
                    "ppo_discount": discount_value,
                    "comparison": delta_name,
                    "metric": metric,
                    "n_paired": n,
                    "mean_left": float(a.mean()) if a.notna().any() else np.nan,
                    "mean_right": float(b.mean()) if b.notna().any() else np.nan,
                    "mean_paired_difference": mean,
                    "bootstrap_95ci_low": low,
                    "bootstrap_95ci_high": high,
                    "fraction_left_greater": float((delta.dropna() > 0).mean()) if delta.notna().any() else np.nan,
                })
    by_discount = pd.DataFrame(by_discount_rows)
    if not by_discount.empty:
        by_discount.to_csv(OUT / "paired_comparisons_by_discount.csv", index=False)

    # Source-row mapping is only reported as present/absent; it is not assumed to be a product ID.
    source_row = find_col(df, ["source_row", "row_id", "source_index"], required=False)
    report = {
        "purpose": "Paired descriptive comparisons using outcomes already stored per evaluation row",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_csv": str(INPUT.relative_to(ROOT)),
        "rows": int(len(df)),
        "split_counts": {str(k): int(v) for k, v in df[split].value_counts(dropna=False).items()},
        "source_row_column_present": source_row,
        "bootstrap": {"method": "row bootstrap of paired differences", "repetitions": BOOTSTRAP_REPS, "seed": SEED, "interval": "percentile 95%"},
        "comparisons": [x[3] for x in comparison_specs],
        "outputs": ["paired_comparison_summary.csv", "paired_comparisons_by_discount.csv", "paired_baseline_audit.md", "paired_baseline_audit.json"],
        "safety": {"model_loaded": False, "inference_performed": False, "training_performed": False, "checkpoint_modified": False, "input_csv_modified": False},
        "limitations": [
            "Paired differences compare stored values for the same evaluation rows; they do not prove counterfactual causal effects.",
            "Bootstrap intervals quantify row-level sampling variability and may be optimistic if observations are clustered by product.",
            "The oracle is an offline comparator and should not be treated as a deployable policy without checking how it was constructed.",
            "Interpret sales, revenue, margin, and reward according to the simulator's definitions; do not equate them with observed real-world profit."
        ],
    }
    (OUT / "paired_baseline_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    md = [
        "# Paired PPO Comparison Audit", "",
        "## Scope",
        "- Read-only analysis of stored per-row PPO, baseline, and oracle values.",
        "- No inference, training, checkpoint modifications, or input CSV changes.",
        "- Differences are paired on each row, but are descriptive rather than causal.", "",
        "## Overall paired comparisons by split",
        overall.to_markdown(index=False) if not overall.empty else "No compatible comparison columns found.", "",
        "## By discount",
        "See `paired_comparisons_by_discount.csv` for per-discount paired differences and bootstrap intervals.", "",
        "## Interpretation limits",
    ]
    md.extend(f"- {x}" for x in report["limitations"])
    md += ["", "## Safety", "- Model loaded: no", "- Inference: no", "- Training: no", "- Checkpoint modified: no", "- Input CSV modified: no"]
    (OUT / "paired_baseline_audit.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print("Paired PPO comparison audit completed (read-only).")
    print(f"Rows: {len(df)}")
    print(f"Splits: {report['split_counts']}")
    print(f"Summary: {OUT / 'paired_baseline_audit.md'}")
    print(f"Overall CSV: {OUT / 'paired_comparison_summary.csv'}")
    if not by_discount.empty:
        print(f"By-discount CSV: {OUT / 'paired_comparisons_by_discount.csv'}")
    print("Safety: no model inference, no training, no checkpoint/input CSV changes.")

if __name__ == "__main__":
    main()
