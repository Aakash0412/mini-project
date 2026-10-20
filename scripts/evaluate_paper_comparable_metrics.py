#!/usr/bin/env python3
"""
Paper-comparable evaluation for the Dynamic Pricing for E-commerce project.

This script is intentionally read-only with respect to existing artifacts:
- It reads existing evaluation/robustness CSVs.
- It writes only to a new output directory and refuses to overwrite it.
- It does not train or modify models.

It reports both comparable metric definitions and protocol differences. In
particular, post-hoc reward reweighting is NOT presented as a policy trained
under that reward configuration.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

PAPER = {
    "title": "A Multimodal Deep Reinforcement Learning Framework for Dynamic Pricing Optimisation in E-Commerce",
    "doi": "10.1109/ACCESS.2026.3680140",
    "test_samples": 6810,
    "aggregate_profit_margin_growth_pct": ">16.7%",
    "sales_volume_change_pct": "-0.3% to -2.0%",
    "ppo_top1_profit_count": 3563,
    "ppo_top1_profit_pct": 52.32,
    "ppo_top3_sales_count": 5062,
    "ppo_top3_sales_pct": 74.33,
}
REWARD_CONFIGS = {
    "sales_oriented": {"alpha": 0.2, "beta": 0.8},
    "balanced": {"alpha": 0.5, "beta": 0.5},
    "profit_oriented": {"alpha": 0.8, "beta": 0.2},
}


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate paper-comparable metrics from existing evaluation artifacts."
    )
    p.add_argument(
        "--evaluation-csv", type=Path,
        default=Path("models/ppo_transfer/transfer_supported_8action_policy_evaluation.csv"),
        help="Existing evaluation CSV for the primary policy.",
    )
    p.add_argument(
        "--product-metrics-csv", type=Path,
        default=Path("reports/algorithm_comparison/paper_aligned_metrics/per_product_growth.csv"),
        help="Existing long-form product metrics for algorithm ranking; optional if absent.",
    )
    p.add_argument(
        "--robustness-csv", type=Path,
        default=Path("reports/robustness/multiseed_aggregate.csv"),
        help="Existing noise robustness aggregate CSV; optional if absent.",
    )
    p.add_argument(
        "--balanced-evaluation-csv", type=Path, default=None,
        help="Optional evaluation CSV generated with a separately trained balanced-reward policy.",
    )
    p.add_argument(
        "--profit-evaluation-csv", type=Path, default=None,
        help="Optional evaluation CSV generated with a separately trained profit-oriented policy.",
    )
    p.add_argument(
        "--output-dir", type=Path, default=Path("reports/base_paper_comparison_v2"),
        help="New output directory. It must not already exist.",
    )
    return p.parse_args()


def require_columns(df: pd.DataFrame, columns: list[str], path: Path):
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")


def read_test(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Evaluation CSV not found: {path}")
    df = pd.read_csv(path)
    require_columns(
        df,
        [
            "split", "source_row", "ppo_reward", "ppo_sales", "ppo_margin",
            "baseline_sales", "baseline_margin",
            "ppo_relative_delta_quantity", "ppo_relative_delta_margin",
        ],
        path,
    )
    test = df.loc[df["split"].astype(str).str.lower().eq("test")].copy()
    if test.empty:
        raise ValueError(f"No split='test' rows found in {path}")
    if test["source_row"].duplicated().any():
        raise ValueError(f"Duplicate source_row values in test split of {path}")
    return test


def safe_ratio_change(outcome: pd.Series, baseline: pd.Series) -> tuple[float, int]:
    o = pd.to_numeric(outcome, errors="coerce").to_numpy(dtype=float)
    b = pd.to_numeric(baseline, errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(o) & np.isfinite(b) & (b > 0)
    if not valid.any():
        return float("nan"), 0
    denominator = b[valid].sum()
    if denominator <= 0:
        return float("nan"), int(valid.sum())
    return float((o[valid].sum() / denominator - 1.0) * 100.0), int(valid.sum())


def mean_row_growth_pct(outcome: pd.Series, baseline: pd.Series) -> tuple[float, int]:
    o = pd.to_numeric(outcome, errors="coerce").to_numpy(dtype=float)
    b = pd.to_numeric(baseline, errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(o) & np.isfinite(b) & (b > 0)
    if not valid.any():
        return float("nan"), 0
    return float(np.mean((o[valid] / b[valid] - 1.0) * 100.0)), int(valid.sum())


def calculate_policy_metrics(test: pd.DataFrame, label: str, reward_config: str, policy_training_status: str):
    sales_agg, n_sales = safe_ratio_change(test["ppo_sales"], test["baseline_sales"])
    margin_agg, n_margin = safe_ratio_change(test["ppo_margin"], test["baseline_margin"])
    sales_row, _ = mean_row_growth_pct(test["ppo_sales"], test["baseline_sales"])
    margin_row, _ = mean_row_growth_pct(test["ppo_margin"], test["baseline_margin"])

    rows = [
        {
            "policy": label, "metric": "test_observations", "value": len(test),
            "unit": "rows", "valid_n": len(test), "reward_config": reward_config,
            "configuration_status": policy_training_status,
        },
        {
            "policy": label, "metric": "mean_reward_per_observation", "value": test["ppo_reward"].mean(),
            "unit": "reward", "valid_n": len(test), "reward_config": reward_config,
            "configuration_status": policy_training_status,
        },
        {
            "policy": label, "metric": "sum_one_step_rewards_over_test_products", "value": test["ppo_reward"].sum(),
            "unit": "summed one-step reward (not sequential cumulative reward)", "valid_n": len(test),
            "reward_config": reward_config, "configuration_status": policy_training_status,
        },
        {
            "policy": label, "metric": "mean_per_row_sales_growth", "value": sales_row,
            "unit": "%", "valid_n": n_sales, "reward_config": reward_config,
            "configuration_status": policy_training_status,
        },
        {
            "policy": label, "metric": "aggregate_sales_growth_ratio_of_totals", "value": sales_agg,
            "unit": "%", "valid_n": n_sales, "reward_config": reward_config,
            "configuration_status": policy_training_status,
        },
        {
            "policy": label, "metric": "mean_per_row_margin_proxy_growth", "value": margin_row,
            "unit": "%", "valid_n": n_margin, "reward_config": reward_config,
            "configuration_status": policy_training_status,
        },
        {
            "policy": label, "metric": "aggregate_margin_proxy_growth_ratio_of_totals", "value": margin_agg,
            "unit": "%", "valid_n": n_margin, "reward_config": reward_config,
            "configuration_status": policy_training_status,
        },
    ]

    # Recompute the same economic outcomes under each reward weighting.
    # This is explicitly post-hoc unless a separate trained-policy CSV is supplied.
    q = pd.to_numeric(test["ppo_relative_delta_quantity"], errors="coerce")
    m = pd.to_numeric(test["ppo_relative_delta_margin"], errors="coerce")
    for name, weights in REWARD_CONFIGS.items():
        reward = weights["alpha"] * m + weights["beta"] * q
        rows.append({
            "policy": label,
            "metric": f"posthoc_mean_reward_{name}",
            "value": float(reward.mean()),
            "unit": "reward per observation",
            "valid_n": int(reward.notna().sum()),
            "reward_config": name,
            "configuration_status": (
                "post-hoc reweighting of this policy's fixed outcomes; not a policy trained with this reward"
            ),
        })
        rows.append({
            "policy": label,
            "metric": f"posthoc_sum_reward_{name}",
            "value": float(reward.sum()),
            "unit": "summed one-step reward",
            "valid_n": int(reward.notna().sum()),
            "reward_config": name,
            "configuration_status": (
                "post-hoc reweighting of this policy's fixed outcomes; not a policy trained with this reward"
            ),
        })
    return rows


def normalise_product_metrics(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    # Expected canonical long format from the project's product-level report.
    required = {"source_row", "algorithm", "sales_growth_pct", "margin_proxy_growth_pct"}
    if required.issubset(df.columns):
        out = df[list(required)].copy()
        out["source_row"] = pd.to_numeric(out["source_row"], errors="coerce")
        out["algorithm"] = out["algorithm"].astype(str).str.upper()
        out["sales_growth_pct"] = pd.to_numeric(out["sales_growth_pct"], errors="coerce")
        out["margin_proxy_growth_pct"] = pd.to_numeric(out["margin_proxy_growth_pct"], errors="coerce")
        return out

    # Also accept a likely legacy schema using 'method' instead of 'algorithm'.
    legacy = {"source_row", "method", "sales_growth_pct", "margin_proxy_growth_pct"}
    if legacy.issubset(df.columns):
        out = df[list(legacy)].copy().rename(columns={"method": "algorithm"})
        out["source_row"] = pd.to_numeric(out["source_row"], errors="coerce")
        out["algorithm"] = out["algorithm"].astype(str).str.upper()
        out["sales_growth_pct"] = pd.to_numeric(out["sales_growth_pct"], errors="coerce")
        out["margin_proxy_growth_pct"] = pd.to_numeric(out["margin_proxy_growth_pct"], errors="coerce")
        return out

    raise ValueError(
        f"{path} exists but does not have a supported schema. "
        "Expected source_row, algorithm (or method), sales_growth_pct, margin_proxy_growth_pct."
    )


def calculate_rankings(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in ["sales_growth_pct", "margin_proxy_growth_pct"]:
        valid = metrics.dropna(subset=["source_row", "algorithm", metric]).copy()
        if valid.empty:
            continue
        if valid.duplicated(["source_row", "algorithm"]).any():
            raise ValueError(f"Duplicate source_row/algorithm pairs in product metrics for {metric}")
        valid["rank"] = valid.groupby("source_row")[metric].rank(method="min", ascending=False)
        methods = sorted(valid["algorithm"].unique())
        for method in methods:
            own = valid.loc[valid["algorithm"].eq(method)]
            rows.append({
                "metric": metric,
                "algorithm": method,
                "n_products_ranked": int(own["source_row"].nunique()),
                "top1_count": int((own["rank"] == 1).sum()),
                "top1_pct": float((own["rank"] == 1).mean() * 100),
                "top3_count": int((own["rank"] <= 3).sum()),
                "top3_pct": float((own["rank"] <= 3).mean() * 100),
                "ranking_scope": f"{len(methods)} methods available in supplied product-metrics CSV",
                "paper_top1_profit_reference_pct": PAPER["ppo_top1_profit_pct"] if metric == "margin_proxy_growth_pct" and method == "PPO" else np.nan,
                "paper_top3_sales_reference_pct": PAPER["ppo_top3_sales_pct"] if metric == "sales_growth_pct" and method == "PPO" else np.nan,
            })
    return pd.DataFrame(rows)


def make_paper_comparison(summary: pd.DataFrame, ranking: Optional[pd.DataFrame]) -> pd.DataFrame:
    values = {}
    for _, r in summary.iterrows():
        values[(r["policy"], r["metric"])] = r["value"]

    primary = "primary_ppo"
    def get(metric):
        return values.get((primary, metric), np.nan)

    rows = [
        {
            "paper_metric": "Aggregate profit-margin growth",
            "paper_reference": PAPER["aggregate_profit_margin_growth_pct"],
            "local_metric": "aggregate_margin_proxy_growth_ratio_of_totals",
            "local_value": get("aggregate_margin_proxy_growth_ratio_of_totals"),
            "local_unit": "%",
            "comparability": "Directional only",
            "reason": "Local outcome is gross-margin proxy; one-step simulation and split differ from paper's sequential/product-time protocol.",
        },
        {
            "paper_metric": "Sales-volume change",
            "paper_reference": PAPER["sales_volume_change_pct"],
            "local_metric": "aggregate_sales_growth_ratio_of_totals",
            "local_value": get("aggregate_sales_growth_ratio_of_totals"),
            "local_unit": "%",
            "comparability": "Directional only",
            "reason": "Local value compares simulated sales with stored historical baseline; evaluation protocol and test population differ.",
        },
        {
            "paper_metric": "PPO Top-1 profit-growth frequency",
            "paper_reference": f'{PAPER["ppo_top1_profit_pct"]:.2f}% ({PAPER["ppo_top1_profit_count"]}/{PAPER["test_samples"]})',
            "local_metric": "PPO Top-1 margin-proxy-growth frequency",
            "local_value": np.nan,
            "local_unit": "%",
            "comparability": "Requires product-metrics CSV",
            "reason": "Local ranking must use the same product-level comparison set; the current four-algorithm set is narrower than the paper's full baseline set.",
        },
        {
            "paper_metric": "PPO Top-3 sales-growth frequency",
            "paper_reference": f'{PAPER["ppo_top3_sales_pct"]:.2f}% ({PAPER["ppo_top3_sales_count"]}/{PAPER["test_samples"]})',
            "local_metric": "PPO Top-3 sales-growth frequency",
            "local_value": np.nan,
            "local_unit": "%",
            "comparability": "Requires product-metrics CSV",
            "reason": "Local ranking must use the same product-level comparison set; the current four-algorithm set is narrower than the paper's full baseline set.",
        },
        {
            "paper_metric": "Cumulative reward",
            "paper_reference": "Paper reports macro-level cumulative reward under multiple α/β settings; exact table/curve value must be matched by setting",
            "local_metric": "sum_one_step_rewards_over_test_products",
            "local_value": get("sum_one_step_rewards_over_test_products"),
            "local_unit": "reward",
            "comparability": "Not numerically equivalent",
            "reason": "Local sum aggregates one-step product rewards; paper's macro evaluation uses a sequential simulation protocol.",
        },
        {
            "paper_metric": "Prediction-noise robustness",
            "paper_reference": "Noise σ ∈ {0, 0.05, 0.10, 0.15} × Std(predicted sales response)",
            "local_metric": "See robustness_comparison.csv",
            "local_value": np.nan,
            "local_unit": "reward retention %",
            "comparability": "Protocol partly aligned",
            "reason": "Noise levels align; local artifact evaluates a fixed policy and reports variation across noise seeds.",
        },
    ]
    if ranking is not None and not ranking.empty:
        for idx, row in ranking.iterrows():
            if row["algorithm"] == "PPO" and row["metric"] == "margin_proxy_growth_pct":
                for item in rows:
                    if item["paper_metric"] == "PPO Top-1 profit-growth frequency":
                        item["local_value"] = row["top1_pct"]
                        item["comparability"] = "Restricted comparison set"
                        item["reason"] = row["ranking_scope"] + "; paper uses a broader set of baseline methods."
            if row["algorithm"] == "PPO" and row["metric"] == "sales_growth_pct":
                for item in rows:
                    if item["paper_metric"] == "PPO Top-3 sales-growth frequency":
                        item["local_value"] = row["top3_pct"]
                        item["comparability"] = "Restricted comparison set"
                        item["reason"] = row["ranking_scope"] + "; paper uses a broader set of baseline methods."
    return pd.DataFrame(rows)


def markdown_report(summary: pd.DataFrame, comparison: pd.DataFrame, ranking: Optional[pd.DataFrame], robustness: Optional[pd.DataFrame]) -> str:
    lines = [
        "# Base-paper comparison report",
        "",
        f"Reference: {PAPER['title']} (IEEE Access 2026, DOI {PAPER['doi']}).",
        "",
        "## Scope and safeguards",
        "",
        "- Read-only evaluation of existing artifacts; no model training or checkpoint changes.",
        "- Output metrics use the local held-out test rows. The paper used 6,810 test samples; this local evaluation uses its own 7,139-row chronological test subset.",
        "- Local margin is a gross-margin proxy, not verified accounting profit.",
        "- Local policy environment is one-step per product. Paper describes month-wise/sequential simulation.",
        "- Local state is 111-D and omits the paper's 10-D review sentiment representation.",
        "- Local primary policy uses eight discounts (5–40%), excluding the paper's 0% action.",
        "",
        "## Local primary-policy metrics",
        "",
        summary.loc[summary["policy"].eq("primary_ppo"), ["metric", "value", "unit", "valid_n", "configuration_status"]].to_markdown(index=False),
        "",
        "## Comparison against paper-reported values",
        "",
        comparison.to_markdown(index=False),
        "",
    ]
    if ranking is not None and not ranking.empty:
        lines += [
            "## Product-level Top-1 / Top-3 frequencies",
            "",
            ranking.to_markdown(index=False),
            "",
            "Ranks are computed within each source_row over only the algorithms present in the supplied local product-metrics file. They are not the paper's full comparison set unless all corresponding baselines are present.",
            "",
        ]
    else:
        lines += [
            "## Product-level Top-1 / Top-3 frequencies",
            "",
            "Not computed: the product-level multi-algorithm CSV was not found or was not supplied. The PPO-vs-oracle evaluation CSV is insufficient to reproduce rankings across the paper's comparison set.",
            "",
        ]
    if robustness is not None and not robustness.empty:
        lines += [
            "## Prediction-noise robustness",
            "",
            robustness.to_markdown(index=False),
            "",
        ]
    else:
        lines += [
            "## Prediction-noise robustness",
            "",
            "Not computed: robustness aggregate CSV not found.",
            "",
        ]
    lines += [
        "## Interpretation",
        "",
        "This report makes local metric definitions as comparable as the available artifacts allow, while explicitly marking protocol differences. A numerical difference from the paper must not be interpreted as a replication failure or success until test population, baseline methods, reward protocol and margin definition are matched.",
        "",
        "Post-hoc α/β reward reweighting only scores the already-selected actions under alternative weights; it does not represent a separately trained policy for each reward configuration.",
        "",
    ]
    return "\n".join(lines)


def main():
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(
            f"Output directory already exists; refusing to overwrite: {args.output_dir}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=False)

    primary_test = read_test(args.evaluation_csv)
    summary_rows = calculate_policy_metrics(
        primary_test, "primary_ppo", "sales_oriented",
        "existing primary PPO checkpoint; trained configuration presumed sales-oriented",
    )

    optional_evals = [
        ("balanced_ppo", args.balanced_evaluation_csv, "balanced"),
        ("profit_ppo", args.profit_evaluation_csv, "profit_oriented"),
    ]
    for label, path, config in optional_evals:
        if path is not None:
            test = read_test(path)
            summary_rows.extend(calculate_policy_metrics(
                test, label, config,
                f"separate evaluation CSV supplied for {config}; verify its checkpoint training config independently",
            ))

    summary = pd.DataFrame(summary_rows)
    product_metrics = normalise_product_metrics(args.product_metrics_csv)
    ranking = calculate_rankings(product_metrics) if product_metrics is not None else None

    comparison = make_paper_comparison(summary, ranking)

    robustness = None
    if args.robustness_csv.exists():
        r = pd.read_csv(args.robustness_csv)
        required = {"noise_fraction", "mean_reward_mean", "reward_retention_percent_mean", "reward_change_percent_mean"}
        if required.issubset(r.columns):
            robustness = r[[
                "noise_fraction", "independent_seeds", "evaluated_products_per_seed",
                "mean_reward_mean", "mean_reward_ci95_low", "mean_reward_ci95_high",
                "reward_retention_percent_mean", "reward_change_percent_mean",
                "reward_change_percent_ci95_low", "reward_change_percent_ci95_high",
            ]].copy()
        else:
            raise ValueError(f"{args.robustness_csv} lacks required columns: {sorted(required - set(r.columns))}")

    summary.to_csv(args.output_dir / "local_metric_summary.csv", index=False)
    comparison.to_csv(args.output_dir / "paper_comparison.csv", index=False)
    if ranking is not None:
        ranking.to_csv(args.output_dir / "topk_comparison.csv", index=False)
    if robustness is not None:
        robustness.to_csv(args.output_dir / "robustness_comparison.csv", index=False)

    metadata = {
        "paper": PAPER,
        "inputs": {
            "evaluation_csv": str(args.evaluation_csv),
            "product_metrics_csv": str(args.product_metrics_csv),
            "robustness_csv": str(args.robustness_csv),
            "balanced_evaluation_csv": str(args.balanced_evaluation_csv) if args.balanced_evaluation_csv else None,
            "profit_evaluation_csv": str(args.profit_evaluation_csv) if args.profit_evaluation_csv else None,
        },
        "test_rows_primary": int(len(primary_test)),
        "output_files": [
            "local_metric_summary.csv", "paper_comparison.csv", "paper_comparison_report.md",
            *(["topk_comparison.csv"] if ranking is not None else []),
            *(["robustness_comparison.csv"] if robustness is not None else []),
        ],
        "warnings": [
            "Local gross-margin proxy is not verified profit.",
            "Local one-step product simulation is not identical to paper's sequential/month-wise protocol.",
            "Local test split and product population differ from paper's 6,810-sample test set.",
            "Post-hoc reward reweighting does not represent separately trained policies.",
            "Local Top-1/Top-3 rankings cover only algorithms present in the supplied product metrics file.",
        ],
    }
    (args.output_dir / "methodology.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (args.output_dir / "paper_comparison_report.md").write_text(
        markdown_report(summary, comparison, ranking, robustness), encoding="utf-8"
    )

    print(f"Created new report directory: {args.output_dir}")
    print(f"Primary test observations: {len(primary_test):,}")
    print(f"Aggregate sales growth: {comparison.loc[comparison.paper_metric.eq('Sales-volume change'), 'local_value'].iloc[0]:.4f}%")
    print(f"Aggregate margin-proxy growth: {comparison.loc[comparison.paper_metric.eq('Aggregate profit-margin growth'), 'local_value'].iloc[0]:.4f}%")
    print(f"Top-k rankings calculated: {'yes' if ranking is not None else 'no (product-metrics CSV missing)'}")
    print("No model was trained and no existing artifact was overwritten.")


if __name__ == "__main__":
    main()
