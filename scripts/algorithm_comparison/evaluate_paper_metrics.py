from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(".")
PPO_FILE = ROOT / "reports/evaluation_audit/final_baseline_comparison/by_product.csv"
INPUTS = {
    "PPO": (PPO_FILE, "ppo_estimated_sales", "ppo_estimated_margin_proxy",
            "ppo_reward", "ppo_discount", "baseline_margin_proxy"),
    "DDPG": (ROOT / "reports/algorithm_comparison/ddpg_test_evaluation.csv",
             "ddpg_sales", "ddpg_margin", "ddpg_reward", "ddpg_discount", "baseline_margin"),
    "SAC": (ROOT / "reports/algorithm_comparison/sac_test_evaluation.csv",
            "sac_sales", "sac_margin", "sac_reward", "sac_discount", "baseline_margin"),
    "DDQN": (ROOT / "reports/algorithm_comparison/ddqn_test_evaluation.csv",
             "ddqn_sales", "ddqn_margin", "ddqn_reward", "ddqn_discount", "baseline_margin"),
}
OUT = ROOT / "reports/algorithm_comparison/paper_aligned_metrics"
SEED = 42
BOOTSTRAPS = 5000


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    output_names = [
        "per_product_growth.csv",
        "growth_summary.csv",
        "topk_rankings.csv",
        "paired_comparisons.csv",
        "methodology.json",
        "paper_aligned_metrics_report.md",
    ]
    existing = [str(OUT / name) for name in output_names if (OUT / name).exists()]
    require(not existing, "Refusing to overwrite existing outputs:\n" + "\n".join(existing))

    frames = {}
    source_sets = {}
    for algorithm, spec in INPUTS.items():
        path, sales_col, margin_col, reward_col, discount_col, baseline_margin_col = spec
        require(path.is_file(), f"Missing input: {path}")
        df = pd.read_csv(path)
        required = {
            "source_row", "baseline_sales", baseline_margin_col,
            sales_col, margin_col, reward_col, discount_col
        }
        missing = required - set(df.columns)
        require(not missing, f"{algorithm}: missing columns {sorted(missing)}")
        require(not df["source_row"].duplicated().any(),
                f"{algorithm}: duplicate source_row values")
        source_sets[algorithm] = set(df["source_row"].astype(int))
        frames[algorithm] = df

    reference = source_sets["PPO"]
    for algorithm, rows in source_sets.items():
        require(rows == reference, f"{algorithm} test rows differ from PPO")

    # Build a common product-level table. source_row is a row locator,
    # not an asserted unique real-world product identifier.
    rows = []
    for algorithm, spec in INPUTS.items():
        _, sales_col, margin_col, reward_col, discount_col, baseline_margin_col = spec
        df = frames[algorithm]
        base_sales = pd.to_numeric(df["baseline_sales"], errors="coerce")
        sales = pd.to_numeric(df[sales_col], errors="coerce")
        base_margin = pd.to_numeric(df[baseline_margin_col], errors="coerce")
        margin = pd.to_numeric(df[margin_col], errors="coerce")

        valid_sales = (
            np.isfinite(base_sales) & np.isfinite(sales) & (base_sales > 0)
        )
        valid_margin = (
            np.isfinite(base_margin) & np.isfinite(margin) & (base_margin > 0)
        )
        sales_growth = pd.Series(np.nan, index=df.index, dtype=float)
        margin_growth = pd.Series(np.nan, index=df.index, dtype=float)
        sales_growth.loc[valid_sales] = (
            (sales.loc[valid_sales] - base_sales.loc[valid_sales])
            / base_sales.loc[valid_sales] * 100
        )
        margin_growth.loc[valid_margin] = (
            (margin.loc[valid_margin] - base_margin.loc[valid_margin])
            / base_margin.loc[valid_margin] * 100
        )

        part = pd.DataFrame({
            "algorithm": algorithm,
            "source_row": df["source_row"].astype(int),
            "discount_pct": pd.to_numeric(df[discount_col], errors="coerce"),
            "reward": pd.to_numeric(df[reward_col], errors="coerce"),
            "baseline_sales": base_sales,
            "estimated_sales": sales,
            "sales_growth_pct": sales_growth,
            "baseline_margin_proxy": base_margin,
            "estimated_margin_proxy": margin,
            "margin_proxy_growth_pct": margin_growth,
        })
        rows.append(part)

    detail = pd.concat(rows, ignore_index=True)
    require(detail["source_row"].nunique() == len(reference),
            "Unexpected source-row count in combined detail data")

    # Top-1 / Top-3 among the four evaluated algorithms for each source row.
    ranking_frames = []
    for metric in ("sales_growth_pct", "margin_proxy_growth_pct"):
        wide = detail.pivot(index="source_row", columns="algorithm", values=metric)
        ranks = wide.rank(axis=1, ascending=False, method="min")
        for algorithm in INPUTS:
            valid = wide[algorithm].notna()
            ranking_frames.append({
                "metric": metric,
                "algorithm": algorithm,
                "products_ranked": int(valid.sum()),
                "top1_count": int(((ranks[algorithm] == 1) & valid).sum()),
                "top1_pct": float(100 * ((ranks[algorithm] == 1) & valid).sum() / max(valid.sum(), 1)),
                "top3_count": int(((ranks[algorithm] <= 3) & valid).sum()),
                "top3_pct": float(100 * ((ranks[algorithm] <= 3) & valid).sum() / max(valid.sum(), 1)),
                "ranking_scope": "PPO, DDPG, SAC, DDQN only; ties use minimum rank",
            })
    topk = pd.DataFrame(ranking_frames)

    # Descriptive product-level distributions.
    summary_rows = []
    for algorithm, group in detail.groupby("algorithm", sort=False):
        for metric in ("reward", "sales_growth_pct", "margin_proxy_growth_pct"):
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            summary_rows.append({
                "algorithm": algorithm,
                "metric": metric,
                "n": len(values),
                "mean": values.mean() if len(values) else np.nan,
                "median": values.median() if len(values) else np.nan,
                "std": values.std(ddof=1) if len(values) > 1 else np.nan,
                "q25": values.quantile(.25) if len(values) else np.nan,
                "q75": values.quantile(.75) if len(values) else np.nan,
                "min": values.min() if len(values) else np.nan,
                "max": values.max() if len(values) else np.nan,
            })
    summary = pd.DataFrame(summary_rows)

    # Paired product-level differences, with bootstrap CI for the mean
    # difference and Wilcoxon signed-rank test where defined.
    rng = np.random.default_rng(SEED)
    paired_rows = []
    for metric in ("reward", "sales_growth_pct", "margin_proxy_growth_pct"):
        wide = detail.pivot(index="source_row", columns="algorithm", values=metric)
        for i, a in enumerate(INPUTS):
            for b in list(INPUTS)[i + 1:]:
                pair = wide[[a, b]].dropna()
                diff = (pair[a] - pair[b]).to_numpy(dtype=float)
                n = len(diff)
                if n == 0:
                    mean_diff = median_diff = ci_low = ci_high = p_value = np.nan
                else:
                    mean_diff = float(np.mean(diff))
                    median_diff = float(np.median(diff))
                    samples = rng.choice(diff, size=(BOOTSTRAPS, n), replace=True).mean(axis=1)
                    ci_low, ci_high = np.quantile(samples, [.025, .975])
                    try:
                        p_value = float(stats.wilcoxon(diff).pvalue) if np.any(diff != 0) else 1.0
                    except ValueError:
                        p_value = np.nan
                paired_rows.append({
                    "metric": metric,
                    "algorithm_a": a,
                    "algorithm_b": b,
                    "difference_definition": f"{a} minus {b}",
                    "paired_n": n,
                    "mean_difference": mean_diff,
                    "median_difference": median_diff,
                    "bootstrap_95ci_low": ci_low,
                    "bootstrap_95ci_high": ci_high,
                    "wilcoxon_p_value": p_value,
                    "bootstrap_resamples": BOOTSTRAPS,
                    "random_seed": SEED,
                })
    paired = pd.DataFrame(paired_rows)

    methodology = {
        "test_products": len(reference),
        "algorithms": list(INPUTS),
        "inputs": {a: str(spec[0]) for a, spec in INPUTS.items()},
        "sales_growth_formula": "100 * (estimated_sales - baseline_sales) / baseline_sales; baseline must be positive",
        "margin_proxy_growth_formula": "100 * (estimated_margin_proxy - baseline_margin_proxy) / baseline_margin_proxy; baseline must be positive",
        "margin_interpretation": "Dataset-derived gross-margin proxy, not verified net profit",
        "topk_scope": "Ranking only among the four evaluated algorithms; ties use minimum rank",
        "paired_tests": "Wilcoxon signed-rank test plus percentile bootstrap CI for paired mean difference",
        "bootstrap_resamples": BOOTSTRAPS,
        "random_seed": SEED,
        "source_row_caveat": "source_row is a dataset row locator; no claim that it is a unique real-world product ID",
        "cumulative_reward_caveat": "Current environment is one-step; this script does not claim multi-month cumulative reward reproduction",
    }

    detail.to_csv(OUT / "per_product_growth.csv", index=False)
    summary.to_csv(OUT / "growth_summary.csv", index=False)
    topk.to_csv(OUT / "topk_rankings.csv", index=False)
    paired.to_csv(OUT / "paired_comparisons.csv", index=False)
    (OUT / "methodology.json").write_text(json.dumps(methodology, indent=2), encoding="utf-8")

    lines = [
        "# Paper-aligned product-level evaluation",
        "",
        f"- Matched test rows: {len(reference):,}",
        "- Algorithms: PPO, DDPG, SAC and DDQN",
        "- Sales growth is calculated relative to baseline sales.",
        "- Margin growth is reported as gross-margin-proxy growth, not verified profit.",
        "- Top-1/Top-3 rankings are restricted to these four algorithms; this is not the paper's full baseline ranking.",
        "- Paired tests compare matching source_row locators. source_row is not assumed to be a product ID.",
        "- This evaluation does not convert the one-step environment into a sequential multi-month simulator.",
        "",
        "## Growth and reward summary",
        "",
        summary.round(6).to_markdown(index=False),
        "",
        "## Top-1 / Top-3 rankings",
        "",
        topk.round(4).to_markdown(index=False),
        "",
        "## Interpretation limits",
        "",
        "- These are counterfactual simulator estimates, not live-market outcomes.",
        "- Positive baseline values are required for percentage growth; invalid denominators are excluded per metric.",
        "- Statistical tests are exploratory and do not correct for multiple comparisons.",
        "- See `paired_comparisons.csv` for paired differences, bootstrap intervals and Wilcoxon p-values.",
        "",
        "## Output files",
        "",
        "- `per_product_growth.csv`",
        "- `growth_summary.csv`",
        "- `topk_rankings.csv`",
        "- `paired_comparisons.csv`",
        "- `methodology.json`",
    ]
    (OUT / "paper_aligned_metrics_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("Paper-aligned evaluation completed.")
    print(f"Products matched: {len(reference)}")
    print(f"Outputs: {OUT}")
    print("\nGrowth summary:")
    print(summary.round(4).to_string(index=False))
    print("\nTop-1 / Top-3:")
    print(topk.round(3).to_string(index=False))
    print("\nNo models were trained and no existing files were overwritten.")


if __name__ == "__main__":
    main()
