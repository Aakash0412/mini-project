
from pathlib import Path
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT = (
    ROOT
    / "reports/base_paper_reproduction/economic_robustness"
    / "economic_robustness_by_product.csv"
)
OUTPUT_DIR = (
    ROOT
    / "reports/base_paper_reproduction/economic_robustness"
)

SUMMARY_CSV = OUTPUT_DIR / "paired_bootstrap_summary.csv"
SUMMARY_JSON = OUTPUT_DIR / "paired_bootstrap_summary.json"

NOISE_LEVELS = [0.00, 0.01, 0.05, 0.10]
METRICS = [
    "simulated_reward",
    "estimated_revenue",
    "estimated_margin_proxy",
    "estimated_sales",
]
N_BOOTSTRAPS = 10_000
SEED = 42


def bootstrap_mean_ci(values, rng):
    """Percentile bootstrap CI for the mean of paired product differences."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return None, None, None

    observed_mean = float(values.mean())
    estimates = np.empty(N_BOOTSTRAPS)

    for start in range(0, N_BOOTSTRAPS, 500):
        count = min(500, N_BOOTSTRAPS - start)
        indices = rng.integers(0, len(values), size=(count, len(values)))
        estimates[start:start + count] = values[indices].mean(axis=1)

    low, high = np.percentile(estimates, [2.5, 97.5])
    return observed_mean, float(low), float(high)


def bootstrap_share_ci(indicators, rng):
    """Percentile bootstrap CI for a product-level event share."""
    values = np.asarray(indicators, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return None, None, None

    observed = float(values.mean() * 100)
    estimates = np.empty(N_BOOTSTRAPS)

    for start in range(0, N_BOOTSTRAPS, 500):
        count = min(500, N_BOOTSTRAPS - start)
        indices = rng.integers(0, len(values), size=(count, len(values)))
        estimates[start:start + count] = values[indices].mean(axis=1) * 100

    low, high = np.percentile(estimates, [2.5, 97.5])
    return observed, float(low), float(high)


def summarize_comparison(
    differences, noise, comparison, metric, rng
):
    differences = np.asarray(differences, dtype=float)
    differences = differences[np.isfinite(differences)]

    mean, ci_low, ci_high = bootstrap_mean_ci(differences, rng)
    win, win_low, win_high = bootstrap_share_ci(differences > 0, rng)
    tie_share = (
        float(np.mean(differences == 0) * 100)
        if len(differences) else None
    )

    return {
        "noise_fraction_of_train_std": noise,
        "comparison": comparison,
        "metric": metric,
        "paired_products": int(len(differences)),
        "mean_difference": mean,
        "mean_difference_ci95_low": ci_low,
        "mean_difference_ci95_high": ci_high,
        "median_difference": (
            float(np.median(differences)) if len(differences) else None
        ),
        "positive_difference_share_percent": win,
        "positive_share_ci95_low": win_low,
        "positive_share_ci95_high": win_high,
        "tie_share_percent": tie_share,
    }


def main():
    if not INPUT.exists():
        raise FileNotFoundError(f"Required evaluation file not found: {INPUT}")

    for path in (SUMMARY_CSV, SUMMARY_JSON):
        if path.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing result: {path}"
            )

    df = pd.read_csv(INPUT)

    required = {
        "noise_fraction_of_train_std",
        "source_row",
        "method",
        *METRICS,
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV is missing columns: {sorted(missing)}")

    if df.duplicated(
        ["noise_fraction_of_train_std", "source_row", "method"]
    ).any():
        raise ValueError("Duplicate product/method/noise records found.")

    rng = np.random.default_rng(SEED)
    results = []

    # Pivot to one row per product, with paired method/metric columns.
    indexed = df.set_index(
        ["noise_fraction_of_train_std", "source_row", "method"]
    )

    # A) PPO minus fixed-20% pricing at each noise level.
    for noise in NOISE_LEVELS:
        level = indexed.loc[noise]

        for metric in METRICS:
            wide = level[metric].unstack("method")
            if not {"ppo", "fixed_20"}.issubset(wide.columns):
                raise ValueError(f"Missing method records at noise {noise}")

            paired = wide[["ppo", "fixed_20"]].dropna()
            differences = (
                paired["ppo"] - paired["fixed_20"]
            ).to_numpy()

            results.append(summarize_comparison(
                differences=differences,
                noise=noise,
                comparison="PPO minus fixed 20%",
                metric=metric,
                rng=rng,
            ))

    # B) Perturbed PPO minus clean PPO for each product.
    clean = indexed.loc[0.00]
    clean_ppo = clean.xs("ppo", level="method")

    for noise in NOISE_LEVELS:
        if noise == 0.00:
            continue

        perturbed = indexed.loc[noise].xs("ppo", level="method")

        for metric in METRICS:
            paired = pd.concat(
                [
                    clean_ppo[metric].rename("clean"),
                    perturbed[metric].rename("perturbed"),
                ],
                axis=1,
                join="inner",
            ).dropna()

            differences = (
                paired["perturbed"] - paired["clean"]
            ).to_numpy()

            results.append(summarize_comparison(
                differences=differences,
                noise=noise,
                comparison="Perturbed PPO minus clean PPO",
                metric=metric,
                rng=rng,
            ))

            # Negative differences indicate deterioration in this metric.
            adverse, adverse_low, adverse_high = bootstrap_share_ci(
                differences < 0, rng
            )

            results.append({
                "noise_fraction_of_train_std": noise,
                "comparison": "Perturbed PPO versus clean PPO",
                "metric": f"{metric}_adverse_share",
                "paired_products": int(len(differences)),
                "mean_difference": None,
                "mean_difference_ci95_low": None,
                "mean_difference_ci95_high": None,
                "median_difference": None,
                "positive_difference_share_percent": None,
                "positive_share_ci95_low": None,
                "positive_share_ci95_high": None,
                "tie_share_percent": None,
                "adverse_share_percent": adverse,
                "adverse_share_ci95_low": adverse_low,
                "adverse_share_ci95_high": adverse_high,
            })

    result_df = pd.DataFrame(results)
    result_df.to_csv(SUMMARY_CSV, index=False)

    metadata = {
        "input_file": str(INPUT.relative_to(ROOT)),
        "bootstrap_resamples": N_BOOTSTRAPS,
        "bootstrap_confidence_level": 0.95,
        "bootstrap_method": "Percentile bootstrap over paired products",
        "seed": SEED,
        "metrics": METRICS,
        "noise_levels": NOISE_LEVELS,
        "interpretation": [
            "Confidence intervals quantify product-sampling uncertainty conditional on the existing simulator.",
            "They do not capture sales-predictor misspecification or real-world uncertainty.",
            "A CI excluding zero supports a nonzero mean paired difference under this evaluation setup.",
            "Positive reward differences do not imply higher revenue or margin proxy.",
        ],
    }
    SUMMARY_JSON.write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print("Paired bootstrap analysis complete.")
    print(f"Input: {INPUT}")
    print(f"CSV: {SUMMARY_CSV}")
    print(f"Metadata: {SUMMARY_JSON}")

    display_columns = [
        "noise_fraction_of_train_std",
        "comparison",
        "metric",
        "paired_products",
        "mean_difference",
        "mean_difference_ci95_low",
        "mean_difference_ci95_high",
        "adverse_share_percent",
        "adverse_share_ci95_low",
        "adverse_share_ci95_high",
    ]
    print("\nResults:")
    print(result_df[
        [c for c in display_columns if c in result_df.columns]
    ].to_string(index=False))


if __name__ == "__main__":
    main()
