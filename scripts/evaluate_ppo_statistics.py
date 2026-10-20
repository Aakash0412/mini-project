
import numpy as np
import pandas as pd

PATH = (
    "models/ppo_transfer/"
    "transfer_supported_8action_policy_evaluation.csv"
)

N_BOOTSTRAP = 10000
SEED = 42
CI_LEVEL = 0.95

df = pd.read_csv(PATH)
test = df.loc[df["split"].eq("test")].copy().reset_index(drop=True)

if test.empty:
    raise ValueError("No held-out test rows found.")

if test["source_row"].duplicated().any():
    raise ValueError("Duplicate source_row values found in test set.")

rng = np.random.default_rng(SEED)
alpha = 1.0 - CI_LEVEL


def bootstrap_mean_ci(values, rng):
    """Percentile bootstrap CI for a sample mean."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.nan, np.nan, np.nan

    estimates = np.empty(N_BOOTSTRAP)
    n = len(values)

    for start in range(0, N_BOOTSTRAP, 250):
        stop = min(start + 250, N_BOOTSTRAP)
        indices = rng.integers(0, n, size=(stop - start, n))
        estimates[start:stop] = values[indices].mean(axis=1)

    lower, upper = np.quantile(
        estimates, [alpha / 2, 1 - alpha / 2]
    )
    return values.mean(), lower, upper


def bootstrap_ratio_change_ci(ppo, baseline, rng):
    """
    Bootstrap ratio-of-means percentage change:
    (mean PPO / mean baseline - 1) * 100.
    """
    ppo = np.asarray(ppo, dtype=float)
    baseline = np.asarray(baseline, dtype=float)
    valid = (
        np.isfinite(ppo)
        & np.isfinite(baseline)
    )
    ppo = ppo[valid]
    baseline = baseline[valid]

    if len(ppo) == 0 or np.isclose(baseline.mean(), 0):
        return np.nan, np.nan, np.nan

    point = (ppo.mean() / baseline.mean() - 1) * 100
    estimates = np.empty(N_BOOTSTRAP)
    n = len(ppo)

    for start in range(0, N_BOOTSTRAP, 250):
        stop = min(start + 250, N_BOOTSTRAP)
        indices = rng.integers(0, n, size=(stop - start, n))
        p_mean = ppo[indices].mean(axis=1)
        b_mean = baseline[indices].mean(axis=1)

        estimates[start:stop] = np.where(
            np.abs(b_mean) > 1e-12,
            (p_mean / b_mean - 1) * 100,
            np.nan,
        )

    estimates = estimates[np.isfinite(estimates)]
    lower, upper = np.quantile(
        estimates, [alpha / 2, 1 - alpha / 2]
    )
    return point, lower, upper


print("=" * 78)
print("STATISTICAL VALIDATION: PPO VS HISTORICAL BASELINE")
print("=" * 78)
print(f"Test rows: {len(test)}")
print(f"Bootstrap replicates: {N_BOOTSTRAP}")
print(f"Random seed: {SEED}")
print(f"Confidence level: {CI_LEVEL:.0%}")
print(
    "Resampling unit: individual test rows. "
    "Intervals assume rows are independent."
)

metrics = [
    {
        "label": "Sales",
        "ppo": "ppo_sales",
        "baseline": "baseline_sales",
    },
    {
        "label": "Margin proxy",
        "ppo": "ppo_margin",
        "baseline": "baseline_margin",
    },
]

summary_rows = []

for metric in metrics:
    label = metric["label"]
    ppo = test[metric["ppo"]].to_numpy(dtype=float)
    baseline = test[metric["baseline"]].to_numpy(dtype=float)

    valid = np.isfinite(ppo) & np.isfinite(baseline)
    ppo = ppo[valid]
    baseline = baseline[valid]

    # Paired differences: each PPO outcome is compared with
    # the baseline outcome for the same test row.
    diff = ppo - baseline

    diff_mean, diff_lo, diff_hi = bootstrap_mean_ci(diff, rng)
    ratio, ratio_lo, ratio_hi = bootstrap_ratio_change_ci(
        ppo, baseline, rng
    )

    print("\n" + "-" * 78)
    print(label.upper())
    print("-" * 78)

    print(f"Valid paired rows: {len(diff)}")
    print(f"PPO mean: {ppo.mean():,.4f}")
    print(f"Baseline mean: {baseline.mean():,.4f}")
    print(f"Mean paired difference: {diff_mean:,.4f}")
    print(f"Median paired difference: {np.median(diff):,.4f}")
    print(
        f"95% bootstrap CI for mean difference: "
        f"[{diff_lo:,.4f}, {diff_hi:,.4f}]"
    )

    print(f"Ratio-of-means change: {ratio:.4f}%")
    print(
        f"95% bootstrap CI for ratio-of-means change: "
        f"[{ratio_lo:.4f}%, {ratio_hi:.4f}%]"
    )

    print(f"Rows with positive difference: {(diff > 0).mean() * 100:.2f}%")
    print(f"Rows with negative difference: {(diff < 0).mean() * 100:.2f}%")
    print(f"Rows with zero difference: {np.isclose(diff, 0).mean() * 100:.2f}%")

    print(f"Mean difference / SD: {diff.mean() / diff.std(ddof=1):.6f}"
          if diff.std(ddof=1) > 0 else "Mean difference / SD: undefined")

    summary_rows.append({
        "metric": label,
        "n": len(diff),
        "ppo_mean": ppo.mean(),
        "baseline_mean": baseline.mean(),
        "mean_paired_difference": diff_mean,
        "difference_ci_lower": diff_lo,
        "difference_ci_upper": diff_hi,
        "ratio_of_means_change_pct": ratio,
        "ratio_change_ci_lower_pct": ratio_lo,
        "ratio_change_ci_upper_pct": ratio_hi,
        "positive_difference_pct": (diff > 0).mean() * 100,
    })

print("\n" + "=" * 78)
print("INTERPRETATION")
print("=" * 78)

for row in summary_rows:
    lo = row["difference_ci_lower"]
    hi = row["difference_ci_upper"]

    if lo > 0:
        conclusion = (
            "The 95% bootstrap CI for the mean paired difference "
            "is entirely above zero."
        )
    elif hi < 0:
        conclusion = (
            "The 95% bootstrap CI for the mean paired difference "
            "is entirely below zero."
        )
    else:
        conclusion = (
            "The 95% bootstrap CI for the mean paired difference "
            "includes zero; this analysis does not establish a "
            "nonzero mean difference at the corresponding CI level."
        )

    print(f"\n{row['metric']}: {conclusion}")

print("\nIMPORTANT LIMITATIONS")
print(
    "- These are simulator-derived comparisons, not causal estimates "
    "of real-world deployment effects."
)
print(
    "- Row-level bootstrap intervals may be too narrow if multiple "
    "rows belong to the same product or otherwise depend on each other."
)
print(
    "- The evaluation CSV does not contain a historical baseline "
    "revenue column, so revenue change versus baseline cannot be "
    "estimated from this file."
)
print(
    "- A confidence interval for a paired difference is not a substitute "
    "for evaluating performance across independent policy-training seeds."
)

print("\nNo project files were modified.")
