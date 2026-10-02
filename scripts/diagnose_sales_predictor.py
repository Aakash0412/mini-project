"""
Sales Predictor Diagnostic
==========================

Diagnoses the CURRENT frozen 112-D sales predictor without retraining it.

Checks:
1. Target distribution
2. Prediction distribution
3. MAE / RMSE / R²
4. Median absolute error
5. Prediction bias
6. Correlation
7. Per-discount performance
8. Prediction response across discount levels
9. Counterfactual monotonicity
10. Train/validation/test performance

IMPORTANT:
- No retraining
- No log1p transformation
- No modification of the current checkpoint
- Uses the current ItaNet-111 state
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.sales_predictor import load_frozen_predictor


# ============================================================
# CONFIG
# ============================================================

STATE_PATH = "data/processed/state/state.npy"
STATE_INDEX_PATH = "data/processed/state/state_row_indices.npy"
DATA_PATH = "data/processed/products_clean.parquet"

CHECKPOINT_DIR = "models/sales_predictor"

DISCOUNTS = [5, 10, 15, 20, 25, 30, 35, 40]

OUTPUT_DIR = "reports/sales_predictor_diagnostics"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("SALES PREDICTOR DIAGNOSTIC")
print("=" * 70)

print("\nLoading data...")

state = np.load(STATE_PATH)
state_indices = np.load(STATE_INDEX_PATH)

df_full = pd.read_parquet(DATA_PATH).reset_index(drop=True)

print(f"Full dataset:       {len(df_full)}")
print(f"State rows:         {len(state)}")
print(f"State dimension:    {state.shape[1]}")
print(f"State index rows:   {len(state_indices)}")

assert state.shape[0] == len(state_indices)

# Align dataset with valid state rows.
df = df_full.iloc[state_indices].reset_index(drop=True)

assert len(df) == len(state)

print(f"Aligned dataset:    {len(df)}")


# ============================================================
# TARGET
# ============================================================

target_col = "Estimated Monthly Sales Growth Rate"

y = (
    pd.to_numeric(df[target_col], errors="coerce")
    .fillna(0.0)
    .astype(np.float32)
    .values
)

discount = (
    pd.to_numeric(df["Discount"], errors="coerce")
    .fillna(0.0)
    .astype(np.float32)
    .values
)


# ============================================================
# SPLITS
# ============================================================

month = pd.to_numeric(df["MonthNum"], errors="coerce").values

train_mask = np.isin(month, [1, 2, 3, 4, 5, 6])
val_mask = np.isin(month, [7, 8])
test_mask = np.isin(month, [9, 10])

print("\nChronological splits:")
print(f"Train: {train_mask.sum()}")
print(f"Val:   {val_mask.sum()}")
print(f"Test:  {test_mask.sum()}")


# ============================================================
# TARGET DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("1. TARGET DISTRIBUTION")
print("=" * 70)

target_stats = {
    "count": int(len(y)),
    "mean": float(np.mean(y)),
    "std": float(np.std(y)),
    "min": float(np.min(y)),
    "p01": float(np.percentile(y, 1)),
    "p05": float(np.percentile(y, 5)),
    "p25": float(np.percentile(y, 25)),
    "median": float(np.median(y)),
    "p75": float(np.percentile(y, 75)),
    "p95": float(np.percentile(y, 95)),
    "p99": float(np.percentile(y, 99)),
    "max": float(np.max(y)),
}

for k, v in target_stats.items():
    print(f"{k:>10}: {v:.4f}")


# ============================================================
# LOAD FROZEN PREDICTOR
# ============================================================

print("\n" + "=" * 70)
print("2. LOADING FROZEN PREDICTOR")
print("=" * 70)

device = torch.device("cpu")

model = load_frozen_predictor(
    checkpoint_dir=CHECKPOINT_DIR,
    device=device,
)

model.eval()

print(f"Device: {device}")
print("Predictor loaded successfully.")
print("Predictor is frozen.")


# ============================================================
# PREDICT
# ============================================================

print("\nGenerating predictions...")

X_state = torch.tensor(state, dtype=torch.float32)

all_predictions = []

with torch.no_grad():
    batch_size = 512

    for start in range(0, len(state), batch_size):
        end = min(start + batch_size, len(state))

        s = X_state[start:end]

        # Observed historical discount.
        a = torch.tensor(
            discount[start:end],
            dtype=torch.float32,
        )

        pred = model.predict(s, a)

        all_predictions.append(pred.cpu().numpy())

predictions = np.concatenate(all_predictions)

assert len(predictions) == len(y)
assert np.all(np.isfinite(predictions))

print(f"Predictions generated: {len(predictions)}")


# ============================================================
# GENERAL METRICS
# ============================================================

def calculate_metrics(name, mask):

    actual = y[mask]
    pred = predictions[mask]

    mae = mean_absolute_error(actual, pred)
    rmse = np.sqrt(mean_squared_error(actual, pred))
    r2 = r2_score(actual, pred)

    median_ae = np.median(np.abs(actual - pred))

    bias = np.mean(pred - actual)

    try:
        pearson = pearsonr(actual, pred).statistic
    except Exception:
        pearson = np.nan

    try:
        spearman = spearmanr(actual, pred).statistic
    except Exception:
        spearman = np.nan

    print(f"\n{name}")
    print("-" * 50)
    print(f"Samples:              {len(actual)}")
    print(f"MAE:                  {mae:.4f}")
    print(f"RMSE:                 {rmse:.4f}")
    print(f"R²:                   {r2:.6f}")
    print(f"Median absolute error:{median_ae:.4f}")
    print(f"Prediction bias:      {bias:.4f}")
    print(f"Pearson correlation:   {pearson:.6f}")
    print(f"Spearman correlation:  {spearman:.6f}")

    return {
        "samples": int(len(actual)),
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2),
        "median_absolute_error": float(median_ae),
        "bias": float(bias),
        "pearson": float(pearson),
        "spearman": float(spearman),
    }


metrics = {}

metrics["train"] = calculate_metrics("TRAIN", train_mask)
metrics["validation"] = calculate_metrics("VALIDATION", val_mask)
metrics["test"] = calculate_metrics("TEST", test_mask)


# ============================================================
# PREDICTION DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("3. PREDICTION DISTRIBUTION")
print("=" * 70)

prediction_stats = {
    "mean": float(np.mean(predictions)),
    "std": float(np.std(predictions)),
    "min": float(np.min(predictions)),
    "p01": float(np.percentile(predictions, 1)),
    "p05": float(np.percentile(predictions, 5)),
    "median": float(np.median(predictions)),
    "p95": float(np.percentile(predictions, 95)),
    "p99": float(np.percentile(predictions, 99)),
    "max": float(np.max(predictions)),
}

for k, v in prediction_stats.items():
    print(f"{k:>10}: {v:.4f}")


# ============================================================
# PER-DISCOUNT PERFORMANCE
# ============================================================

print("\n" + "=" * 70)
print("4. PER-DISCOUNT PERFORMANCE")
print("=" * 70)

discount_results = {}

for d in DISCOUNTS:

    mask = test_mask & (discount == d)

    if mask.sum() == 0:
        continue

    actual = y[mask]
    pred = predictions[mask]

    mae = mean_absolute_error(actual, pred)
    rmse = np.sqrt(mean_squared_error(actual, pred))

    discount_results[int(d)] = {
        "samples": int(mask.sum()),
        "actual_mean": float(np.mean(actual)),
        "predicted_mean": float(np.mean(pred)),
        "MAE": float(mae),
        "RMSE": float(rmse),
        "bias": float(np.mean(pred - actual)),
    }

    print(
        f"{d:>2}% | "
        f"N={mask.sum():5d} | "
        f"Actual={np.mean(actual):10.4f} | "
        f"Pred={np.mean(pred):10.4f} | "
        f"MAE={mae:10.4f} | "
        f"RMSE={rmse:10.4f}"
    )


# ============================================================
# COUNTERFACTUAL DISCOUNT RESPONSE
# ============================================================

print("\n" + "=" * 70)
print("5. COUNTERFACTUAL DISCOUNT RESPONSE")
print("=" * 70)

# Use TEST states only.
test_indices = np.where(test_mask)[0]

# Limit to a manageable deterministic sample.
sample_size = min(2000, len(test_indices))

rng = np.random.default_rng(42)

sample_indices = rng.choice(
    test_indices,
    size=sample_size,
    replace=False,
)

sample_states = torch.tensor(
    state[sample_indices],
    dtype=torch.float32,
)

counterfactual_means = {}

counterfactual_matrix = []

with torch.no_grad():

    for d in DISCOUNTS:

        actions = torch.full(
            (sample_size,),
            float(d),
            dtype=torch.float32,
        )

        pred = model.predict(
            sample_states,
            actions,
        ).cpu().numpy()

        counterfactual_matrix.append(pred)

        counterfactual_means[d] = {
            "mean": float(np.mean(pred)),
            "median": float(np.median(pred)),
            "std": float(np.std(pred)),
            "min": float(np.min(pred)),
            "max": float(np.max(pred)),
        }

        print(
            f"{d:>2}% -> "
            f"mean growth={np.mean(pred):10.4f} | "
            f"median={np.median(pred):10.4f}"
        )

counterfactual_matrix = np.array(counterfactual_matrix)


# ============================================================
# DISCOUNT RESPONSE SLOPE
# ============================================================

print("\n" + "=" * 70)
print("6. DISCOUNT RESPONSE CHECK")
print("=" * 70)

mean_curve = np.array(
    [counterfactual_means[d]["mean"] for d in DISCOUNTS]
)

differences = np.diff(mean_curve)

print("Mean predicted-growth differences:")
for i, diff in enumerate(differences):
    print(
        f"{DISCOUNTS[i]:>2}% -> "
        f"{DISCOUNTS[i+1]:>2}% : "
        f"{diff:+.6f}"
    )

monotonic_increasing = bool(np.all(differences >= 0))

print(
    f"\nMean counterfactual response monotonic increasing: "
    f"{monotonic_increasing}"
)


# ============================================================
# PRODUCT-LEVEL COUNTERFACTUAL STABILITY
# ============================================================

print("\n" + "=" * 70)
print("7. PRODUCT-LEVEL COUNTERFACTUAL CHECK")
print("=" * 70)

# Count products whose predicted growth decreases
# when moving to a higher discount.

decrease_counts = {}

for j in range(len(DISCOUNTS) - 1):

    lower = counterfactual_matrix[j]
    higher = counterfactual_matrix[j + 1]

    decreases = np.sum(higher < lower)

    decrease_counts[
        f"{DISCOUNTS[j]}->{DISCOUNTS[j+1]}"
    ] = int(decreases)

    percentage = 100 * decreases / sample_size

    print(
        f"{DISCOUNTS[j]:>2}% -> {DISCOUNTS[j+1]:>2}% : "
        f"{decreases:5d}/{sample_size} "
        f"({percentage:6.2f}%) decrease"
    )


# ============================================================
# EXTREME ERROR ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("8. EXTREME ERROR ANALYSIS")
print("=" * 70)

test_actual = y[test_mask]
test_pred = predictions[test_mask]

test_error = np.abs(test_actual - test_pred)

quantiles = [50, 75, 90, 95, 99, 99.5, 99.9]

for q in quantiles:
    print(
        f"Absolute error P{q:4.1f}: "
        f"{np.percentile(test_error, q):.4f}"
    )

large_error_threshold = np.percentile(test_error, 99)

large_error_mask = test_error >= large_error_threshold

print(
    f"\nTop 1% error threshold: {large_error_threshold:.4f}"
)

print(
    f"Top 1% errors account for "
    f"{large_error_mask.sum()} / {len(test_error)} "
    f"test samples."
)


# ============================================================
# BASELINE COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("9. SIMPLE BASELINE")
print("=" * 70)

# Training mean baseline.
train_mean = np.mean(y[train_mask])

baseline_pred = np.full(
    test_mask.sum(),
    train_mean,
)

baseline_mae = mean_absolute_error(
    test_actual,
    baseline_pred,
)

baseline_rmse = np.sqrt(
    mean_squared_error(
        test_actual,
        baseline_pred,
    )
)

baseline_r2 = r2_score(
    test_actual,
    baseline_pred,
)

print(f"Training mean: {train_mean:.4f}")
print(f"Baseline MAE:  {baseline_mae:.4f}")
print(f"Baseline RMSE: {baseline_rmse:.4f}")
print(f"Baseline R²:   {baseline_r2:.6f}")

print("\nPredictor vs baseline:")
print(
    f"MAE improvement: "
    f"{100 * (baseline_mae - metrics['test']['MAE']) / baseline_mae:.2f}%"
)

print(
    f"RMSE improvement: "
    f"{100 * (baseline_rmse - metrics['test']['RMSE']) / baseline_rmse:.2f}%"
)


# ============================================================
# SAVE RESULTS
# ============================================================

results = {
    "state_dimension": int(state.shape[1]),
    "samples": int(len(state)),
    "target": target_col,
    "discount_bins": DISCOUNTS,
    "target_distribution": target_stats,
    "prediction_distribution": prediction_stats,
    "metrics": metrics,
    "per_discount_test": discount_results,
    "counterfactual_mean_curve": counterfactual_means,
    "counterfactual_monotonic_mean": monotonic_increasing,
    "counterfactual_decrease_counts": decrease_counts,
    "baseline": {
        "train_mean": float(train_mean),
        "MAE": float(baseline_mae),
        "RMSE": float(baseline_rmse),
        "R2": float(baseline_r2),
    },
}

output_path = os.path.join(
    OUTPUT_DIR,
    "sales_predictor_diagnostic.json",
)

with open(output_path, "w") as f:
    json.dump(results, f, indent=2)

print("\n" + "=" * 70)
print("DIAGNOSTIC COMPLETE")
print("=" * 70)
print(f"Saved: {output_path}")