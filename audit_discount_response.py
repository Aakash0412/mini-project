
from pathlib import Path
import json
import numpy as np
import pandas as pd

from src.environment.counterfactual_simulator import (
    CounterfactualSimulator,
    DISCOUNT_BINS,
)

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data/processed/products_clean.parquet"
STATE_PATH = ROOT / "data/processed/state/state.npy"
INDEX_PATH = ROOT / "data/processed/state/state_row_indices.npy"
OUTPUT_DIR = ROOT / "reports/paper_alignment/discount_response_audit"

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(DATA_PATH)
    states = np.load(STATE_PATH)
    indices = np.load(INDEX_PATH)

    if len(states) != len(indices):
        raise ValueError("State/index length mismatch.")
    if states.ndim != 2 or states.shape[1] != 111:
        raise ValueError(f"Unexpected state shape: {states.shape}")
    if np.any(indices < 0) or np.any(indices >= len(df)):
        raise ValueError("Invalid source row index.")

    aligned = df.iloc[indices].copy().reset_index(drop=True)
    if len(aligned) != len(states):
        raise ValueError("State/product alignment failed.")

    required = [
        "ASIN", "MonthNum", "Discount",
        "Estimated Monthly Sales Growth Rate"
    ]
    missing = [c for c in required if c not in aligned.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Assess whether the recorded target changes with discount
    # within a product-month group.
    grouped = aligned.groupby(["ASIN", "MonthNum"], dropna=False)
    group_stats = grouped.agg(
        row_count=("Discount", "size"),
        discount_count=("Discount", "nunique"),
        target_count=("Estimated Monthly Sales Growth Rate", "nunique"),
        target_min=("Estimated Monthly Sales Growth Rate", "min"),
        target_max=("Estimated Monthly Sales Growth Rate", "max"),
    ).reset_index()

    multi_discount = group_stats[group_stats["discount_count"] > 1]
    target_constant = multi_discount["target_count"] <= 1

    target_audit = {
        "aligned_rows": int(len(aligned)),
        "unique_asins": int(aligned["ASIN"].nunique()),
        "unique_asin_month_groups": int(len(group_stats)),
        "groups_with_multiple_discounts": int(len(multi_discount)),
        "multi_discount_groups_with_constant_target": int(target_constant.sum()),
        "fraction_constant_target_among_multi_discount_groups": (
            float(target_constant.mean()) if len(multi_discount) else None
        ),
        "interpretation": (
            "Observed targets do not identify discount-specific demand "
            "response when the target is constant within multi-discount "
            "product-month groups."
        ),
    }

    group_stats.to_csv(
        OUTPUT_DIR / "product_month_target_audit.csv", index=False
    )

    # Evaluate the frozen predictor at every supported discount.
    # Do not train or modify any model.
    simulator = CounterfactualSimulator()
    rng = np.random.default_rng(42)

    candidate_positions = np.flatnonzero(
        np.all(np.isfinite(states), axis=1)
    )
    if len(candidate_positions) == 0:
        raise ValueError("No finite states available.")

    sample_size = min(500, len(candidate_positions))
    positions = rng.choice(
        candidate_positions, size=sample_size, replace=False
    )

    records = []
    for position in positions:
        state = states[position]
        predictions = simulator.predict_all_discounts(state)
        values = np.array(
            [predictions[d] for d in DISCOUNT_BINS], dtype=float
        )

        if not np.all(np.isfinite(values)):
            raise ValueError(
                f"Non-finite predictions at state position {position}"
            )

        records.append({
            "source_row_index": int(indices[position]),
            "asin": str(df.iloc[indices[position]]["ASIN"]),
            "month": int(df.iloc[indices[position]]["MonthNum"]),
            "prediction_min": float(values.min()),
            "prediction_max": float(values.max()),
            "prediction_range": float(values.max() - values.min()),
            "prediction_std_across_discounts": float(values.std()),
            "prediction_monotonic_non_decreasing": bool(
                np.all(np.diff(values) >= 0)
            ),
            **{
                f"predicted_growth_{discount}": float(prediction)
                for discount, prediction in zip(DISCOUNT_BINS, values)
            },
        })

    prediction_df = pd.DataFrame(records)
    prediction_df.to_csv(
        OUTPUT_DIR / "predictor_discount_sensitivity.csv", index=False
    )

    sensitivity = prediction_df["prediction_range"]
    target_audit.update({
        "predictor_checkpoint_dir": "models/sales_predictor",
        "discount_bins": DISCOUNT_BINS,
        "sampled_states": int(sample_size),
        "median_prediction_range": float(sensitivity.median()),
        "mean_prediction_range": float(sensitivity.mean()),
        "fraction_with_prediction_range_below_1_percentage_point": float(
            (sensitivity < 1.0).mean()
        ),
        "fraction_monotonic_non_decreasing": float(
            prediction_df[
                "prediction_monotonic_non_decreasing"
            ].mean()
        ),
    })

    with open(
        OUTPUT_DIR / "audit_summary.json", "w", encoding="utf-8"
    ) as file:
        json.dump(target_audit, file, indent=2)

    print(json.dumps(target_audit, indent=2))
    print(f"\nReports written to: {OUTPUT_DIR}")
    print("Existing checkpoints and dataset were not modified.")

if __name__ == "__main__":
    main()
