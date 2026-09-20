"""
Phase 4: Build the ItaNet 121-D (or available-D) state vector.

Loads cached feature arrays and uses ItaNetStateBuilder to concatenate
them in the paper-defined order. Saves the resulting state matrix and
produces validation artefacts.
"""

import sys
import os
import json
import numpy as np
import pandas as pd

# Allow importing from src/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.itanet import ItaNetStateBuilder, PAPER_STATE_DIM


def build_state():
    print("=" * 55)
    print("PHASE 4 — ItaNet State Construction")
    print("=" * 55)

    os.makedirs("data/processed/state", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    # ── Load available feature arrays ───────────────────────
    text_features = np.load("data/processed/features/text_features.npy")
    attribute_features = np.load("data/processed/features/attribute_features.npy")

    # Image and sentiment features are not available from this dataset.
    # They are NOT loaded or fabricated. See docs/STATE_CONSTRUCTION.md.
    image_features = None       # UNAVAILABLE — no image files in dataset
    sentiment_features = None   # UNAVAILABLE — no review text in dataset

    # ── Build state ──────────────────────────────────────────
    builder = ItaNetStateBuilder(
        text_features=text_features,
        attribute_features=attribute_features,
        image_features=image_features,
        sentiment_features=sentiment_features,
    )

    report = builder.modality_report()
    print("\nModality report:")
    for k, v in report.items():
        print(f"  {k}: {v}")

    state = builder.build()
    print(f"\nState array shape: {state.shape}")
    print(f"  → Available state dim : {state.shape[1]}")
    print(f"  → Paper target dim    : {PAPER_STATE_DIM} (121)")
    print(f"  → Missing dims        : {report['missing_dims']} (Image 32 + Sentiment 10)")

    # ── Save state ───────────────────────────────────────────
    np.save("data/processed/state/state.npy", state)
    print("\nSaved: data/processed/state/state.npy")

    # ── Load index for alignment check ───────────────────────
    index = pd.read_parquet("data/processed/features/feature_index.parquet")
    assert len(state) == len(index), (
        f"State length {len(state)} does not match index length {len(index)}"
    )

    # ── Save split-aligned state arrays ─────────────────────
    df_clean = pd.read_parquet("data/processed/products_clean.parquet").reset_index(drop=True)

    train_mask = df_clean["MonthNum"].isin([1, 2, 3, 4, 5, 6]).values
    val_mask   = df_clean["MonthNum"].isin([7, 8]).values
    test_mask  = df_clean["MonthNum"].isin([9, 10]).values

    np.save("data/processed/state/state_train.npy", state[train_mask])
    np.save("data/processed/state/state_val.npy",   state[val_mask])
    np.save("data/processed/state/state_test.npy",  state[test_mask])

    print(f"Saved train state: {state[train_mask].shape}")
    print(f"Saved val   state: {state[val_mask].shape}")
    print(f"Saved test  state: {state[test_mask].shape}")

    # ── Per-split leakage check ──────────────────────────────
    train_months = df_clean.loc[train_mask, "MonthNum"].unique()
    val_months   = df_clean.loc[val_mask,   "MonthNum"].unique()
    test_months  = df_clean.loc[test_mask,  "MonthNum"].unique()
    assert max(train_months) < min(val_months),   "LEAKAGE: train overlaps val"
    assert max(val_months)   < min(test_months),  "LEAKAGE: val overlaps test"
    print("\nLeakage check PASSED.")

    # ── Save validation JSON ─────────────────────────────────
    stats = {
        "num_samples":          int(state.shape[0]),
        "state_dimension":      int(state.shape[1]),
        "paper_state_dimension": PAPER_STATE_DIM,
        "missing_dims":          int(report["missing_dims"]),
        "modalities_included":  ["Text (32)", "Attributes (47)"],
        "modalities_missing":   ["Image (32)", "Sentiment (10)"],
        "train_samples":        int(state[train_mask].shape[0]),
        "val_samples":          int(state[val_mask].shape[0]),
        "test_samples":         int(state[test_mask].shape[0]),
        "has_nan":              bool(np.isnan(state).any()),
        "has_inf":              bool(np.isinf(state).any()),
        "leakage_check":        "PASSED",
        "state_mean":           float(state.mean()),
        "state_std":            float(state.std()),
        "state_min":            float(state.min()),
        "state_max":            float(state.max()),
    }

    with open("reports/state_validation.json", "w") as f:
        json.dump(stats, f, indent=4)
    print("Saved: reports/state_validation.json")

    # ── Save statistics markdown ─────────────────────────────
    with open("reports/state_statistics.md", "w") as f:
        f.write("# State Statistics — Phase 4\n\n")
        f.write(f"| Metric | Value |\n|---|---|\n")
        for k, v in stats.items():
            f.write(f"| {k} | {v} |\n")
    print("Saved: reports/state_statistics.md")
    print("\nPhase 4 complete.")


if __name__ == "__main__":
    build_state()
