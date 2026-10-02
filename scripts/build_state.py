"""
Phase 4 — Build the ItaNet-111 State Vector.

Current implementation:
    Image      : 32-D
    Text       : 32-D
    Attributes : 47-D
    Sentiment  : unavailable

Total:
    32 + 32 + 47 = 111-D

The original paper defines a 121-D state by additionally including
10-D review sentiment. The public dataset does not provide the
required review text/database, so sentiment is intentionally omitted.

Image features have already been extracted locally using ResNet-50
and are loaded from:
    data/processed/features/image_features.npy

Rows with unavailable image features are preserved in the master
row alignment but marked invalid for complete-state predictor
training.
"""

import sys
import os
import json

import numpy as np
import pandas as pd

# Allow importing from src/
sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from src.models.itanet import (
    ItaNetStateBuilder,
    PAPER_STATE_DIM,
    AVAILABLE_STATE_DIM,
)


# ─────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────

FEATURE_DIR = "data/processed/features"
STATE_DIR = "data/processed/state"

TEXT_PATH = os.path.join(FEATURE_DIR, "text_features.npy")
ATTRIBUTE_PATH = os.path.join(FEATURE_DIR, "attribute_features.npy")
IMAGE_PATH = os.path.join(FEATURE_DIR, "image_features.npy")

INDEX_PATH = os.path.join(FEATURE_DIR, "feature_index.parquet")
CLEAN_DATA_PATH = "data/processed/products_clean.parquet"


def build_state():
    print("=" * 60)
    print("PHASE 4 — ItaNet-111 State Construction")
    print("=" * 60)

    os.makedirs(STATE_DIR, exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    # ─────────────────────────────────────────────────────────
    # Load feature arrays
    # ─────────────────────────────────────────────────────────

    print("\nLoading feature arrays...")

    text_features = np.load(TEXT_PATH)
    attribute_features = np.load(ATTRIBUTE_PATH)
    image_features = np.load(IMAGE_PATH)

    print(f"  Text       : {text_features.shape}")
    print(f"  Attributes : {attribute_features.shape}")
    print(f"  Image      : {image_features.shape}")

    # ─────────────────────────────────────────────────────────
    # Basic alignment validation
    # ─────────────────────────────────────────────────────────

    n_rows = len(text_features)

    if len(attribute_features) != n_rows:
        raise ValueError(
            "Feature row mismatch: text and attributes have "
            f"different row counts: "
            f"{len(text_features)} vs {len(attribute_features)}"
        )

    if len(image_features) != n_rows:
        raise ValueError(
            "Feature row mismatch: image features contain "
            f"{len(image_features)} rows but expected {n_rows}"
        )

    # ─────────────────────────────────────────────────────────
    # Detect incomplete image rows
    # ─────────────────────────────────────────────────────────

    image_valid_mask = np.isfinite(image_features).all(axis=1)

    invalid_image_mask = ~image_valid_mask

    valid_count = int(image_valid_mask.sum())
    invalid_count = int(invalid_image_mask.sum())

    print("\nImage feature validity:")
    print(f"  Valid rows   : {valid_count}")
    print(f"  Invalid rows : {invalid_count}")

    if invalid_count:
        invalid_indices = np.flatnonzero(invalid_image_mask)

        print(
            "  Invalid row indices:",
            invalid_indices.tolist()
        )

    # ─────────────────────────────────────────────────────────
    # Build state only from complete rows
    # ─────────────────────────────────────────────────────────
    #
    # ItaNetStateBuilder intentionally rejects NaNs.
    # Therefore we construct the complete-state matrix only from
    # rows with valid image features.
    #
    # The original row indices are preserved separately so that
    # downstream training/evaluation can remain aligned with the
    # source dataset.

    print("\nConstructing 111-D state...")

    builder = ItaNetStateBuilder(
        image_features=image_features[image_valid_mask],
        text_features=text_features[image_valid_mask],
        attribute_features=attribute_features[image_valid_mask],
        sentiment_features=None,
    )

    report = builder.modality_report()

    print("\nModality report:")
    for key, value in report.items():
        print(f"  {key}: {value}")

    state_valid = builder.build()

    print(f"\nComplete state shape: {state_valid.shape}")

    if state_valid.shape[1] != AVAILABLE_STATE_DIM:
        raise ValueError(
            f"Expected {AVAILABLE_STATE_DIM}-D state, "
            f"got {state_valid.shape[1]}-D"
        )

    # ─────────────────────────────────────────────────────────
    # Save complete state
    # ─────────────────────────────────────────────────────────

    np.save(
        os.path.join(STATE_DIR, "state.npy"),
        state_valid,
    )

    print(
        f"Saved: {os.path.join(STATE_DIR, 'state.npy')}"
    )

    # Preserve source row identity.
    valid_row_indices = np.flatnonzero(image_valid_mask)

    np.save(
        os.path.join(STATE_DIR, "state_row_indices.npy"),
        valid_row_indices,
    )

    print(
        f"Saved: {os.path.join(STATE_DIR, 'state_row_indices.npy')}"
    )

    # ─────────────────────────────────────────────────────────
    # Load source index and cleaned dataset
    # ─────────────────────────────────────────────────────────

    index = pd.read_parquet(INDEX_PATH)

    if len(index) != n_rows:
        raise ValueError(
            f"Feature index contains {len(index)} rows, "
            f"expected {n_rows}"
        )

    df_clean = pd.read_parquet(
        CLEAN_DATA_PATH
    ).reset_index(drop=True)

    if len(df_clean) != n_rows:
        raise ValueError(
            f"Clean dataset contains {len(df_clean)} rows, "
            f"expected {n_rows}"
        )

    if not (index["ASIN"] == df_clean["ASIN"]).all():
        raise AssertionError(
            "ASIN alignment mismatch between feature index and clean dataset"
        )

    # ─────────────────────────────────────────────────────────
    # Chronological split masks
    # ─────────────────────────────────────────────────────────

    train_mask_all = df_clean["MonthNum"].isin(
        [1, 2, 3, 4, 5, 6]
    ).values

    val_mask_all = df_clean["MonthNum"].isin(
        [7, 8]
    ).values

    test_mask_all = df_clean["MonthNum"].isin(
        [9, 10]
    ).values

    # Combine chronological split with complete-state validity.
    train_mask = train_mask_all & image_valid_mask
    val_mask = val_mask_all & image_valid_mask
    test_mask = test_mask_all & image_valid_mask

    # ─────────────────────────────────────────────────────────
    # Convert full-row masks to state-row positions
    # ─────────────────────────────────────────────────────────

    state_train_mask = train_mask[image_valid_mask]
    state_val_mask = val_mask[image_valid_mask]
    state_test_mask = test_mask[image_valid_mask]

    state_train = state_valid[state_train_mask]
    state_val = state_valid[state_val_mask]
    state_test = state_valid[state_test_mask]

    # ─────────────────────────────────────────────────────────
    # Save split-aligned states
    # ─────────────────────────────────────────────────────────

    np.save(
        os.path.join(STATE_DIR, "state_train.npy"),
        state_train,
    )

    np.save(
        os.path.join(STATE_DIR, "state_val.npy"),
        state_val,
    )

    np.save(
        os.path.join(STATE_DIR, "state_test.npy"),
        state_test,
    )

    print("\nSplit state shapes:")
    print(f"  Train : {state_train.shape}")
    print(f"  Val   : {state_val.shape}")
    print(f"  Test  : {state_test.shape}")

    # ─────────────────────────────────────────────────────────
    # Save validity information
    # ─────────────────────────────────────────────────────────

    validity_df = pd.DataFrame(
        {
            "row_index": np.arange(n_rows),
            "ASIN": df_clean["ASIN"].values,
            "image_features_valid": image_valid_mask,
            "complete_state_valid": image_valid_mask,
            "invalid_reason": np.where(image_valid_mask, "None", "Missing Image Feature"),
            "split": np.select(
                [
                    train_mask_all,
                    val_mask_all,
                    test_mask_all,
                ],
                [
                    "train",
                    "validation",
                    "test",
                ],
                default="unknown",
            ),
        }
    )

    validity_path = os.path.join(
        STATE_DIR,
        "state_validity.csv",
    )

    validity_df.to_csv(
        validity_path,
        index=False,
    )

    print(f"\nSaved: {validity_path}")

    # ─────────────────────────────────────────────────────────
    # Leakage checks
    # ─────────────────────────────────────────────────────────

    train_months = df_clean.loc[
        train_mask_all,
        "MonthNum"
    ].unique()

    val_months = df_clean.loc[
        val_mask_all,
        "MonthNum"
    ].unique()

    test_months = df_clean.loc[
        test_mask_all,
        "MonthNum"
    ].unique()

    if not (max(train_months) < min(val_months)):
        raise AssertionError(
            "LEAKAGE: train overlaps validation"
        )

    if not (max(val_months) < min(test_months)):
        raise AssertionError(
            "LEAKAGE: validation overlaps test"
        )

    print("\nChronological leakage check PASSED.")

    # ─────────────────────────────────────────────────────────
    # State integrity checks
    # ─────────────────────────────────────────────────────────

    if np.isnan(state_valid).any():
        raise AssertionError(
            "State contains NaN values"
        )

    if np.isinf(state_valid).any():
        raise AssertionError(
            "State contains Inf values"
        )

    # ─────────────────────────────────────────────────────────
    # Validation statistics
    # ─────────────────────────────────────────────────────────

    stats = {
        "implementation": "ItaNet-111",
        "num_source_samples": int(n_rows),
        "num_complete_state_samples": int(len(state_valid)),
        "num_incomplete_samples": int(invalid_count),

        "state_dimension": int(state_valid.shape[1]),
        "paper_state_dimension": int(PAPER_STATE_DIM),
        "available_state_dimension": int(AVAILABLE_STATE_DIM),
        "missing_dimensions": int(
            PAPER_STATE_DIM - state_valid.shape[1]
        ),

        "modalities_included": [
            "Image (32)",
            "Text (32)",
            "Attributes (47)",
        ],

        "modalities_missing": [
            "Sentiment (10)",
        ],

        "train_samples_complete": int(
            len(state_train)
        ),
        "val_samples_complete": int(
            len(state_val)
        ),
        "test_samples_complete": int(
            len(state_test)
        ),

        "train_samples_source": int(
            train_mask_all.sum()
        ),
        "val_samples_source": int(
            val_mask_all.sum()
        ),
        "test_samples_source": int(
            test_mask_all.sum()
        ),

        "has_nan": bool(
            np.isnan(state_valid).any()
        ),
        "has_inf": bool(
            np.isinf(state_valid).any()
        ),

        "leakage_check": "PASSED",

        "state_mean": float(
            state_valid.mean()
        ),
        "state_std": float(
            state_valid.std()
        ),
        "state_min": float(
            state_valid.min()
        ),
        "state_max": float(
            state_valid.max()
        ),
    }

    validation_path = "reports/state_validation.json"

    with open(validation_path, "w") as f:
        json.dump(stats, f, indent=4)

    print(f"Saved: {validation_path}")

    # ─────────────────────────────────────────────────────────
    # Human-readable report
    # ─────────────────────────────────────────────────────────

    statistics_path = "reports/state_statistics.md"

    with open(statistics_path, "w") as f:
        f.write("# ItaNet-111 State Statistics\n\n")

        f.write(
            "The current implementation uses Image + Text + "
            "Attributes. The 10-D review sentiment modality is "
            "not included because the required review data is "
            "not available in the released dataset.\n\n"
        )

        f.write("| Metric | Value |\n")
        f.write("|---|---|\n")

        for key, value in stats.items():
            f.write(f"| {key} | {value} |\n")

    print(f"Saved: {statistics_path}")

    print("\n" + "=" * 60)
    print("PHASE 4 — ItaNet-111 state construction complete.")
    print("=" * 60)

    return state_valid


if __name__ == "__main__":
    build_state()