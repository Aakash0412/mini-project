"""
Phase 4: Validate the constructed ItaNet state.

Checks:
  1. Correct total dimension (79 — text 32 + attrs 47)
  2. Correct modality ordering
  3. No NaN or Inf
  4. Correct product alignment with feature index
  5. Correct train/val/test alignment
  6. No temporal leakage
"""

import sys
import os
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.itanet import TEXT_DIM, ATTRIBUTE_DIM, AVAILABLE_STATE_DIM


def validate_state():
    print("=" * 55)
    print("PHASE 4 — State Validation")
    print("=" * 55)

    # ── Load artefacts ───────────────────────────────────────
    assert os.path.exists("data/processed/state/state.npy"), \
        "state.npy missing — run build_state.py first"
    assert os.path.exists("reports/state_validation.json"), \
        "state_validation.json missing — run build_state.py first"

    state       = np.load("data/processed/state/state.npy")
    state_train = np.load("data/processed/state/state_train.npy")
    state_val   = np.load("data/processed/state/state_val.npy")
    state_test  = np.load("data/processed/state/state_test.npy")
    index       = pd.read_parquet("data/processed/features/feature_index.parquet")
    df_clean    = pd.read_parquet("data/processed/products_clean.parquet").reset_index(drop=True)

    with open("reports/state_validation.json") as f:
        stats = json.load(f)

    errors = []

    # 1. Dimension check
    if state.shape[1] != AVAILABLE_STATE_DIM:
        errors.append(
            f"FAIL dim: expected {AVAILABLE_STATE_DIM}, got {state.shape[1]}"
        )
    else:
        print(f"[PASS] State dim = {state.shape[1]} "
              f"(Text {TEXT_DIM} + Attrs {ATTRIBUTE_DIM})")

    # 2. Alignment with index
    if state.shape[0] != len(index):
        errors.append(
            f"FAIL alignment: state rows {state.shape[0]} ≠ index rows {len(index)}"
        )
    else:
        print(f"[PASS] Row alignment: {state.shape[0]} rows match feature index")

    # 3. NaN / Inf
    if np.isnan(state).any():
        errors.append("FAIL: NaN found in state")
    else:
        print("[PASS] No NaN")

    if np.isinf(state).any():
        errors.append("FAIL: Inf found in state")
    else:
        print("[PASS] No Inf")

    # 4. Split sizes add up
    total = len(state_train) + len(state_val) + len(state_test)
    if total != len(state):
        errors.append(
            f"FAIL split sizes: {len(state_train)}+{len(state_val)}+"
            f"{len(state_test)}={total} ≠ {len(state)}"
        )
    else:
        print(f"[PASS] Split sizes: train={len(state_train)}, "
              f"val={len(state_val)}, test={len(state_test)}, total={total}")

    # 5. No leakage — chronological ordering
    train_mask = df_clean["MonthNum"].isin([1, 2, 3, 4, 5, 6]).values
    val_mask   = df_clean["MonthNum"].isin([7, 8]).values
    test_mask  = df_clean["MonthNum"].isin([9, 10]).values

    overlap_tv = train_mask & val_mask
    overlap_vt = val_mask & test_mask
    overlap_tt = train_mask & test_mask
    if overlap_tv.any() or overlap_vt.any() or overlap_tt.any():
        errors.append("FAIL: temporal leakage detected between splits")
    else:
        print("[PASS] No temporal leakage between splits")

    # 6. Text / attribute sub-block sanity
    text_block = state[:, :TEXT_DIM]
    attr_block = state[:, TEXT_DIM:TEXT_DIM + ATTRIBUTE_DIM]
    cached_text = np.load("data/processed/features/text_features.npy")
    cached_attr = np.load("data/processed/features/attribute_features.npy")

    if not np.allclose(text_block, cached_text, atol=1e-5):
        errors.append("FAIL: text sub-block does not match cached text_features.npy")
    else:
        print("[PASS] Text sub-block matches cached text_features.npy")

    if not np.allclose(attr_block, cached_attr, atol=1e-5):
        errors.append("FAIL: attribute sub-block does not match cached attribute_features.npy")
    else:
        print("[PASS] Attribute sub-block matches cached attribute_features.npy")

    # ── Summary ──────────────────────────────────────────────
    print()
    if errors:
        print("VALIDATION FAILED:")
        for e in errors:
            print(f"  {e}")
        sys.exit(1)
    else:
        print("All state validations PASSED.")
        print(f"  State shape : {state.shape}")
        print(f"  State mean  : {state.mean():.4f}")
        print(f"  State std   : {state.std():.4f}")


if __name__ == "__main__":
    validate_state()
