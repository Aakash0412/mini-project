"""
tests/test_phase_8_counterfactual_simulator.py

Unit tests for Phase 8: CounterfactualSimulator
================================================

Tests cover:
  1. Module imports cleanly.
  2. CounterfactualSimulator instantiates from real checkpoint.
  3. Predictor input_dim is confirmed as 112 at runtime.
  4. predict() returns a finite float for each of the 9 discount bins.
  5. predict_all_discounts() returns exactly 9 entries covering all bins.
  6. Predictions are not all identical (model is non-constant).
  7. Batch prediction matches single-sample predictions.
  8. predict() rejects invalid state dimensions.
  9. predict() rejects invalid discount values.
 10. predict() rejects states with NaN values.
 11. predict_batch() rejects mismatched lengths.
 12. predict_batch() rejects invalid discount values.
 13. summary() returns a non-empty string.
 14. discount_bins property matches DISCOUNT_BINS constant.
"""

import os
import sys
import pytest
import numpy as np

# ---------------------------------------------------------------------------
# Path setup — allow running from any working directory
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.environment.counterfactual_simulator import (
    CounterfactualSimulator,
    DISCOUNT_BINS,
    STATE_DIM,
    INPUT_DIM,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

CHECKPOINT_DIR = os.path.join(_ROOT, "models", "sales_predictor")

REQUIRES_CHECKPOINT = pytest.mark.skipif(
    not os.path.exists(os.path.join(CHECKPOINT_DIR, "best_model.pth")),
    reason="Sales predictor checkpoint not found; run scripts/train_sales_predictor.py first.",
)

STATE_FILE = os.path.join(_ROOT, "data", "processed", "state", "state_train.npy")

REQUIRES_STATE = pytest.mark.skipif(
    not os.path.exists(STATE_FILE),
    reason="state_train.npy not found; run scripts/build_state.py first.",
)


@pytest.fixture(scope="module")
def simulator():
    """Shared simulator instance across all tests (loaded once)."""
    return CounterfactualSimulator(checkpoint_dir=CHECKPOINT_DIR)


@pytest.fixture(scope="module")
def sample_state():
    """Load the first valid state row from the training split."""
    arr = np.load(STATE_FILE)
    return arr[0].astype(np.float32)


# ---------------------------------------------------------------------------
# Test 1 — Import
# ---------------------------------------------------------------------------

def test_import():
    """Module and key constants import without error."""
    assert DISCOUNT_BINS == [0, 5, 10, 15, 20, 25, 30, 35, 40]
    assert STATE_DIM == 111
    assert INPUT_DIM == 112


# ---------------------------------------------------------------------------
# Test 2 — Instantiation
# ---------------------------------------------------------------------------

@REQUIRES_CHECKPOINT
def test_instantiation(simulator):
    """CounterfactualSimulator instantiates without error."""
    assert simulator is not None


# ---------------------------------------------------------------------------
# Test 3 — Input dimension confirmed at runtime
# ---------------------------------------------------------------------------

@REQUIRES_CHECKPOINT
def test_input_dim(simulator):
    """Predictor input_dim is 112 (111 state + 1 discount)."""
    assert simulator.input_dim == 112


# ---------------------------------------------------------------------------
# Test 4 — predict() returns finite float for each discount
# ---------------------------------------------------------------------------

@REQUIRES_CHECKPOINT
@REQUIRES_STATE
@pytest.mark.parametrize("discount", DISCOUNT_BINS)
def test_predict_single_discount(simulator, sample_state, discount):
    """predict() returns a finite float for every valid discount level."""
    result = simulator.predict(sample_state, discount)
    assert isinstance(result, float), f"Expected float, got {type(result)}"
    assert np.isfinite(result), f"Prediction for discount={discount} is not finite: {result}"


# ---------------------------------------------------------------------------
# Test 5 — predict_all_discounts() covers all 9 bins
# ---------------------------------------------------------------------------

@REQUIRES_CHECKPOINT
@REQUIRES_STATE
def test_predict_all_discounts_keys(simulator, sample_state):
    """predict_all_discounts() returns exactly the 9 paper-defined discount bins."""
    result = simulator.predict_all_discounts(sample_state)
    assert isinstance(result, dict)
    assert set(result.keys()) == set(DISCOUNT_BINS), (
        f"Keys mismatch. Expected {DISCOUNT_BINS}, got {sorted(result.keys())}"
    )


# ---------------------------------------------------------------------------
# Test 6 — Predictions are not all identical
# ---------------------------------------------------------------------------

@REQUIRES_CHECKPOINT
@REQUIRES_STATE
def test_predictions_not_constant(simulator, sample_state):
    """Predictions across different discounts must have non-zero variance."""
    result = simulator.predict_all_discounts(sample_state)
    values = list(result.values())
    assert np.std(values) > 0, (
        "All predictions are identical — the model is not responding to discount changes."
    )


# ---------------------------------------------------------------------------
# Test 7 — Batch vs single consistency
# ---------------------------------------------------------------------------

@REQUIRES_CHECKPOINT
@REQUIRES_STATE
def test_batch_matches_single(simulator, sample_state):
    """predict_batch() must match predict() for the same (state, discount) pairs."""
    states_batch = np.stack([sample_state] * len(DISCOUNT_BINS))  # (9, 111)
    discounts_batch = np.array(DISCOUNT_BINS, dtype=np.float32)   # (9,)

    batch_preds = simulator.predict_batch(states_batch, discounts_batch)
    single_preds = [simulator.predict(sample_state, d) for d in DISCOUNT_BINS]

    np.testing.assert_allclose(
        batch_preds,
        np.array(single_preds, dtype=np.float32),
        rtol=1e-5,
        err_msg="Batch predictions do not match single-sample predictions.",
    )


# ---------------------------------------------------------------------------
# Test 8 — Reject wrong state dimension
# ---------------------------------------------------------------------------

@REQUIRES_CHECKPOINT
def test_rejects_wrong_state_dim(simulator):
    """predict() raises ValueError if state is not (111,)."""
    bad_state = np.ones(80, dtype=np.float32)
    with pytest.raises(ValueError, match="111"):
        simulator.predict(bad_state, discount=10)


# ---------------------------------------------------------------------------
# Test 9 — Reject invalid discount
# ---------------------------------------------------------------------------

@REQUIRES_CHECKPOINT
@REQUIRES_STATE
def test_rejects_invalid_discount(simulator, sample_state):
    """predict() raises ValueError for a discount not in DISCOUNT_BINS."""
    with pytest.raises(ValueError, match="DISCOUNT_BINS"):
        simulator.predict(sample_state, discount=7)


# ---------------------------------------------------------------------------
# Test 10 — Reject NaN state
# ---------------------------------------------------------------------------

@REQUIRES_CHECKPOINT
def test_rejects_nan_state(simulator):
    """predict() raises ValueError if state contains NaN."""
    nan_state = np.full(STATE_DIM, np.nan, dtype=np.float32)
    with pytest.raises(ValueError, match="NaN"):
        simulator.predict(nan_state, discount=10)


# ---------------------------------------------------------------------------
# Test 11 — Batch: mismatched lengths
# ---------------------------------------------------------------------------

@REQUIRES_CHECKPOINT
@REQUIRES_STATE
def test_batch_rejects_mismatched_lengths(simulator, sample_state):
    """predict_batch() raises ValueError if states and discounts have different lengths."""
    states = np.stack([sample_state] * 3)   # (3, 111)
    discounts = np.array([10, 20], dtype=np.float32)  # (2,) — mismatch
    with pytest.raises(ValueError):
        simulator.predict_batch(states, discounts)


# ---------------------------------------------------------------------------
# Test 12 — Batch: invalid discounts
# ---------------------------------------------------------------------------

@REQUIRES_CHECKPOINT
@REQUIRES_STATE
def test_batch_rejects_invalid_discounts(simulator, sample_state):
    """predict_batch() raises ValueError for unsupported discount values."""
    states = np.stack([sample_state] * 3)
    discounts = np.array([10, 17, 20], dtype=np.float32)  # 17 is invalid
    with pytest.raises(ValueError, match="Invalid discount"):
        simulator.predict_batch(states, discounts)


# ---------------------------------------------------------------------------
# Test 13 — summary()
# ---------------------------------------------------------------------------

@REQUIRES_CHECKPOINT
def test_summary_non_empty(simulator):
    """summary() returns a non-empty string."""
    s = simulator.summary()
    assert isinstance(s, str)
    assert len(s) > 50
    assert "111" in s   # state dim
    assert "112" in s   # input dim
    assert "0%" in s or "0 %" in s or "0" in s  # 0% out-of-support note


# ---------------------------------------------------------------------------
# Test 14 — discount_bins property
# ---------------------------------------------------------------------------

@REQUIRES_CHECKPOINT
def test_discount_bins_property(simulator):
    """discount_bins property returns the full paper-defined action space."""
    assert simulator.discount_bins == DISCOUNT_BINS
