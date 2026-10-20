
from __future__ import annotations

import numpy as np

from src.environment.counterfactual_simulator import (
    CounterfactualSimulator,
    DISCOUNT_BINS,
)


def isotonic_non_decreasing(values: np.ndarray) -> np.ndarray:
    """Least-squares projection onto non-decreasing sequences."""
    values = np.asarray(values, dtype=np.float64)

    if values.ndim != 1 or len(values) == 0:
        raise ValueError("values must be a non-empty 1-D array.")
    if not np.all(np.isfinite(values)):
        raise ValueError("values contain NaN or Inf.")

    means = []
    weights = []

    for value in values:
        means.append(float(value))
        weights.append(1)

        while len(means) >= 2 and means[-2] > means[-1]:
            total_weight = weights[-2] + weights[-1]
            merged_mean = (
                means[-2] * weights[-2]
                + means[-1] * weights[-1]
            ) / total_weight

            means[-2:] = [merged_mean]
            weights[-2:] = [total_weight]

    projected = []
    for mean, weight in zip(means, weights):
        projected.extend([mean] * weight)

    return np.asarray(projected, dtype=np.float32)


class MonotoneCounterfactualSimulator:
    """
    Experimental wrapper around the frozen sales predictor.

    Candidate-discount predictions are projected onto a non-decreasing
    curve. This is an explicit modeling assumption, not an observed
    causal relationship.
    """

    def __init__(self, base_simulator=None):
        self.base = (
            base_simulator
            if base_simulator is not None
            else CounterfactualSimulator()
        )
        self.discount_bins = list(DISCOUNT_BINS)

    @property
    def state_dim(self):
        return self.base.state_dim

    @property
    def input_dim(self):
        return self.base.input_dim

    @property
    def config(self):
        return self.base.config

    @property
    def checkpoint_dir(self):
        return self.base.checkpoint_dir

    @property
    def device(self):
        return self.base.device

    def predict_all_discounts(self, state):
        raw = self.base.predict_all_discounts(state)
        values = np.asarray(
            [raw[d] for d in self.discount_bins],
            dtype=np.float32,
        )
        projected = isotonic_non_decreasing(values)

        return {
            discount: float(value)
            for discount, value in zip(
                self.discount_bins, projected
            )
        }

    def predict(self, state, discount):
        discount = float(discount)

        if discount not in self.discount_bins:
            raise ValueError(
                f"Unsupported discount {discount}; "
                f"expected one of {self.discount_bins}."
            )

        return self.predict_all_discounts(state)[int(discount)]

    def predict_batch(self, states, discounts):
        states = np.asarray(states, dtype=np.float32)
        discounts = np.asarray(discounts, dtype=np.float32)

        if states.ndim != 2 or states.shape[1] != self.state_dim:
            raise ValueError(
                f"states must have shape (B, {self.state_dim})."
            )
        if discounts.ndim != 1 or len(discounts) != len(states):
            raise ValueError("discounts must match the state batch.")
        if not np.all(np.isfinite(states)):
            raise ValueError("states contain NaN or Inf.")

        return np.asarray([
            self.predict(state, float(discount))
            for state, discount in zip(states, discounts)
        ], dtype=np.float32)

    def summary(self):
        return (
            "MonotoneCounterfactualSimulator\n"
            f"  base checkpoint: {self.checkpoint_dir}\n"
            f"  discount bins: {self.discount_bins}\n"
            "  response constraint: non-decreasing\n"
            "  status: experimental assumption"
        )

