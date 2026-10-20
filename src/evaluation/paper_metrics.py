"""Paper-aligned evaluation metrics (Sec. IV-B, Eqs. 9-14).

Pure functions. No I/O, no model loading, no randomness.

References
----------
Liu et al., IEEE Access 2026.
    Eq.  9: Delta M_{t,i} = (M_{t,i} - M_{t-1,i}) / M_{t-1,i}
    Eq. 10: mean over products of Eq. 9
    Eq. 11: Delta Q_{t,i} = (Q_{t,i} - Q_{t-1,i}) / Q_{t-1,i}
    Eq. 12: Delta Q_{t,i} = g_hat_{t-1,i}     (identity under Eq. 4)
    Eq. 13: mean over products of Eq. 11
    Eq. 14: r_t = alpha * mean(dM) + beta * mean(dQ)
"""

from __future__ import annotations

from typing import Dict, Iterable, Mapping

import numpy as np


def delta_margin(margin_t: np.ndarray, margin_prev: np.ndarray) -> np.ndarray:
    """Eq. 9: per-product profit-margin growth rate."""
    margin_t = np.asarray(margin_t, dtype=np.float64)
    margin_prev = np.asarray(margin_prev, dtype=np.float64)
    if margin_t.shape != margin_prev.shape:
        raise ValueError(
            f"shape mismatch: margin_t {margin_t.shape} vs "
            f"margin_prev {margin_prev.shape}"
        )
    with np.errstate(divide="ignore", invalid="ignore"):
        out = (margin_t - margin_prev) / margin_prev
    out[~np.isfinite(out)] = np.nan
    return out


def delta_quantity(quantity_t: np.ndarray, quantity_prev: np.ndarray) -> np.ndarray:
    """Eq. 11: per-product sales growth rate."""
    quantity_t = np.asarray(quantity_t, dtype=np.float64)
    quantity_prev = np.asarray(quantity_prev, dtype=np.float64)
    if quantity_t.shape != quantity_prev.shape:
        raise ValueError(
            f"shape mismatch: quantity_t {quantity_t.shape} vs "
            f"quantity_prev {quantity_prev.shape}"
        )
    with np.errstate(divide="ignore", invalid="ignore"):
        out = (quantity_t - quantity_prev) / quantity_prev
    out[~np.isfinite(out)] = np.nan
    return out


def mean_growth(values: Iterable[float]) -> float:
    """Mean across products, ignoring non-finite entries."""
    arr = np.asarray(list(values), dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    return float(finite.mean()) if finite.size else float("nan")


def count_nonfinite(values: Iterable[float]) -> int:
    arr = np.asarray(list(values), dtype=np.float64)
    return int((~np.isfinite(arr)).sum())


def paper_reward(
    delta_margin_per_product: np.ndarray,
    delta_quantity_per_product: np.ndarray,
    alpha: float,
    beta: float,
) -> float:
    """Eq. 14: r_t = alpha * mean(dM) + beta * mean(dQ)."""
    if abs((alpha + beta) - 1.0) > 1e-8:
        raise ValueError(f"alpha + beta must equal 1, got {alpha}+{beta}")
    return (
        alpha * mean_growth(delta_margin_per_product)
        + beta * mean_growth(delta_quantity_per_product)
    )


def top1_top3(
    metric_by_model_by_product: Mapping[str, np.ndarray],
    higher_is_better: bool = True,
) -> Dict[str, Dict[str, float]]:
    """Compute Top-1 and Top-3 frequencies for each model."""
    models = list(metric_by_model_by_product.keys())
    if not models:
        return {}

    arrays = [
        np.asarray(metric_by_model_by_product[m], dtype=np.float64)
        for m in models
    ]
    n = arrays[0].shape[0]
    for m, arr in zip(models, arrays):
        if arr.shape != (n,):
            raise ValueError(
                f"model '{m}' has shape {arr.shape}, expected ({n},)"
            )

    mat = np.stack(arrays, axis=0)
    if not higher_is_better:
        mat = -mat

    finite_mask = np.isfinite(mat)
    mat_for_rank = np.where(finite_mask, mat, -np.inf)

    result: Dict[str, Dict[str, float]] = {}
    for i, model in enumerate(models):
        my = mat_for_rank[i:i + 1, :]
        better = (mat_for_rank > my).sum(axis=0)
        my_finite = finite_mask[i, :]
        ranks = np.where(my_finite, better + 1, np.inf)
        top1 = int((ranks == 1).sum())
        top3 = int((ranks <= 3).sum())
        valid = int(my_finite.sum())
        result[model] = {
            "top1": top1,
            "top3": top3,
            "top1_frac": float(top1 / valid) if valid else float("nan"),
            "top3_frac": float(top3 / valid) if valid else float("nan"),
            "n_products": valid,
        }
    return result


def reward_retention(reward_noise: float, reward_original: float) -> float:
    """Table 12: R_noise / R_original as a percentage."""
    if reward_original == 0 or not np.isfinite(reward_original):
        return float("nan")
    return float(100.0 * reward_noise / reward_original)
