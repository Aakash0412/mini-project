from __future__ import annotations

import numpy as np


NEORL_ACTION_SCALE = 5.0

NEORL_DISCOUNT_FACTORS = np.array(
    [
        0.95,
        0.90,
        0.85,
        0.80,
        0.75,
        0.70,
        0.65,
        0.60,
    ],
    dtype=np.float32,
)

TARGET_DISCOUNTS = np.array(
    [
        0,
        5,
        10,
        15,
        20,
        25,
        30,
        35,
        40,
    ],
    dtype=np.float32,
)


def decode_neorl_action(
    normalized_action: np.ndarray,
) -> np.ndarray:
    """
    Convert NeoRL dataset action back to its environment-scale form.

    Dataset actions are divided by action_space.high[0] == 5.
    """
    action = np.asarray(normalized_action, dtype=np.float32)

    if action.shape[-1] != 2:
        raise ValueError(
            f"Expected final action dimension 2, got {action.shape}"
        )

    return action * NEORL_ACTION_SCALE


def coupon_factor_to_discount_percent(
    coupon_factor: np.ndarray,
) -> np.ndarray:
    """
    Convert coupon factor into percentage discount.

    Example:
        0.95 -> 5%
        0.80 -> 20%
        0.60 -> 40%
    """
    factor = np.asarray(coupon_factor, dtype=np.float32)

    return (1.0 - factor) * 100.0


def decode_discount_from_dataset_action(
    normalized_action: np.ndarray,
) -> np.ndarray:
    """
    Recover the NeoRL coupon discount percentage
    from a normalized dataset action.
    """
    raw_action = decode_neorl_action(normalized_action)

    return coupon_factor_to_discount_percent(
        raw_action[..., 1]
    )