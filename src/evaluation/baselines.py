"""Heuristic baselines from paper Sec. IV-A-1.

Implemented: OPS, COP, TBP, RWD, DOP, NEBP.
Deferred to v3 (need training): UCB, TS, DQNF, DT, TDQN.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from src.environment.counterfactual_simulator import DISCOUNT_BINS
from src.environment.pricing_environment import ProductContext


BaselineFn = Callable[..., int]


def _index_of(percent: int) -> int:
    return DISCOUNT_BINS.index(int(percent))


def ops(ctx, month, rng=None) -> int:
    p = int(round(ctx.observed_discount))
    return _index_of(p) if p in DISCOUNT_BINS else _index_of(0)


def cop(ctx, month, rng=None) -> int:
    target = int(round(ctx.observed_discount)) + 5
    nearest = min(DISCOUNT_BINS, key=lambda d: abs(d - target))
    return _index_of(nearest)


def tbp(ctx, month, rng=None) -> int:
    schedule = {1: 0, 2: 5, 3: 10, 4: 15, 5: 20,
                6: 25, 7: 30, 8: 35, 9: 40, 10: 40}
    return _index_of(schedule.get(int(month), 0))


def rwd(ctx, month, rng=None) -> int:
    if rng is None:
        rng = np.random.default_rng(0)
    return int(rng.integers(0, len(DISCOUNT_BINS)))


_DOP_MEDIAN: list = [None]


def set_dop_median(value: float) -> None:
    _DOP_MEDIAN[0] = float(value)


def dop(ctx, month, rng=None) -> int:
    median = _DOP_MEDIAN[0]
    if median is None:
        return _index_of(10)
    if ctx.last_monthly_sales >= median:
        return _index_of(5)
    return _index_of(20)


def nebp(ctx, month, rng=None) -> int:
    return _index_of(0)


HEURISTIC_BASELINES = {
    "OPS": ops, "COP": cop, "TBP": tbp,
    "RWD": rwd, "DOP": dop, "NEBP": nebp,
}

NOT_IMPLEMENTED_IN_V2 = ["UCB", "TS", "DQNF", "DT", "TDQN"]