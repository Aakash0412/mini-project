"""Explicit profit definition used by the v2 evaluator.

Paper Eqs. 5-8 define profit via C, S, V, Q^r(t). Dataset provides only
Gross Margin Rate (GMR). Documented reduction:

    C = 1 - GMR,  S = 0,  V = 0,  Q^r = 0

so Eq. 8 becomes M = R^r*(1-d)*GMR*Q = candidate_price * Q_t * GMR.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.environment.pricing_economics import PricingOutcome


PROFIT_DEFINITION = (
    "gross_profit_proxy: M = R^r*(1-d)*GMR*Q, "
    "with C=1-GMR, S=0, V=0, Q^r=0 "
    "(paper Eq. 8 partially reduced; not exact accounting profit)"
)


@dataclass(frozen=True)
class ProfitLabel:
    outcome: PricingOutcome
    profit_definition: str = PROFIT_DEFINITION

    @property
    def gross_profit_proxy(self) -> Optional[float]:
        return self.outcome.estimated_margin


def label_outcome(outcome: PricingOutcome) -> ProfitLabel:
    return ProfitLabel(outcome=outcome)