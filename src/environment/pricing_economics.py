from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PricingOutcome:
    discount_percent: float
    reference_price: float
    candidate_price: float
    estimated_sales: float
    estimated_revenue: float
    estimated_margin: float | None


def _validate_finite(value: float, name: str):
    if not np.isfinite(value):
        raise ValueError(
            f"{name} must be finite."
        )


def validate_inputs(
    observed_price: float,
    observed_discount: float,
    last_monthly_sales: float,
    gross_margin_rate: float | None,
    predicted_growth_rate: float,
):
    _validate_finite(
        observed_price,
        "Observed price",
    )

    _validate_finite(
        observed_discount,
        "Observed discount",
    )

    _validate_finite(
        last_monthly_sales,
        "Last monthly sales",
    )

    _validate_finite(
        predicted_growth_rate,
        "Predicted growth rate",
    )

    if observed_price <= 0:
        raise ValueError(
            "Observed price must be positive."
        )

    if not 0 <= observed_discount <= 100:
        raise ValueError(
            "Observed discount must be between 0 and 100."
        )

    if last_monthly_sales < 0:
        raise ValueError(
            "Last monthly sales cannot be negative."
        )

    if gross_margin_rate is not None:
        _validate_finite(
            gross_margin_rate,
            "Gross margin rate",
        )


def reconstruct_reference_price(
    observed_price: float,
    observed_discount: float,
) -> float:
    """
    Reconstruct an approximate pre-discount reference price.

    This is a dataset-derived pricing proxy,
    not a product-cost estimate.
    """

    _validate_finite(
        observed_price,
        "Observed price",
    )

    _validate_finite(
        observed_discount,
        "Observed discount",
    )

    if observed_price <= 0:
        raise ValueError(
            "Observed price must be positive."
        )

    if not 0 <= observed_discount < 100:
        raise ValueError(
            "Observed discount must be between 0 and 99.999."
        )

    denominator = (
        1.0 - observed_discount / 100.0
    )

    reference_price = (
        observed_price / denominator
    )

    if not np.isfinite(reference_price):
        raise ValueError(
            "Reference price is not finite."
        )

    return float(reference_price)


def candidate_price(
    reference_price: float,
    discount_percent: float,
) -> float:

    _validate_finite(
        reference_price,
        "Reference price",
    )

    _validate_finite(
        discount_percent,
        "Candidate discount",
    )

    if reference_price <= 0:
        raise ValueError(
            "Reference price must be positive."
        )

    if not 0 <= discount_percent <= 100:
        raise ValueError(
            "Candidate discount must be between 0 and 100."
        )

    price = (
        reference_price
        * (1.0 - discount_percent / 100.0)
    )

    if price < 0 or not np.isfinite(price):
        raise ValueError(
            "Candidate price is invalid."
        )

    return float(price)


def estimated_sales(
    last_monthly_sales: float,
    predicted_growth_rate: float,
) -> float:
    """
    Convert predicted sales-growth rate into estimated
    monthly sales.

    Example:
        100 sales + 20% growth -> 120 estimated sales.

    Growth below -100% is bounded at zero because
    negative sales are not physically meaningful.
    """

    _validate_finite(
        last_monthly_sales,
        "Last monthly sales",
    )

    _validate_finite(
        predicted_growth_rate,
        "Predicted growth rate",
    )

    if last_monthly_sales < 0:
        raise ValueError(
            "Last monthly sales cannot be negative."
        )

    growth_multiplier = (
        1.0
        + predicted_growth_rate / 100.0
    )

    sales = max(
        0.0,
        last_monthly_sales * growth_multiplier,
    )

    if not np.isfinite(sales):
        raise ValueError(
            "Estimated sales are not finite."
        )

    return float(sales)


def calculate_pricing_outcome(
    observed_price: float,
    observed_discount: float,
    last_monthly_sales: float,
    predicted_growth_rate: float,
    candidate_discount: float,
    gross_margin_rate: float | None = None,
) -> PricingOutcome:

    validate_inputs(
        observed_price=observed_price,
        observed_discount=observed_discount,
        last_monthly_sales=last_monthly_sales,
        gross_margin_rate=gross_margin_rate,
        predicted_growth_rate=predicted_growth_rate,
    )

    reference_price = reconstruct_reference_price(
        observed_price=observed_price,
        observed_discount=observed_discount,
    )

    price = candidate_price(
        reference_price=reference_price,
        discount_percent=candidate_discount,
    )

    sales = estimated_sales(
        last_monthly_sales=last_monthly_sales,
        predicted_growth_rate=predicted_growth_rate,
    )

    revenue = price * sales

    if gross_margin_rate is None:
        margin = None
    else:
        margin = (
            revenue
            * gross_margin_rate
            / 100.0
        )

    return PricingOutcome(
        discount_percent=float(candidate_discount),
        reference_price=reference_price,
        candidate_price=price,
        estimated_sales=sales,
        estimated_revenue=float(revenue),
        estimated_margin=(
            None
            if margin is None
            else float(margin)
        ),
    )