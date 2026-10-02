import numpy as np
import pytest

from src.environment.pricing_economics import (
    candidate_price,
    calculate_pricing_outcome,
    estimated_sales,
    reconstruct_reference_price,
)


def test_reference_price_reconstruction():
    # Observed price = $90 at 10% discount.
    # Reference price should be $100.
    reference = reconstruct_reference_price(
        observed_price=90.0,
        observed_discount=10.0,
    )

    assert np.isclose(reference, 100.0)


def test_candidate_price():
    price = candidate_price(
        reference_price=100.0,
        discount_percent=20.0,
    )

    assert np.isclose(price, 80.0)


def test_estimated_sales():
    # 1,000 previous sales with 20% predicted growth
    # should produce 1,200 estimated sales.
    sales = estimated_sales(
        last_monthly_sales=1000.0,
        predicted_growth_rate=20.0,
    )

    assert np.isclose(sales, 1200.0)


def test_negative_growth():
    # 1,000 sales with -50% growth -> 500.
    sales = estimated_sales(
        last_monthly_sales=1000.0,
        predicted_growth_rate=-50.0,
    )

    assert np.isclose(sales, 500.0)


def test_growth_below_minus_100_is_bounded():
    sales = estimated_sales(
        last_monthly_sales=1000.0,
        predicted_growth_rate=-150.0,
    )

    assert sales == 0.0


def test_complete_pricing_outcome():
    outcome = calculate_pricing_outcome(
        observed_price=90.0,
        observed_discount=10.0,
        last_monthly_sales=1000.0,
        predicted_growth_rate=20.0,
        candidate_discount=20.0,
        gross_margin_rate=50.0,
    )

    # Reference price = 100
    assert np.isclose(
        outcome.reference_price,
        100.0,
    )

    # Candidate price = 80
    assert np.isclose(
        outcome.candidate_price,
        80.0,
    )

    # Estimated sales = 1200
    assert np.isclose(
        outcome.estimated_sales,
        1200.0,
    )

    # Revenue = 80 * 1200
    assert np.isclose(
        outcome.estimated_revenue,
        96000.0,
    )

    # Margin = 50% of revenue
    assert np.isclose(
        outcome.estimated_margin,
        48000.0,
    )


def test_missing_margin_is_allowed():
    outcome = calculate_pricing_outcome(
        observed_price=90.0,
        observed_discount=10.0,
        last_monthly_sales=1000.0,
        predicted_growth_rate=20.0,
        candidate_discount=20.0,
        gross_margin_rate=None,
    )

    assert outcome.estimated_margin is None


def test_invalid_price():
    with pytest.raises(ValueError):
        reconstruct_reference_price(
            observed_price=0.0,
            observed_discount=10.0,
        )


def test_invalid_discount():
    with pytest.raises(ValueError):
        candidate_price(
            reference_price=100.0,
            discount_percent=120.0,
        )


def test_invalid_sales():
    with pytest.raises(ValueError):
        estimated_sales(
            last_monthly_sales=-10.0,
            predicted_growth_rate=20.0,
        )


def test_non_finite_growth():
    with pytest.raises(ValueError):
        estimated_sales(
            last_monthly_sales=1000.0,
            predicted_growth_rate=np.nan,
        )