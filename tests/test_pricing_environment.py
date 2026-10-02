import numpy as np
import pytest

from src.environment.pricing_environment import (
    DynamicPricingEnvironment,
    ProductContext,
)


def make_state():
    return np.ones(111, dtype=np.float32)


def make_product():
    return ProductContext(
        observed_price=20.0,
        observed_discount=20.0,
        last_monthly_sales=1000.0,
        gross_margin_rate=60.0,
    )


def test_action_to_discount():
    env = DynamicPricingEnvironment()

    assert env.action_to_discount(0) == 0.0
    assert env.action_to_discount(1) == 5.0
    assert env.action_to_discount(8) == 40.0


def test_invalid_action():
    env = DynamicPricingEnvironment()

    with pytest.raises(ValueError):
        env.action_to_discount(-1)

    with pytest.raises(ValueError):
        env.action_to_discount(9)


def test_reset():
    env = DynamicPricingEnvironment()

    state = env.reset(
        make_state(),
        make_product(),
    )

    assert state.shape == (111,)
    assert np.isfinite(state).all()
    assert env.last_action is None


def test_reset_rejects_bad_state():
    env = DynamicPricingEnvironment()

    with pytest.raises(ValueError):
        env.reset(
            np.zeros(110, dtype=np.float32),
            make_product(),
        )


def test_reset_rejects_nan_state():
    env = DynamicPricingEnvironment()

    state = make_state()
    state[0] = np.nan

    with pytest.raises(ValueError):
        env.reset(state, make_product())


def test_predict_action():
    env = DynamicPricingEnvironment()

    env.reset(
        make_state(),
        make_product(),
    )

    outcome = env.predict_action(4)

    assert outcome.discount_percent == 20.0
    assert outcome.candidate_price > 0
    assert outcome.estimated_sales >= 0
    assert outcome.estimated_revenue >= 0
    assert outcome.estimated_margin is not None


def test_predict_all_actions():
    env = DynamicPricingEnvironment()

    env.reset(
        make_state(),
        make_product(),
    )

    outcomes = env.predict_all_actions()

    assert len(outcomes) == 9

    discounts = [
        outcomes[action].discount_percent
        for action in range(9)
    ]

    assert discounts == [
        0.0,
        5.0,
        10.0,
        15.0,
        20.0,
        25.0,
        30.0,
        35.0,
        40.0,
    ]


def test_step_returns_economic_reward():
    env = DynamicPricingEnvironment()

    state = env.reset(
        make_state(),
        make_product(),
    )

    next_state, reward, terminated, info = env.step(4)

    assert next_state.shape == (111,)
    assert np.isfinite(reward)
    assert terminated is True

    assert info["discount_percent"] == 20.0
    assert np.isfinite(info["candidate_price"])
    assert np.isfinite(info["estimated_sales"])
    assert np.isfinite(info["estimated_revenue"])
    assert np.isfinite(info["estimated_margin"])
    assert np.isfinite(info["delta_quantity"])
    assert np.isfinite(info["delta_margin"])


def test_reward_is_relative_to_observed_baseline():
    env = DynamicPricingEnvironment()

    env.reset(
        make_state(),
        make_product(),
    )

    outcome = env.predict_action(4)

    _, reward, _, info = env.step(4)

    expected_delta_quantity = (
        outcome.estimated_sales
        - 1000.0
    )

    expected_delta_margin = (
        outcome.estimated_margin
        - env.observed_outcome.estimated_margin
    )

    assert np.isclose(
        info["delta_quantity"],
        expected_delta_quantity,
    )

    assert np.isclose(
        info["delta_margin"],
        expected_delta_margin,
    )


def test_missing_margin_is_rejected_for_paper_reward():
    env = DynamicPricingEnvironment()

    env.reset(
        make_state(),
        ProductContext(
            observed_price=20.0,
            observed_discount=20.0,
            last_monthly_sales=1000.0,
            gross_margin_rate=None,
        ),
    )

    with pytest.raises(ValueError):
        env.step(4)


def test_first_action_does_not_compare_growth_to_zero():
    env = DynamicPricingEnvironment()

    env.reset(
        make_state(),
        make_product(),
    )

    _, _, _, info = env.step(4)

    # Quantity change is against observed sales = 1000,
    # not against an artificial zero prediction.
    assert np.isclose(
        info["delta_quantity"],
        info["estimated_sales"] - 1000.0,
    )