from src.environment.reward import (
    RewardConfig,
    SALES_REWARD,
    BALANCED_REWARD,
    PROFIT_REWARD,
    get_reward_config,
    calculate_reward,
    calculate_reward_from_config,
    reward_breakdown,
)


def test_reward_configs_are_valid():
    assert SALES_REWARD.alpha == 0.2
    assert SALES_REWARD.beta == 0.8

    assert BALANCED_REWARD.alpha == 0.5
    assert BALANCED_REWARD.beta == 0.5

    assert PROFIT_REWARD.alpha == 0.8
    assert PROFIT_REWARD.beta == 0.2


def test_sales_reward():
    reward = calculate_reward(
        delta_margin=10.0,
        delta_quantity=20.0,
        config=SALES_REWARD,
    )

    assert reward == 18.0


def test_balanced_reward():
    reward = calculate_reward(
        delta_margin=10.0,
        delta_quantity=20.0,
        config=BALANCED_REWARD,
    )

    assert reward == 15.0


def test_profit_reward():
    reward = calculate_reward(
        delta_margin=10.0,
        delta_quantity=20.0,
        config=PROFIT_REWARD,
    )

    assert reward == 12.0


def test_named_reward_config():
    assert get_reward_config("sales") == SALES_REWARD
    assert get_reward_config("balanced") == BALANCED_REWARD
    assert get_reward_config("profit") == PROFIT_REWARD


def test_invalid_reward_config():
    try:
        RewardConfig("invalid", 0.7, 0.7)
        assert False
    except ValueError:
        assert True


def test_reward_breakdown():
    result = reward_breakdown(
        delta_margin=10.0,
        delta_quantity=20.0,
        config=SALES_REWARD,
    )

    assert result["delta_margin"] == 10.0
    assert result["delta_quantity"] == 20.0
    assert result["alpha"] == 0.2
    assert result["beta"] == 0.8
    assert result["reward"] == 18.0


def test_calculate_reward_from_config():
    reward = calculate_reward_from_config(
        delta_margin=10.0,
        delta_quantity=20.0,
        config_name="balanced",
    )

    assert reward == 15.0