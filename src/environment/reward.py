"""
Phase 9: Reward Function

Implements the reward formulation used by the base paper:

    r_t = alpha * Delta_M_t + beta * Delta_Q_t

where:
    Delta_M_t = change in margin-related objective
    Delta_Q_t = change in sales/quantity-related objective

Important:
The current Amazon dataset does not provide reliable product-cost
variables required to derive true profit/margin. Therefore this module
does NOT fabricate cost values.

The reward function accepts the already-computed margin and quantity
changes as inputs. The pricing environment is responsible for providing
those values when sufficient information is available.
"""

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class RewardConfig:
    """Configuration for the paper-style reward."""

    name: str
    alpha: float
    beta: float

    def __post_init__(self) -> None:
        if self.alpha < 0 or self.beta < 0:
            raise ValueError("alpha and beta must be non-negative.")

        if abs((self.alpha + self.beta) - 1.0) > 1e-8:
            raise ValueError("alpha + beta must equal 1.")


# Paper-aligned reward configurations.
SALES_REWARD = RewardConfig(
    name="sales",
    alpha=0.2,
    beta=0.8,
)

BALANCED_REWARD = RewardConfig(
    name="balanced",
    alpha=0.5,
    beta=0.5,
)

PROFIT_REWARD = RewardConfig(
    name="profit",
    alpha=0.8,
    beta=0.2,
)


REWARD_CONFIGS: Dict[str, RewardConfig] = {
    "sales": SALES_REWARD,
    "balanced": BALANCED_REWARD,
    "profit": PROFIT_REWARD,
}


def get_reward_config(name: str) -> RewardConfig:
    """
    Return a predefined reward configuration.

    Parameters
    ----------
    name:
        One of: "sales", "balanced", "profit".
    """
    key = str(name).strip().lower()

    if key not in REWARD_CONFIGS:
        raise ValueError(
            f"Unknown reward configuration '{name}'. "
            f"Available configurations: {list(REWARD_CONFIGS.keys())}"
        )

    return REWARD_CONFIGS[key]


def calculate_reward(
    delta_margin: float,
    delta_quantity: float,
    config: RewardConfig = SALES_REWARD,
) -> float:
    """
    Calculate the paper-style weighted reward.

    Formula:

        reward = alpha * Delta_M + beta * Delta_Q

    Parameters
    ----------
    delta_margin:
        Change in the margin-related objective.

    delta_quantity:
        Change in the sales/quantity-related objective.

    config:
        Reward configuration containing alpha and beta.

    Returns
    -------
    float
        Scalar reward.
    """

    reward = (
        config.alpha * float(delta_margin)
        + config.beta * float(delta_quantity)
    )

    return float(reward)


def calculate_reward_from_config(
    delta_margin: float,
    delta_quantity: float,
    config_name: str = "sales",
) -> float:
    """
    Convenience wrapper using a named reward configuration.
    """

    config = get_reward_config(config_name)

    return calculate_reward(
        delta_margin=delta_margin,
        delta_quantity=delta_quantity,
        config=config,
    )


def reward_breakdown(
    delta_margin: float,
    delta_quantity: float,
    config: RewardConfig = SALES_REWARD,
) -> Dict[str, float]:
    """
    Return the individual reward contributions.

    Useful for debugging and experiment reporting.
    """

    margin_component = config.alpha * float(delta_margin)
    quantity_component = config.beta * float(delta_quantity)

    return {
        "alpha": config.alpha,
        "beta": config.beta,
        "delta_margin": float(delta_margin),
        "delta_quantity": float(delta_quantity),
        "margin_component": float(margin_component),
        "quantity_component": float(quantity_component),
        "reward": float(margin_component + quantity_component),
    }