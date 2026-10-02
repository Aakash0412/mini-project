from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from src.environment.counterfactual_simulator import (
    CounterfactualSimulator,
    DISCOUNT_BINS,
)
from src.environment.pricing_economics import calculate_pricing_outcome
from src.environment.reward import (
    RewardConfig,
    SALES_REWARD,
    calculate_reward,
)


@dataclass(frozen=True)
class ProductContext:
    """
    Product-level information required by the pricing environment.

    These values come directly from the dataset rather than being
    fabricated or inferred from the 111-D state.
    """

    observed_price: float
    observed_discount: float
    last_monthly_sales: float
    gross_margin_rate: Optional[float] = None


class DynamicPricingEnvironment:
    """
    Economic counterfactual environment for dynamic pricing.

    State:
        111-D ItaNet state.

    Action:
        One of the 9 paper-defined discount levels:
        [0, 5, 10, ..., 40].

    The frozen sales predictor estimates sales growth for each
    candidate discount. The economics layer then converts that
    prediction into estimated sales, price, revenue and margin.

    Reward:
        The paper-style reward combines margin change and quantity
        change.

        For PPO stability in this local implementation, the absolute
        changes are normalized relative to the observed baseline:

            relative_delta_margin =
                delta_margin / |baseline_margin|

            relative_delta_quantity =
                delta_quantity / baseline_sales

        The normalized values are then combined using the configured
        alpha/beta reward weights.

    Important:
        The economics are a dataset-derived proxy. They are not an
        exact reconstruction of product costs because the released
        dataset does not contain complete cost accounting fields.
    """

    STATE_DIM = 111
    ACTION_DIM = len(DISCOUNT_BINS)

    def __init__(
        self,
        simulator: Optional[CounterfactualSimulator] = None,
        reward_config: RewardConfig = SALES_REWARD,
    ) -> None:
        self.simulator = (
            simulator
            if simulator is not None
            else CounterfactualSimulator()
        )

        self.reward_config = reward_config

        self.state: Optional[np.ndarray] = None
        self.product: Optional[ProductContext] = None

        self.observed_outcome = None
        self.last_outcome = None
        self.last_action: Optional[int] = None

    # -----------------------------------------------------------------
    # Reset
    # -----------------------------------------------------------------
    def reset(
        self,
        state: np.ndarray,
        product: ProductContext,
    ) -> np.ndarray:
        """
        Reset the environment for one product/state.
        """

        state = np.asarray(
            state,
            dtype=np.float32,
        )

        if state.shape != (self.STATE_DIM,):
            raise ValueError(
                f"state must have shape ({self.STATE_DIM},), "
                f"got {state.shape}"
            )

        if not np.all(np.isfinite(state)):
            raise ValueError(
                "state contains NaN or Inf"
            )

        if not isinstance(product, ProductContext):
            raise TypeError(
                "product must be a ProductContext"
            )

        self.state = state
        self.product = product
        self.last_action = None

        # -------------------------------------------------------------
        # Observed product baseline
        #
        # Growth is set to zero here because this represents the
        # observed baseline rather than a counterfactual prediction.
        # -------------------------------------------------------------
        self.observed_outcome = calculate_pricing_outcome(
            observed_price=product.observed_price,
            observed_discount=product.observed_discount,
            last_monthly_sales=product.last_monthly_sales,
            predicted_growth_rate=0.0,
            candidate_discount=product.observed_discount,
            gross_margin_rate=product.gross_margin_rate,
        )

        self.last_outcome = self.observed_outcome

        return self.state.copy()

    # -----------------------------------------------------------------
    # Action conversion
    # -----------------------------------------------------------------
    def action_to_discount(
        self,
        action: int,
    ) -> float:
        """
        Convert discrete action index to discount percentage.
        """

        if not isinstance(
            action,
            (int, np.integer),
        ):
            raise ValueError(
                "action must be an integer"
            )

        action = int(action)

        if not 0 <= action < self.ACTION_DIM:
            raise ValueError(
                f"action must be in [0, {self.ACTION_DIM - 1}], "
                f"got {action}"
            )

        return float(
            DISCOUNT_BINS[action]
        )

    # -----------------------------------------------------------------
    # Internal readiness check
    # -----------------------------------------------------------------
    def _require_ready(self) -> None:
        if (
            self.state is None
            or self.product is None
        ):
            raise RuntimeError(
                "Environment is not initialized. "
                "Call reset() first."
            )

    # -----------------------------------------------------------------
    # Counterfactual prediction
    # -----------------------------------------------------------------
    def predict_action(
        self,
        action: int,
    ):
        """
        Predict the economic outcome for one candidate action.
        """

        self._require_ready()

        discount = self.action_to_discount(
            action
        )

        growth = self.simulator.predict(
            self.state,
            discount,
        )

        outcome = calculate_pricing_outcome(
            observed_price=self.product.observed_price,
            observed_discount=self.product.observed_discount,
            last_monthly_sales=self.product.last_monthly_sales,
            predicted_growth_rate=float(growth),
            candidate_discount=discount,
            gross_margin_rate=self.product.gross_margin_rate,
        )

        return outcome

    # -----------------------------------------------------------------
    # All actions
    # -----------------------------------------------------------------
    def predict_all_actions(
        self,
    ) -> Dict[int, object]:
        """
        Return economic outcomes for all nine candidate discounts.
        """

        self._require_ready()

        return {
            action: self.predict_action(action)
            for action in range(
                self.ACTION_DIM
            )
        }

    # -----------------------------------------------------------------
    # Step
    # -----------------------------------------------------------------
    def step(
        self,
        action: int,
    ):
        """
        Execute one pricing action.

        The reward is calculated relative to the observed product
        baseline rather than relative to the previous arbitrary
        counterfactual action.

        Returns:
            next_state,
            reward,
            terminated,
            info
        """

        self._require_ready()

        outcome = self.predict_action(
            action
        )

        # -------------------------------------------------------------
        # Margin is required for the configured paper reward.
        # -------------------------------------------------------------
        if (
            outcome.estimated_margin is None
            or self.observed_outcome.estimated_margin is None
        ):
            raise ValueError(
                "Gross Margin Rate is required for the configured "
                "paper reward because delta_margin cannot be computed."
            )

        # -------------------------------------------------------------
        # Absolute economic changes
        # -------------------------------------------------------------
        delta_quantity = (
            outcome.estimated_sales
            - self.product.last_monthly_sales
        )

        delta_margin = (
            outcome.estimated_margin
            - self.observed_outcome.estimated_margin
        )

        # -------------------------------------------------------------
        # Relative normalization
        #
        # Products have very different sales and monetary scales.
        # Normalizing against each product's observed baseline makes
        # the reward numerically comparable across products.
        # -------------------------------------------------------------
        epsilon = 1e-8

        baseline_sales = max(
            abs(
                self.product.last_monthly_sales
            ),
            epsilon,
        )

        baseline_margin = max(
            abs(
                self.observed_outcome.estimated_margin
            ),
            epsilon,
        )

        relative_delta_quantity = (
            delta_quantity
            / baseline_sales
        )

        relative_delta_margin = (
            delta_margin
            / baseline_margin
        )

        # -------------------------------------------------------------
        # Paper-style weighted reward
        # -------------------------------------------------------------
        reward = calculate_reward(
            delta_margin=relative_delta_margin,
            delta_quantity=relative_delta_quantity,
            config=self.reward_config,
        )

        self.last_action = int(action)
        self.last_outcome = outcome

        predicted_growth = float(
            self.simulator.predict(
                self.state,
                outcome.discount_percent,
            )
        )

        # -------------------------------------------------------------
        # Diagnostic information
        # -------------------------------------------------------------
        info = {
            "action": int(action),

            "discount_percent": (
                outcome.discount_percent
            ),

            "reference_price": (
                outcome.reference_price
            ),

            "candidate_price": (
                outcome.candidate_price
            ),

            "predicted_growth_rate": (
                predicted_growth
            ),

            "estimated_sales": (
                outcome.estimated_sales
            ),

            "estimated_revenue": (
                outcome.estimated_revenue
            ),

            "estimated_margin": (
                outcome.estimated_margin
            ),

            # ---------------------------------------------------------
            # Absolute changes
            # ---------------------------------------------------------
            "delta_quantity": (
                delta_quantity
            ),

            "delta_margin": (
                delta_margin
            ),

            # ---------------------------------------------------------
            # Relative changes actually used by PPO
            # ---------------------------------------------------------
            "relative_delta_quantity": (
                relative_delta_quantity
            ),

            "relative_delta_margin": (
                relative_delta_margin
            ),

            "reward": float(
                reward
            ),

            # ---------------------------------------------------------
            # Baseline information
            # ---------------------------------------------------------
            "baseline_discount": (
                self.product.observed_discount
            ),

            "baseline_sales": (
                self.product.last_monthly_sales
            ),

            "baseline_margin": (
                self.observed_outcome.estimated_margin
            ),

            "reward_config": (
                self.reward_config.name
            ),

            # ---------------------------------------------------------
            # Important support flag
            #
            # The released dataset has observed discounts of 5–40%.
            # Therefore 0% is part of the paper's action space but is
            # outside the observed training support.
            # ---------------------------------------------------------
            "out_of_training_support": (
                float(outcome.discount_percent) == 0.0
            ),
        }

        # -------------------------------------------------------------
        # Current environment is a one-step counterfactual decision.
        # -------------------------------------------------------------
        terminated = True

        return (
            self.state.copy(),
            float(reward),
            terminated,
            info,
        )