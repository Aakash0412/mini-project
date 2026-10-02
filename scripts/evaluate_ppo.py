from __future__ import annotations

import sys
from pathlib import Path

# ---------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch

from src.environment.ppo_dataset import load_ppo_dataset
from src.environment.pricing_environment import DynamicPricingEnvironment
from src.models.ppo_agent import PPOAgent


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

# New 8-action observed-support PPO checkpoint.
CHECKPOINT = Path(
    "models/ppo/ppo_batched_supported_8action_500_updates.pth"
)

OUTPUT_PATH = Path(
    "models/ppo/supported_8action_policy_evaluation.csv"
)

# IMPORTANT:
# The PPO actor has 8 outputs.
#
# PPO action 0 -> 5%
# PPO action 1 -> 10%
# PPO action 2 -> 15%
# PPO action 3 -> 20%
# PPO action 4 -> 25%
# PPO action 5 -> 30%
# PPO action 6 -> 35%
# PPO action 7 -> 40%
#
# The DynamicPricingEnvironment internally uses:
# 0 -> 0%
# 1 -> 5%
# ...
# 8 -> 40%
#
# Therefore we translate PPO actions using:
# environment_action = ppo_action + 1
#
DISCOUNT_BINS = [
    5,
    10,
    15,
    20,
    25,
    30,
    35,
    40,
]

STATE_DIM = 111
ACTION_DIM = 8

EPSILON = 1e-8


# ---------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------

def load_agent(checkpoint_path: Path) -> PPOAgent:
    """Load the trained 8-action PPO actor and critic."""

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    # Validate checkpoint metadata if available.
    checkpoint_action_dim = checkpoint.get(
        "action_dim",
        None,
    )

    if checkpoint_action_dim is not None:
        if int(checkpoint_action_dim) != ACTION_DIM:
            raise ValueError(
                "Checkpoint action dimension mismatch. "
                f"Expected {ACTION_DIM}, "
                f"found {checkpoint_action_dim}."
            )

    agent = PPOAgent(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        actor_lr=8e-4,
        critic_lr=8e-4,
    )

    agent.actor.load_state_dict(
        checkpoint["actor_state_dict"]
    )

    agent.critic.load_state_dict(
        checkpoint["critic_state_dict"]
    )

    agent.actor.eval()
    agent.critic.eval()

    return agent


# ---------------------------------------------------------------------
# Deterministic action selection
# ---------------------------------------------------------------------

def select_deterministic_action(
    agent: PPOAgent,
    state: np.ndarray,
) -> int:
    """
    Select the highest-probability PPO action.

    Stochastic sampling is intentionally disabled during evaluation.
    """

    state_tensor = torch.as_tensor(
        state,
        dtype=torch.float32,
    ).unsqueeze(0)

    with torch.no_grad():

        output = agent.actor(
            state_tensor
        )

        # Actor may return logits directly or a distribution.
        if hasattr(output, "logits"):
            logits = output.logits
        else:
            logits = output

        action = torch.argmax(
            logits,
            dim=-1,
        ).item()

    return int(action)


# ---------------------------------------------------------------------
# Relative reward calculation
# ---------------------------------------------------------------------

def calculate_relative_reward(
    delta_margin: float,
    delta_quantity: float,
    baseline_margin: float,
    baseline_quantity: float,
    alpha: float,
    beta: float,
) -> float:
    """
    Calculate the normalized relative economic reward.

    This matches the current training environment:

        relative_delta_quantity =
            delta_quantity /
            max(abs(baseline_quantity), EPSILON)

        relative_delta_margin =
            delta_margin /
            max(abs(baseline_margin), EPSILON)

        reward =
            alpha * relative_delta_margin
            + beta * relative_delta_quantity
    """

    relative_delta_quantity = (
        delta_quantity
        / max(
            abs(baseline_quantity),
            EPSILON,
        )
    )

    relative_delta_margin = (
        delta_margin
        / max(
            abs(baseline_margin),
            EPSILON,
        )
    )

    reward = (
        alpha * relative_delta_margin
        + beta * relative_delta_quantity
    )

    return float(reward)


# ---------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------

def main():

    print("=" * 70)
    print("8-ACTION PPO ECONOMIC POLICY VALIDATION")
    print("=" * 70)

    print()
    print(f"Checkpoint: {CHECKPOINT}")

    print()
    print("PPO action space:")
    print(
        "  "
        + ", ".join(
            f"{discount}%"
            for discount in DISCOUNT_BINS
        )
    )

    print(
        f"Action dimension: {ACTION_DIM}"
    )

    # -----------------------------------------------------------------
    # Load dataset
    # -----------------------------------------------------------------

    dataset = load_ppo_dataset()

    print(
        f"Loaded economic PPO samples: "
        f"{len(dataset.states)}"
    )

    # -----------------------------------------------------------------
    # Load dataframe for chronological split information
    # -----------------------------------------------------------------

    df = pd.read_parquet(
        "data/processed/products_clean.parquet"
    )

    source_indices = dataset.source_row_indices

    aligned = df.iloc[
        source_indices
    ].copy()

    # -----------------------------------------------------------------
    # Chronological split
    # -----------------------------------------------------------------

    train_mask = aligned[
        "MonthNum"
    ].between(1, 6)

    val_mask = aligned[
        "MonthNum"
    ].between(7, 8)

    test_mask = aligned[
        "MonthNum"
    ].between(9, 10)

    split_masks = {
        "train": train_mask.to_numpy(),
        "validation": val_mask.to_numpy(),
        "test": test_mask.to_numpy(),
    }

    # -----------------------------------------------------------------
    # Load PPO agent
    # -----------------------------------------------------------------

    agent = load_agent(
        CHECKPOINT
    )

    # -----------------------------------------------------------------
    # Create pricing environment
    # -----------------------------------------------------------------

    environment = DynamicPricingEnvironment()

    # Make sure evaluation and training use the same reward
    # configuration.

    alpha = environment.reward_config.alpha
    beta = environment.reward_config.beta

    print()
    print("Reward configuration:")
    print(
        f"  Alpha (margin):   {alpha}"
    )
    print(
        f"  Beta (quantity):  {beta}"
    )
    print(
        "  Reward type:      normalized relative reward"
    )

    results = []

    # -----------------------------------------------------------------
    # Evaluate each split
    # -----------------------------------------------------------------

    for split_name, mask in split_masks.items():

        positions = np.flatnonzero(
            mask
        )

        print()
        print("-" * 70)
        print(
            f"{split_name.upper()} SET: "
            f"{len(positions)} products"
        )
        print("-" * 70)

        for position in positions:

            state = dataset.states[
                position
            ]

            product = dataset.products[
                position
            ]

            # ---------------------------------------------------------
            # Reset environment for this product
            # ---------------------------------------------------------

            environment.reset(
                state=state,
                product=product,
            )

            # ---------------------------------------------------------
            # Baseline economics
            # ---------------------------------------------------------

            baseline_sales = (
                product.last_monthly_sales
            )

            baseline_margin = (
                environment
                .observed_outcome
                .estimated_margin
            )

            # Products without valid margin cannot be evaluated
            # with the economic reward.
            if baseline_margin is None:
                continue

            # ---------------------------------------------------------
            # PPO selected action
            # ---------------------------------------------------------

            ppo_action = (
                select_deterministic_action(
                    agent,
                    state,
                )
            )

            if not (
                0 <= ppo_action < ACTION_DIM
            ):
                raise RuntimeError(
                    f"Invalid PPO action {ppo_action}. "
                    f"Expected 0..{ACTION_DIM - 1}."
                )

            # ---------------------------------------------------------
            # Translate 8-action PPO index to the environment's
            # original 9-action index.
            #
            # PPO:
            #   0 -> 5%
            #   1 -> 10%
            #   ...
            #   7 -> 40%
            #
            # Environment:
            #   1 -> 5%
            #   2 -> 10%
            #   ...
            #   8 -> 40%
            # ---------------------------------------------------------

            environment_ppo_action = (
                ppo_action + 1
            )

            ppo_outcome = (
                environment.predict_action(
                    environment_ppo_action
                )
            )

            # ---------------------------------------------------------
            # PPO economic changes
            # ---------------------------------------------------------

            ppo_delta_quantity = (
                ppo_outcome.estimated_sales
                - baseline_sales
            )

            ppo_delta_margin = (
                ppo_outcome.estimated_margin
                - baseline_margin
            )

            ppo_reward = (
                calculate_relative_reward(
                    delta_margin=ppo_delta_margin,
                    delta_quantity=ppo_delta_quantity,
                    baseline_margin=baseline_margin,
                    baseline_quantity=baseline_sales,
                    alpha=alpha,
                    beta=beta,
                )
            )

            # ---------------------------------------------------------
            # Exhaustive counterfactual search
            #
            # IMPORTANT:
            # The oracle for this experiment must use ONLY:
            #
            #   5,10,15,20,25,30,35,40%
            #
            # It must NOT include 0%.
            # ---------------------------------------------------------

            all_outcomes = (
                environment.predict_all_actions()
            )

            oracle_values = {}

            for environment_action, outcome in (
                all_outcomes.items()
            ):

                # Environment action 0 corresponds to 0%,
                # which is excluded from this experiment.

                if environment_action == 0:
                    continue

                if (
                    outcome.estimated_margin
                    is None
                ):
                    continue

                delta_quantity = (
                    outcome.estimated_sales
                    - baseline_sales
                )

                delta_margin = (
                    outcome.estimated_margin
                    - baseline_margin
                )

                oracle_reward = (
                    calculate_relative_reward(
                        delta_margin=delta_margin,
                        delta_quantity=delta_quantity,
                        baseline_margin=baseline_margin,
                        baseline_quantity=baseline_sales,
                        alpha=alpha,
                        beta=beta,
                    )
                )

                oracle_values[
                    environment_action
                ] = oracle_reward

            if not oracle_values:
                continue

            # ---------------------------------------------------------
            # Oracle action
            # ---------------------------------------------------------

            oracle_environment_action = max(
                oracle_values,
                key=oracle_values.get,
            )

            oracle_outcome = (
                all_outcomes[
                    oracle_environment_action
                ]
            )

            oracle_reward = (
                oracle_values[
                    oracle_environment_action
                ]
            )

            # Convert environment action back to
            # the 0..7 PPO action indexing.

            oracle_ppo_action = (
                oracle_environment_action - 1
            )

            # ---------------------------------------------------------
            # Save result
            # ---------------------------------------------------------

            results.append(
                {
                    "split": split_name,

                    "source_row": int(
                        source_indices[
                            position
                        ]
                    ),

                    "observed_discount": float(
                        product.observed_discount
                    ),

                    # -------------------------------------------------
                    # PPO
                    # -------------------------------------------------

                    "ppo_action": int(
                        ppo_action
                    ),

                    "ppo_discount": float(
                        DISCOUNT_BINS[
                            ppo_action
                        ]
                    ),

                    "ppo_environment_action": int(
                        environment_ppo_action
                    ),

                    "ppo_reward": float(
                        ppo_reward
                    ),

                    "ppo_sales": float(
                        ppo_outcome
                        .estimated_sales
                    ),

                    "ppo_revenue": float(
                        ppo_outcome
                        .estimated_revenue
                    ),

                    "ppo_margin": float(
                        ppo_outcome
                        .estimated_margin
                    ),

                    # -------------------------------------------------
                    # Oracle
                    # -------------------------------------------------

                    "oracle_action": int(
                        oracle_ppo_action
                    ),

                    "oracle_discount": float(
                        DISCOUNT_BINS[
                            oracle_ppo_action
                        ]
                    ),

                    "oracle_environment_action": int(
                        oracle_environment_action
                    ),

                    "oracle_reward": float(
                        oracle_reward
                    ),

                    "oracle_sales": float(
                        oracle_outcome
                        .estimated_sales
                    ),

                    "oracle_revenue": float(
                        oracle_outcome
                        .estimated_revenue
                    ),

                    "oracle_margin": float(
                        oracle_outcome
                        .estimated_margin
                    ),

                    # -------------------------------------------------
                    # Baseline
                    # -------------------------------------------------

                    "baseline_sales": float(
                        baseline_sales
                    ),

                    "baseline_margin": float(
                        baseline_margin
                    ),

                    # -------------------------------------------------
                    # Additional PPO diagnostics
                    # -------------------------------------------------

                    "ppo_delta_quantity": float(
                        ppo_delta_quantity
                    ),

                    "ppo_delta_margin": float(
                        ppo_delta_margin
                    ),

                    "ppo_relative_delta_quantity": float(
                        ppo_delta_quantity
                        / max(
                            abs(
                                baseline_sales
                            ),
                            EPSILON,
                        )
                    ),

                    "ppo_relative_delta_margin": float(
                        ppo_delta_margin
                        / max(
                            abs(
                                baseline_margin
                            ),
                            EPSILON,
                        )
                    ),
                }
            )

    # -----------------------------------------------------------------
    # Results dataframe
    # -----------------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    if results_df.empty:
        raise RuntimeError(
            "No valid evaluation results were produced."
        )

    # -----------------------------------------------------------------
    # Save evaluation
    # -----------------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    # -----------------------------------------------------------------
    # Evaluation summary
    # -----------------------------------------------------------------

    print()
    print("=" * 70)
    print("8-ACTION PPO EVALUATION SUMMARY")
    print("=" * 70)

    for split_name in [
        "train",
        "validation",
        "test",
    ]:

        subset = results_df[
            results_df["split"]
            == split_name
        ]

        if subset.empty:
            continue

        ppo_reward_mean = (
            subset[
                "ppo_reward"
            ].mean()
        )

        oracle_reward_mean = (
            subset[
                "oracle_reward"
            ].mean()
        )

        regret = (
            subset["oracle_reward"]
            - subset["ppo_reward"]
        ).mean()

        oracle_match = (
            subset["ppo_action"]
            == subset["oracle_action"]
        ).mean()

        # PPO reward as percentage of oracle reward.
        if abs(oracle_reward_mean) > EPSILON:
            oracle_efficiency = (
                ppo_reward_mean
                / oracle_reward_mean
                * 100.0
            )
        else:
            oracle_efficiency = np.nan

        print()
        print(
            split_name.upper()
        )

        print(
            f"Samples:             "
            f"{len(subset)}"
        )

        print(
            f"PPO mean reward:     "
            f"{ppo_reward_mean:.6f}"
        )

        print(
            f"Oracle mean reward:  "
            f"{oracle_reward_mean:.6f}"
        )

        print(
            f"Mean oracle gap:     "
            f"{regret:.6f}"
        )

        print(
            f"PPO/oracle reward:   "
            f"{oracle_efficiency:.2f}%"
        )

        print(
            f"Oracle action match: "
            f"{oracle_match * 100:.2f}%"
        )

        # -------------------------------------------------------------
        # PPO distribution
        # -------------------------------------------------------------

        print()
        print(
            "PPO discount distribution:"
        )

        distribution = (
            subset[
                "ppo_discount"
            ]
            .value_counts()
            .sort_index()
        )

        for discount in DISCOUNT_BINS:

            count = int(
                distribution.get(
                    discount,
                    0,
                )
            )

            percentage = (
                count
                / len(subset)
                * 100
            )

            print(
                f"  {discount:>2.0f}% : "
                f"{count:>5} "
                f"({percentage:>6.2f}%)"
            )

        # -------------------------------------------------------------
        # Oracle distribution
        # -------------------------------------------------------------

        print()
        print(
            "8-action oracle discount distribution:"
        )

        oracle_distribution = (
            subset[
                "oracle_discount"
            ]
            .value_counts()
            .sort_index()
        )

        for discount in DISCOUNT_BINS:

            count = int(
                oracle_distribution.get(
                    discount,
                    0,
                )
            )

            percentage = (
                count
                / len(subset)
                * 100
            )

            print(
                f"  {discount:>2.0f}% : "
                f"{count:>5} "
                f"({percentage:>6.2f}%)"
            )

    # -----------------------------------------------------------------
    # Finite-value validation
    # -----------------------------------------------------------------

    numeric_columns = [
        "ppo_reward",
        "oracle_reward",
        "ppo_sales",
        "oracle_sales",
        "ppo_revenue",
        "oracle_revenue",
        "ppo_margin",
        "oracle_margin",
        "baseline_sales",
        "baseline_margin",
        "ppo_delta_quantity",
        "ppo_delta_margin",
        "ppo_relative_delta_quantity",
        "ppo_relative_delta_margin",
    ]

    finite = np.isfinite(
        results_df[
            numeric_columns
        ].to_numpy()
    ).all()

    print()
    print(
        "Finite numeric results: "
        f"{'PASS' if finite else 'FAIL'}"
    )

    # -----------------------------------------------------------------
    # Action validity validation
    # -----------------------------------------------------------------

    valid_ppo_actions = (
        results_df["ppo_action"]
        .between(
            0,
            ACTION_DIM - 1,
        )
        .all()
    )

    valid_oracle_actions = (
        results_df["oracle_action"]
        .between(
            0,
            ACTION_DIM - 1,
        )
        .all()
    )

    valid_discounts = (
        results_df["ppo_discount"]
        .isin(
            DISCOUNT_BINS
        )
        .all()
        and
        results_df["oracle_discount"]
        .isin(
            DISCOUNT_BINS
        )
        .all()
    )

    print(
        "PPO action validity:    "
        f"{'PASS' if valid_ppo_actions else 'FAIL'}"
    )

    print(
        "Oracle action validity: "
        f"{'PASS' if valid_oracle_actions else 'FAIL'}"
    )

    print(
        "Discount validity:      "
        f"{'PASS' if valid_discounts else 'FAIL'}"
    )

    print()
    print(
        f"Evaluated products: "
        f"{len(results_df)}"
    )

    print(
        f"Saved evaluation: "
        f"{OUTPUT_PATH}"
    )


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------

if __name__ == "__main__":
    main()