"""
Phase 15 — Robustness Evaluation

Tests robustness of the transfer-warm-start PPO policy to
sales-prediction errors.

Base-paper methodology:
    g_tilde = g_hat + epsilon
    epsilon ~ N(0, sigma^2)

where sigma is expressed as a fraction of the empirical
standard deviation of predicted sales response on the test set:

    0
    0.05 * Std(g_hat)
    0.10 * Std(g_hat)
    0.15 * Std(g_hat)

The PPO policy itself is NOT retrained or modified.

The experiment uses:
    - frozen 111-D state
    - frozen sales predictor
    - frozen transfer PPO policy
    - same pricing economics
    - same normalized reward
    - same test split
    - same 8-action observed-support space

This is an offline simulated robustness experiment.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# ---------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------

from src.environment.ppo_dataset import load_ppo_dataset
from src.environment.pricing_environment import (
    DynamicPricingEnvironment,
)
from src.environment.pricing_economics import (
    calculate_pricing_outcome,
)
from src.models.ppo_agent import PPOAgent


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

SEED = 42

STATE_DIM = 111
ACTION_DIM = 8

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

NOISE_LEVELS = [
    0.0,
    0.05,
    0.10,
    0.15,
]

CHECKPOINT = (
    PROJECT_ROOT
    / "models"
    / "ppo_transfer"
    / "ppo_transfer_supported_8action_1000_updates.pth"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "robustness"
)

RESULTS_PATH = (
    OUTPUT_DIR
    / "robustness_results.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "robustness_summary.json"
)

EPSILON = 1e-8


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------

def set_seed(seed: int = SEED) -> None:

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------
# PPO loading
# ---------------------------------------------------------------------

def load_agent(
    checkpoint_path: Path,
) -> PPOAgent:

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: "
            f"{checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

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
# Deterministic PPO action
# ---------------------------------------------------------------------

def select_deterministic_action(
    agent: PPOAgent,
    state: np.ndarray,
) -> int:

    state_tensor = torch.as_tensor(
        state,
        dtype=torch.float32,
    ).unsqueeze(0)

    with torch.no_grad():

        output = agent.actor(
            state_tensor
        )

        if hasattr(
            output,
            "logits",
        ):
            logits = output.logits
        else:
            logits = output

        action = torch.argmax(
            logits,
            dim=-1,
        ).item()

    action = int(action)

    if not (
        0 <= action < ACTION_DIM
    ):
        raise RuntimeError(
            f"Invalid PPO action: {action}"
        )

    return action


# ---------------------------------------------------------------------
# Relative reward
# ---------------------------------------------------------------------

def calculate_relative_reward(
    delta_margin: float,
    delta_quantity: float,
    baseline_margin: float,
    baseline_quantity: float,
    alpha: float,
    beta: float,
) -> float:

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

    return float(
        alpha * relative_delta_margin
        + beta * relative_delta_quantity
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    set_seed(SEED)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 72)
    print("PHASE 15 — PPO ROBUSTNESS EVALUATION")
    print("=" * 72)

    print()
    print(
        f"Checkpoint: {CHECKPOINT}"
    )

    print()
    print(
        "Noise levels:"
    )

    for level in NOISE_LEVELS:
        print(
            f"  {level:.2f} × Std(predicted growth)"
        )

    # -------------------------------------------------------------
    # Load dataset
    # -------------------------------------------------------------

    dataset = load_ppo_dataset()

    print()
    print(
        f"Loaded economic PPO samples: "
        f"{len(dataset.states)}"
    )

    # -------------------------------------------------------------
    # Align dataframe and create test split
    # -------------------------------------------------------------

    df = pd.read_parquet(
        PROJECT_ROOT
        / "data"
        / "processed"
        / "products_clean.parquet"
    )

    source_indices = (
        dataset.source_row_indices
    )

    aligned = df.iloc[
        source_indices
    ].copy()

    aligned = aligned.reset_index(
        drop=True
    )

    test_mask = (
        aligned["MonthNum"]
        .between(9, 10)
        .to_numpy()
    )

    test_positions = np.flatnonzero(
        test_mask
    )

    print()
    print(
        f"Test products: "
        f"{len(test_positions)}"
    )

    # -------------------------------------------------------------
    # Load policy
    # -------------------------------------------------------------

    agent = load_agent(
        CHECKPOINT
    )

    print(
        "Transfer PPO checkpoint loaded."
    )

    # -------------------------------------------------------------
    # Create environment
    # -------------------------------------------------------------

    environment = (
        DynamicPricingEnvironment()
    )

    alpha = (
        environment.reward_config.alpha
    )

    beta = (
        environment.reward_config.beta
    )

    print()
    print(
        "Reward configuration:"
    )
    print(
        f"  Alpha (margin): "
        f"{alpha}"
    )
    print(
        f"  Beta (quantity): "
        f"{beta}"
    )
    print(
        "  Reward type: normalized relative reward"
    )

    # -------------------------------------------------------------
    # Step 1:
    # Obtain the original deterministic policy predictions
    # -------------------------------------------------------------

    base_records = []

    print()
    print(
        "Collecting baseline test predictions..."
    )

    for position in test_positions:

        state = dataset.states[
            position
        ]

        product = dataset.products[
            position
        ]

        environment.reset(
            state=state,
            product=product,
        )

        baseline_sales = (
            product.last_monthly_sales
        )

        baseline_margin = (
            environment
            .observed_outcome
            .estimated_margin
        )

        if baseline_margin is None:
            continue

        ppo_action = (
            select_deterministic_action(
                agent,
                state,
            )
        )

        environment_action = (
            ppo_action + 1
        )

        discount = float(
            DISCOUNT_BINS[
                ppo_action
            ]
        )

        # ---------------------------------------------------------
        # Original frozen prediction
        # ---------------------------------------------------------

        predicted_growth = float(
            environment.simulator.predict(
                state,
                discount,
            )
        )

        # ---------------------------------------------------------
        # Original economic outcome
        # ---------------------------------------------------------

        original_outcome = (
            calculate_pricing_outcome(
                observed_price=(
                    product.observed_price
                ),
                observed_discount=(
                    product.observed_discount
                ),
                last_monthly_sales=(
                    product.last_monthly_sales
                ),
                predicted_growth_rate=(
                    predicted_growth
                ),
                candidate_discount=(
                    discount
                ),
                gross_margin_rate=(
                    product.gross_margin_rate
                ),
            )
        )

        delta_quantity = (
            original_outcome.estimated_sales
            - baseline_sales
        )

        delta_margin = (
            original_outcome.estimated_margin
            - baseline_margin
        )

        original_reward = (
            calculate_relative_reward(
                delta_margin=delta_margin,
                delta_quantity=delta_quantity,
                baseline_margin=baseline_margin,
                baseline_quantity=baseline_sales,
                alpha=alpha,
                beta=beta,
            )
        )

        base_records.append(
            {
                "position": int(
                    position
                ),
                "source_row": int(
                    source_indices[
                        position
                    ]
                ),
                "ppo_action": int(
                    ppo_action
                ),
                "environment_action": int(
                    environment_action
                ),
                "discount": discount,
                "predicted_growth": (
                    predicted_growth
                ),
                "baseline_sales": float(
                    baseline_sales
                ),
                "baseline_margin": float(
                    baseline_margin
                ),
                "original_sales": float(
                    original_outcome
                    .estimated_sales
                ),
                "original_revenue": float(
                    original_outcome
                    .estimated_revenue
                ),
                "original_margin": float(
                    original_outcome
                    .estimated_margin
                ),
                "original_reward": float(
                    original_reward
                ),
                "observed_discount": float(
                    product.observed_discount
                ),
                "observed_price": float(
                    product.observed_price
                ),
                "last_monthly_sales": float(
                    product.last_monthly_sales
                ),
                "gross_margin_rate": (
                    None
                    if product.gross_margin_rate
                    is None
                    else float(
                        product.gross_margin_rate
                    )
                ),
            }
        )

    if not base_records:
        raise RuntimeError(
            "No valid test records were produced."
        )

    base_df = pd.DataFrame(
        base_records
    )

    # -------------------------------------------------------------
    # Prediction standard deviation
    # -------------------------------------------------------------

    prediction_std = float(
        base_df[
            "predicted_growth"
        ].std(
            ddof=0
        )
    )

    prediction_mean = float(
        base_df[
            "predicted_growth"
        ].mean()
    )

    print()
    print(
        "Test predicted-growth statistics:"
    )

    print(
        f"  Mean: "
        f"{prediction_mean:.6f}"
    )

    print(
        f"  Std:  "
        f"{prediction_std:.6f}"
    )

    print(
        f"  Min:  "
        f"{base_df['predicted_growth'].min():.6f}"
    )

    print(
        f"  Max:  "
        f"{base_df['predicted_growth'].max():.6f}"
    )

    # -------------------------------------------------------------
    # Generate one standard-normal noise vector.
    #
    # Using the same underlying draws across sigma values makes
    # comparisons between noise levels cleaner.
    # -------------------------------------------------------------

    rng = np.random.default_rng(
        SEED
    )

    standard_noise = rng.normal(
        loc=0.0,
        scale=1.0,
        size=len(base_df),
    )

    # -------------------------------------------------------------
    # Evaluate noise levels
    # -------------------------------------------------------------

    results = []

    original_mean_reward = float(
        base_df[
            "original_reward"
        ].mean()
    )

    original_mean_growth = float(
        base_df[
            "predicted_growth"
        ].mean()
    )

    for noise_fraction in NOISE_LEVELS:

        noise_std = (
            noise_fraction
            * prediction_std
        )

        noise = (
            standard_noise
            * noise_std
        )

        noisy_growth = (
            base_df[
                "predicted_growth"
            ].to_numpy()
            + noise
        )

        rewards = []
        sales_values = []
        revenue_values = []
        margin_values = []

        for index, record in base_df.iterrows():

            growth = float(
                noisy_growth[index]
            )

            outcome = (
                calculate_pricing_outcome(
                    observed_price=(
                        record[
                            "observed_price"
                        ]
                    ),
                    observed_discount=(
                        record[
                            "observed_discount"
                        ]
                    ),
                    last_monthly_sales=(
                        record[
                            "last_monthly_sales"
                        ]
                    ),
                    predicted_growth_rate=growth,
                    candidate_discount=(
                        record[
                            "discount"
                        ]
                    ),
                    gross_margin_rate=(
                        record[
                            "gross_margin_rate"
                        ]
                    ),
                )
            )

            delta_quantity = (
                outcome.estimated_sales
                - record[
                    "baseline_sales"
                ]
            )

            delta_margin = (
                outcome.estimated_margin
                - record[
                    "baseline_margin"
                ]
            )

            reward = (
                calculate_relative_reward(
                    delta_margin=delta_margin,
                    delta_quantity=delta_quantity,
                    baseline_margin=(
                        record[
                            "baseline_margin"
                        ]
                    ),
                    baseline_quantity=(
                        record[
                            "baseline_sales"
                        ]
                    ),
                    alpha=alpha,
                    beta=beta,
                )
            )

            rewards.append(
                float(reward)
            )

            sales_values.append(
                float(
                    outcome.estimated_sales
                )
            )

            revenue_values.append(
                float(
                    outcome.estimated_revenue
                )
            )

            margin_values.append(
                float(
                    outcome.estimated_margin
                )
            )

        rewards = np.asarray(
            rewards,
            dtype=np.float64,
        )

        sales_values = np.asarray(
            sales_values,
            dtype=np.float64,
        )

        revenue_values = np.asarray(
            revenue_values,
            dtype=np.float64,
        )

        margin_values = np.asarray(
            margin_values,
            dtype=np.float64,
        )

        mean_reward = float(
            rewards.mean()
        )

        reward_std = float(
            rewards.std(
                ddof=0
            )
        )

        retention = (
            mean_reward
            / max(
                abs(
                    original_mean_reward
                ),
                EPSILON,
            )
            * 100.0
        )

        reward_change = (
            mean_reward
            - original_mean_reward
        )

        reward_change_percent = (
            reward_change
            / max(
                abs(
                    original_mean_reward
                ),
                EPSILON,
            )
            * 100.0
        )

        mean_growth = float(
            noisy_growth.mean()
        )

        mean_absolute_noise = float(
            np.abs(
                noise
            ).mean()
        )

        print()
        print(
            "-" * 72
        )

        print(
            f"Noise level: "
            f"{noise_fraction:.2f} × Std"
        )

        print(
            f"  Absolute noise std: "
            f"{noise_std:.6f}"
        )

        print(
            f"  Mean growth: "
            f"{mean_growth:.6f}"
        )

        print(
            f"  Mean |noise|: "
            f"{mean_absolute_noise:.6f}"
        )

        print(
            f"  Mean reward: "
            f"{mean_reward:.6f}"
        )

        print(
            f"  Reward std: "
            f"{reward_std:.6f}"
        )

        print(
            f"  Reward retention: "
            f"{retention:.2f}%"
        )

        print(
            f"  Reward change: "
            f"{reward_change_percent:.2f}%"
        )

        print(
            f"  Mean estimated sales: "
            f"{sales_values.mean():.6f}"
        )

        print(
            f"  Mean estimated revenue: "
            f"{revenue_values.mean():.6f}"
        )

        print(
            f"  Mean estimated margin: "
            f"{margin_values.mean():.6f}"
        )

        results.append(
            {
                "noise_fraction": float(
                    noise_fraction
                ),
                "noise_std_absolute": float(
                    noise_std
                ),
                "prediction_std_test": float(
                    prediction_std
                ),
                "mean_predicted_growth": float(
                    mean_growth
                ),
                "mean_absolute_noise": float(
                    mean_absolute_noise
                ),
                "mean_reward": float(
                    mean_reward
                ),
                "reward_std": float(
                    reward_std
                ),
                "reward_retention_percent": float(
                    retention
                ),
                "reward_change": float(
                    reward_change
                ),
                "reward_change_percent": float(
                    reward_change_percent
                ),
                "mean_estimated_sales": float(
                    sales_values.mean()
                ),
                "mean_estimated_revenue": float(
                    revenue_values.mean()
                ),
                "mean_estimated_margin": float(
                    margin_values.mean()
                ),
                "evaluated_products": int(
                    len(rewards)
                ),
            }
        )

    # -------------------------------------------------------------
    # Save results
    # -------------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    results_df.to_csv(
        RESULTS_PATH,
        index=False,
    )

    # -------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------

    finite_results = np.isfinite(
        results_df.select_dtypes(
            include=[np.number]
        ).to_numpy()
    ).all()

    noise_levels_valid = (
        results_df[
            "noise_fraction"
        ].tolist()
        == NOISE_LEVELS
    )

    product_counts_valid = (
        results_df[
            "evaluated_products"
        ].nunique()
        == 1
        and
        int(
            results_df[
                "evaluated_products"
            ].iloc[0]
        )
        == len(base_df)
    )

    # sigma=0 should reproduce the original
    # deterministic policy reward.

    sigma_zero_reward = float(
        results_df.loc[
            results_df[
                "noise_fraction"
            ] == 0.0,
            "mean_reward",
        ].iloc[0]
    )

    sigma_zero_match = np.isclose(
        sigma_zero_reward,
        original_mean_reward,
        atol=1e-6,
    )

    print()
    print("=" * 72)
    print("PHASE 15 ROBUSTNESS VALIDATION")
    print("=" * 72)

    print(
        f"Finite numeric results: "
        f"{'PASS' if finite_results else 'FAIL'}"
    )

    print(
        f"Noise levels:           "
        f"{'PASS' if noise_levels_valid else 'FAIL'}"
    )

    print(
        f"Product count:          "
        f"{'PASS' if product_counts_valid else 'FAIL'}"
    )

    print(
        f"Sigma=0 baseline match: "
        f"{'PASS' if sigma_zero_match else 'FAIL'}"
    )

    if not all(
        [
            finite_results,
            noise_levels_valid,
            product_counts_valid,
            sigma_zero_match,
        ]
    ):
        raise RuntimeError(
            "Phase 15 robustness validation failed."
        )

    # -------------------------------------------------------------
    # Save metadata
    # -------------------------------------------------------------

    metadata = {
        "phase": 15,
        "method": (
            "Zero-mean Gaussian noise injected "
            "into frozen sales-growth predictions"
        ),
        "checkpoint": str(
            CHECKPOINT
        ),
        "state_dim": STATE_DIM,
        "action_dim": ACTION_DIM,
        "discount_bins": DISCOUNT_BINS,
        "reward_alpha": alpha,
        "reward_beta": beta,
        "test_products": int(
            len(base_df)
        ),
        "prediction_mean_test": (
            prediction_mean
        ),
        "prediction_std_test": (
            prediction_std
        ),
        "original_mean_reward": (
            original_mean_reward
        ),
        "original_mean_growth": (
            original_mean_growth
        ),
        "noise_fractions": NOISE_LEVELS,
        "random_seed": SEED,
        "same_standard_noise_across_levels": True,
        "sentiment": (
            "not applicable; review text unavailable"
        ),
        "simulation_only": True,
        "validation": {
            "finite_results": bool(
                finite_results
            ),
            "noise_levels_valid": bool(
                noise_levels_valid
            ),
            "product_count_valid": bool(
                product_counts_valid
            ),
            "sigma_zero_baseline_match": bool(
                sigma_zero_match
            ),
        },
    }

    with open(
        SUMMARY_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2,
        )

    # -------------------------------------------------------------
    # Final table
    # -------------------------------------------------------------

    print()
    print("=" * 72)
    print("PHASE 15 ROBUSTNESS SUMMARY")
    print("=" * 72)

    print()

    print(
        results_df[
            [
                "noise_fraction",
                "noise_std_absolute",
                "mean_reward",
                "reward_std",
                "reward_retention_percent",
                "reward_change_percent",
                "mean_predicted_growth",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "Phase 15 robustness validation: PASS"
    )

    print()
    print(
        f"Saved results: {RESULTS_PATH}"
    )

    print(
        f"Saved metadata: {SUMMARY_PATH}"
    )


if __name__ == "__main__":
    main()