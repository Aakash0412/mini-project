from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.environment.ppo_dataset import load_ppo_dataset
from src.environment.pricing_environment import DynamicPricingEnvironment
from src.environment.reward import SALES_REWARD


CHECKPOINT = PROJECT_ROOT / "models" / "ddqn" / "ddqn_sales_seed42.pth"
OUTPUT = (
    PROJECT_ROOT
    / "reports"
    / "algorithm_comparison"
    / "ddqn_test_evaluation.csv"
)
DISCOUNT_BINS = [5, 10, 15, 20, 25, 30, 35, 40]
EPSILON = 1e-8
STATE_DIM = 111
ACTION_DIM = 9


class QNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(STATE_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, ACTION_DIM),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.network(states)


def relative_reward(
    delta_margin: float,
    delta_sales: float,
    baseline_margin: float,
    baseline_sales: float,
) -> float:
    relative_margin = delta_margin / max(abs(baseline_margin), EPSILON)
    relative_sales = delta_sales / max(abs(baseline_sales), EPSILON)
    return float(
        SALES_REWARD.alpha * relative_margin
        + SALES_REWARD.beta * relative_sales
    )


def main() -> None:
    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"DDQN checkpoint not found: {CHECKPOINT}")

    if OUTPUT.exists():
        raise FileExistsError(
            f"Evaluation output already exists: {OUTPUT}. "
            "Preserve it and choose a new output name before rerunning."
        )

    # This checkpoint is a trusted local artifact created by our trainer.
    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=True,
    )

    if checkpoint.get("algorithm") != "DDQN":
        raise ValueError("Checkpoint is not identified as DDQN.")
    if checkpoint.get("state_dim") != STATE_DIM:
        raise ValueError("Unexpected checkpoint state dimension.")
    if checkpoint.get("action_dim") != ACTION_DIM:
        raise ValueError("Unexpected checkpoint action dimension.")

    online = QNetwork().cpu()
    online.load_state_dict(checkpoint["online_state_dict"])
    online.eval()

    dataset = load_ppo_dataset()
    products_df = pd.read_parquet(
        PROJECT_ROOT / "data" / "processed" / "products_clean.parquet"
    )
    aligned = products_df.iloc[dataset.source_row_indices].copy()

    positions = np.flatnonzero(
        aligned["MonthNum"].between(9, 10).to_numpy()
    )
    if len(positions) == 0:
        raise RuntimeError("No test products found for MonthNum 9-10.")

    environment = DynamicPricingEnvironment(reward_config=SALES_REWARD)
    results = []

    print("DDQN held-out test evaluation")
    print("Checkpoint:", CHECKPOINT)
    print("Test products before margin validation:", len(positions))
    print("Evaluation discount support:", DISCOUNT_BINS)
    print("Action 0 (0% discount) excluded at evaluation for parity.")

    with torch.no_grad():
        for position in positions:
            state = dataset.states[position]
            product = dataset.products[position]

            environment.reset(state=state, product=product)
            baseline_sales = float(product.last_monthly_sales)
            baseline_margin = environment.observed_outcome.estimated_margin

            if baseline_margin is None or not np.isfinite(baseline_margin):
                continue

            state_tensor = torch.as_tensor(
                state, dtype=torch.float32
            ).unsqueeze(0)
            q_values = online(state_tensor).cpu().numpy().reshape(-1)

            if q_values.shape != (ACTION_DIM,) or not np.isfinite(q_values).all():
                raise ValueError(
                    f"Invalid DDQN Q-values at dataset position {position}"
                )

            # Indices 1..8 correspond to discounts 5..40%.
            supported_action = int(np.argmax(q_values[1:]) + 1)
            selected_discount = float(DISCOUNT_BINS[supported_action - 1])

            selected_outcome = environment.predict_action(supported_action)
            delta_sales = float(
                selected_outcome.estimated_sales - baseline_sales
            )
            delta_margin = float(
                selected_outcome.estimated_margin - baseline_margin
            )
            reward = relative_reward(
                delta_margin, delta_sales, baseline_margin, baseline_sales
            )

            oracle_candidates = []
            for action, outcome in environment.predict_all_actions().items():
                if action == 0 or outcome.estimated_margin is None:
                    continue

                candidate_reward = relative_reward(
                    float(outcome.estimated_margin - baseline_margin),
                    float(outcome.estimated_sales - baseline_sales),
                    baseline_margin,
                    baseline_sales,
                )
                oracle_candidates.append(
                    {
                        "discount": float(outcome.discount_percent),
                        "reward": candidate_reward,
                    }
                )

            if not oracle_candidates:
                continue

            oracle = max(oracle_candidates, key=lambda item: item["reward"])

            results.append(
                {
                    "split": "test",
                    "source_row": int(dataset.source_row_indices[position]),
                    "ddqn_discount": selected_discount,
                    "ddqn_reward": reward,
                    "ddqn_sales": float(selected_outcome.estimated_sales),
                    "ddqn_revenue": float(selected_outcome.estimated_revenue),
                    "ddqn_margin": float(selected_outcome.estimated_margin),
                    "baseline_sales": baseline_sales,
                    "baseline_margin": float(baseline_margin),
                    "oracle_discount": oracle["discount"],
                    "oracle_reward": oracle["reward"],
                    "oracle_gap": oracle["reward"] - reward,
                    "oracle_action_match": (
                        selected_discount == oracle["discount"]
                    ),
                }
            )

    results_df = pd.DataFrame(results)
    if results_df.empty:
        raise RuntimeError("No valid DDQN test results were produced.")

    numeric_columns = results_df.select_dtypes(include=[np.number]).columns
    if not np.isfinite(results_df[numeric_columns].to_numpy()).all():
        raise ValueError("Non-finite values detected in evaluation results.")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(OUTPUT, index=False)

    print("\nTEST SUMMARY")
    print("Evaluated products:", len(results_df))
    print(f"DDQN mean reward: {results_df['ddqn_reward'].mean():.6f}")
    print(f"Oracle mean reward: {results_df['oracle_reward'].mean():.6f}")
    print(f"Mean oracle gap: {results_df['oracle_gap'].mean():.6f}")
    print(
        "Oracle action match:",
        f"{100 * results_df['oracle_action_match'].mean():.2f}%",
    )
    print("\nDDQN discount distribution:")
    print(
        results_df["ddqn_discount"]
        .value_counts(normalize=True)
        .reindex(DISCOUNT_BINS, fill_value=0)
        .mul(100)
        .round(2)
        .to_string()
    )
    print("\nSaved:", OUTPUT)


if __name__ == "__main__":
    main()
