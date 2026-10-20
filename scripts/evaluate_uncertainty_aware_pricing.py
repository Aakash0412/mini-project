from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.environment.ppo_dataset import load_ppo_dataset
from src.environment.pricing_economics import calculate_pricing_outcome
from src.environment.reward import get_reward_config
from src.models.ppo_agent import PPOAgent
from src.models.sales_predictor import load_frozen_predictor

DISCOUNTS = np.array([5, 10, 15, 20, 25, 30, 35, 40], dtype=np.float64)
STATE_DIM = 111
ACTION_DIM = len(DISCOUNTS)
EPSILON = 1e-8


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare frozen PPO with MC-dropout risk-adjusted pricing."
    )
    parser.add_argument(
        "--checkpoint", type=Path,
        default=Path("models/ppo_transfer/ppo_transfer_supported_8action_1000_updates.pth"),
    )
    parser.add_argument("--reward-config", choices=("sales", "balanced", "profit"), default="sales")
    parser.add_argument("--mc-samples", type=int, default=20)
    parser.add_argument("--lambda-penalty", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/uncertainty_aware_pricing"))
    return parser.parse_args()


def load_actor(checkpoint_path: Path) -> nn.Module:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"PPO checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if int(checkpoint.get("action_dim", ACTION_DIM)) != ACTION_DIM:
        raise ValueError(f"Expected an {ACTION_DIM}-action PPO checkpoint.")
    agent = PPOAgent(
        state_dim=STATE_DIM, action_dim=ACTION_DIM,
        actor_lr=8e-4, critic_lr=8e-4,
    )
    agent.actor.load_state_dict(checkpoint["actor_state_dict"])
    agent.actor.eval()
    return agent.actor


def deterministic_actions(actor: nn.Module, states: np.ndarray, batch_size: int) -> np.ndarray:
    result = []
    with torch.no_grad():
        for start in range(0, len(states), batch_size):
            tensor = torch.as_tensor(states[start:start + batch_size], dtype=torch.float32)
            output = actor(tensor)
            logits = output.logits if hasattr(output, "logits") else output
            result.extend(torch.argmax(logits, dim=-1).cpu().numpy().tolist())
    return np.asarray(result, dtype=np.int64)


def predict_mc_growth(
    model: nn.Module,
    states: np.ndarray,
    samples: int,
    batch_size: int,
    seed: int,
) -> np.ndarray:
    """Return growth predictions shaped (MC samples, products, 8 discounts)."""
    if samples < 2:
        raise ValueError("--mc-samples must be at least 2.")
    model.eval()
    # Enable dropout only; BatchNorm remains in eval mode and weights stay frozen.
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()

    rng = np.random.default_rng(seed)
    predictions = []
    for sample in range(samples):
        chunks = []
        with torch.no_grad():
            for start in range(0, len(states), batch_size):
                batch_states = np.asarray(states[start:start + batch_size], dtype=np.float32)
                n = len(batch_states)
                repeated_states = np.repeat(batch_states, ACTION_DIM, axis=0)
                repeated_discounts = np.tile(DISCOUNTS.astype(np.float32), n).reshape(-1, 1)
                state_tensor = torch.as_tensor(repeated_states, dtype=torch.float32)
                discount_tensor = torch.as_tensor(repeated_discounts, dtype=torch.float32)
                pred = model.predict(state_tensor, discount_tensor).cpu().numpy()
                chunks.append(pred.reshape(n, ACTION_DIM))
        predictions.append(np.concatenate(chunks, axis=0))
        # Distinct, reproducible dropout masks between MC passes.
        torch.manual_seed(int(rng.integers(0, 2**31 - 1)))

    model.eval()
    return np.stack(predictions, axis=0)


def reward_samples_for_product(growth: np.ndarray, product, alpha: float, beta: float) -> np.ndarray:
    """Vectorize existing economics/reward formulas; growth shape is (MC, 8)."""
    observed_price = float(product.observed_price)
    observed_discount = float(product.observed_discount)
    baseline_sales = float(product.last_monthly_sales)
    margin_rate = float(product.gross_margin_rate)

    baseline = calculate_pricing_outcome(
        observed_price=observed_price,
        observed_discount=observed_discount,
        last_monthly_sales=baseline_sales,
        predicted_growth_rate=0.0,
        candidate_discount=observed_discount,
        gross_margin_rate=margin_rate,
    )
    baseline_margin = float(baseline.estimated_margin)

    reference_price = observed_price / (1.0 - observed_discount / 100.0)
    candidate_prices = reference_price * (1.0 - DISCOUNTS / 100.0)
    estimated_sales = np.maximum(
        0.0, baseline_sales * (1.0 + growth / 100.0)
    )
    estimated_revenue = estimated_sales * candidate_prices[None, :]
    estimated_margin = estimated_revenue * margin_rate / 100.0

    relative_quantity = (
        estimated_sales - baseline_sales
    ) / max(abs(baseline_sales), EPSILON)
    relative_margin = (
        estimated_margin - baseline_margin
    ) / max(abs(baseline_margin), EPSILON)
    return alpha * relative_margin + beta * relative_quantity


def main():
    args = parse_args()
    if args.lambda_penalty < 0:
        raise ValueError("--lambda-penalty must be nonnegative.")
    if args.mc_samples < 2:
        raise ValueError("--mc-samples must be at least 2.")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_ppo_dataset()
    dataframe = pd.read_parquet("data/processed/products_clean.parquet")
    aligned = dataframe.iloc[dataset.source_row_indices].copy().reset_index(drop=True)
    states = np.asarray(dataset.states, dtype=np.float32)
    products = list(dataset.products)

    if len(states) != len(products) or len(states) != len(aligned):
        raise RuntimeError("Dataset states, products, and dataframe rows are misaligned.")

    train_mask = aligned["MonthNum"].between(1, 6).to_numpy()
    val_mask = aligned["MonthNum"].between(7, 8).to_numpy()
    test_mask = aligned["MonthNum"].between(9, 10).to_numpy()
    if not val_mask.any() or not test_mask.any():
        raise RuntimeError("Expected non-empty validation months 7-8 and test months 9-10.")

    reward_config = get_reward_config(args.reward_config)
    alpha, beta = float(reward_config.alpha), float(reward_config.beta)
    predictor = load_frozen_predictor("models/sales_predictor", device=torch.device("cpu"))
    for parameter in predictor.parameters():
        parameter.requires_grad_(False)
    actor = load_actor(args.checkpoint)

    # Validation calibration is restricted to historical observed-discount pairs.
    val_positions = np.flatnonzero(val_mask)
    val_growth = predict_mc_growth(
        predictor, states[val_positions], args.mc_samples, args.batch_size, args.seed + 100
    )
    val_mean = val_growth.mean(axis=0)
    val_std = val_growth.std(axis=0, ddof=1)
    actual_growth = pd.to_numeric(
        aligned.iloc[val_positions]["Estimated Monthly Sales Growth Rate"], errors="coerce"
    ).to_numpy(dtype=float)
    observed_discounts = pd.to_numeric(
        aligned.iloc[val_positions]["Discount"], errors="coerce"
    ).to_numpy(dtype=float)
    discount_index = {int(discount): i for i, discount in enumerate(DISCOUNTS)}
    calibration_rows = []

    for local_i, source_position in enumerate(val_positions):
        discount = observed_discounts[local_i]
        actual = actual_growth[local_i]
        if not np.isfinite(discount) or not np.isfinite(actual) or not discount.is_integer():
            continue
        action = discount_index.get(int(discount))
        if action is None:
            continue
        mean_prediction = float(val_mean[local_i, action])
        uncertainty = float(val_std[local_i, action])
        calibration_rows.append({
            "source_row": int(dataset.source_row_indices[source_position]),
            "observed_discount": float(discount),
            "actual_growth": float(actual),
            "mc_mean_growth": mean_prediction,
            "mc_std_growth": uncertainty,
            "absolute_error": abs(mean_prediction - float(actual)),
        })

    calibration = pd.DataFrame(calibration_rows)
    if calibration.empty:
        raise RuntimeError("No validation rows matched observed discounts from 5% to 40%.")
    bins = min(10, max(2, len(calibration) // 20))
    calibration["uncertainty_decile"] = pd.qcut(
        calibration["mc_std_growth"].rank(method="first"),
        q=bins, labels=False, duplicates="drop",
    ) + 1
    calibration_summary = calibration.groupby("uncertainty_decile", observed=True).agg(
        samples=("absolute_error", "size"),
        mean_mc_std_growth=("mc_std_growth", "mean"),
        mean_absolute_error=("absolute_error", "mean"),
        median_absolute_error=("absolute_error", "median"),
    ).reset_index()
    calibration.to_csv(args.output_dir / "validation_uncertainty_calibration_rows.csv", index=False)
    calibration_summary.to_csv(
        args.output_dir / "validation_uncertainty_calibration_summary.csv", index=False
    )

    # Test set: risk score = mean simulated reward - lambda * reward standard deviation.
    test_positions = np.flatnonzero(test_mask)
    test_states = states[test_positions]
    test_products = [products[i] for i in test_positions]
    test_growth = predict_mc_growth(
        predictor, test_states, args.mc_samples, args.batch_size, args.seed + 200
    )
    reward_tensor = np.empty((args.mc_samples, len(test_positions), ACTION_DIM), dtype=np.float64)
    for i, product in enumerate(test_products):
        reward_tensor[:, i, :] = reward_samples_for_product(
            test_growth[:, i, :], product, alpha, beta
        )

    mean_reward = reward_tensor.mean(axis=0)
    reward_std = reward_tensor.std(axis=0, ddof=1)
    ppo_action = deterministic_actions(actor, test_states, args.batch_size)
    risk_score = mean_reward - args.lambda_penalty * reward_std
    risk_action = np.argmax(risk_score, axis=1)
    oracle_action = np.argmax(mean_reward, axis=1)

    rows = []
    for i, source_position in enumerate(test_positions):
        pa, ra, oa = int(ppo_action[i]), int(risk_action[i]), int(oracle_action[i])
        rows.append({
            "source_row": int(dataset.source_row_indices[source_position]),
            "ppo_action": pa,
            "ppo_discount": float(DISCOUNTS[pa]),
            "ppo_mean_simulated_reward": float(mean_reward[i, pa]),
            "ppo_reward_std": float(reward_std[i, pa]),
            "risk_action": ra,
            "risk_discount": float(DISCOUNTS[ra]),
            "risk_mean_simulated_reward": float(mean_reward[i, ra]),
            "risk_reward_std": float(reward_std[i, ra]),
            "risk_score": float(risk_score[i, ra]),
            "oracle_action": oa,
            "oracle_discount": float(DISCOUNTS[oa]),
            "oracle_mean_simulated_reward": float(mean_reward[i, oa]),
            "action_changed": bool(pa != ra),
            "ppo_oracle_gap": float(mean_reward[i, oa] - mean_reward[i, pa]),
            "risk_oracle_gap": float(mean_reward[i, oa] - mean_reward[i, ra]),
            "mean_uncertainty_across_actions": float(reward_std[i].mean()),
        })

    results = pd.DataFrame(rows)
    results.to_csv(args.output_dir / "test_policy_comparison.csv", index=False)
    ppo_mean = float(results["ppo_mean_simulated_reward"].mean())
    risk_mean = float(results["risk_mean_simulated_reward"].mean())
    oracle_mean = float(results["oracle_mean_simulated_reward"].mean())
    correlation = calibration["mc_std_growth"].corr(
        calibration["absolute_error"], method="spearman"
    )
    summary = {
        "checkpoint": str(args.checkpoint),
        "reward_config": args.reward_config,
        "alpha_margin": alpha,
        "beta_quantity": beta,
        "mc_samples": args.mc_samples,
        "lambda_penalty": args.lambda_penalty,
        "validation_calibration_samples": int(len(calibration)),
        "validation_spearman_uncertainty_vs_absolute_error": (
            float(correlation) if pd.notna(correlation) else None
        ),
        "test_samples": int(len(results)),
        "mean_ppo_simulated_reward": ppo_mean,
        "mean_risk_selected_simulated_reward": risk_mean,
        "mean_oracle_simulated_reward": oracle_mean,
        "mean_ppo_oracle_gap": float(results["ppo_oracle_gap"].mean()),
        "mean_risk_oracle_gap": float(results["risk_oracle_gap"].mean()),
        "action_change_rate": float(results["action_changed"].mean()),
        "mean_ppo_reward_std": float(results["ppo_reward_std"].mean()),
        "mean_risk_selected_reward_std": float(results["risk_reward_std"].mean()),
        "risk_reward_retention_vs_ppo_percent": float(100.0 * risk_mean / max(abs(ppo_mean), EPSILON)),
        "caveats": [
            "Rewards are model-based counterfactual estimates, not live-market outcomes.",
            "MC-dropout spread is a heuristic uncertainty proxy, not automatically a calibrated prediction interval.",
            "Validation calibration measures historical observed-discount pairs only.",
            "The oracle is optimal under this learned simulator, not ground truth.",
            "The environment is one-step and margin is a dataset-derived proxy.",
        ],
    }
    with open(args.output_dir / "summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print("=" * 68)
    print("UNCERTAINTY-AWARE PRICING EVALUATION")
    print("=" * 68)
    print(f"Validation calibration rows: {len(calibration)}")
    print(f"Validation Spearman (uncertainty vs absolute error): {correlation:.4f}")
    print(f"Test products: {len(results)}")
    print(f"Mean PPO simulated reward:     {ppo_mean:.6f}")
    print(f"Mean risk-selected reward:     {risk_mean:.6f}")
    print(f"Mean oracle simulated reward:  {oracle_mean:.6f}")
    print(f"Mean PPO oracle gap:           {results['ppo_oracle_gap'].mean():.6f}")
    print(f"Mean risk oracle gap:          {results['risk_oracle_gap'].mean():.6f}")
    print(f"Action change rate:            {results['action_changed'].mean() * 100:.2f}%")
    print(f"Risk reward retention vs PPO:  {summary['risk_reward_retention_vs_ppo_percent']:.3f}%")
    print(f"Outputs saved in: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
