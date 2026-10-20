from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.environment.ppo_dataset import load_ppo_dataset
from src.models.ppo_agent import PPOAgent

CHECKPOINT = ROOT / "models/ppo_reward_experiments/profit_entropy_005/ppo_transfer_supported_8action_200_updates.pth"
OUTPUT_DIR = ROOT / "models/ppo_reward_experiments/profit_entropy_005/policy_diagnostics"
OUTPUT_CSV = OUTPUT_DIR / "test_policy_probabilities.csv"
DISCOUNTS = [5, 10, 15, 20, 25, 30, 35, 40]

def main():
    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"Missing checkpoint: {CHECKPOINT}")

    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    actor_state = checkpoint.get("actor_state_dict")
    if not isinstance(actor_state, dict):
        raise ValueError("Checkpoint does not contain actor_state_dict")

    output_weight = actor_state.get("network.4.weight")
    if output_weight is None or tuple(output_weight.shape) != (8, 128):
        shape = None if output_weight is None else tuple(output_weight.shape)
        raise ValueError(f"Expected an 8-action output layer of shape (8, 128), got {shape}")

    print("Checkpoint architecture verified: 111 inputs, 8 actions")

    dataset = load_ppo_dataset()
    df = pd.read_parquet(ROOT / "data/processed/products_clean.parquet")
    source_indices = np.asarray(dataset.source_row_indices)
    aligned = df.iloc[source_indices]
    test_positions = np.flatnonzero(aligned["MonthNum"].between(9, 10).to_numpy())

    agent = PPOAgent(state_dim=111, action_dim=8)
    agent.actor.load_state_dict(checkpoint["actor_state_dict"])
    agent.actor.eval()

    states = torch.as_tensor(dataset.states[test_positions], dtype=torch.float32)
    with torch.no_grad():
        logits = agent.actor(states)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()

    entropy = -(probs * np.log(np.clip(probs, 1e-12, 1.0))).sum(axis=1)
    actions = probs.argmax(axis=1)

    result = pd.DataFrame({
        "source_row": source_indices[test_positions],
        "selected_discount": np.asarray(DISCOUNTS)[actions],
        "policy_entropy": entropy,
        "max_probability": probs.max(axis=1),
    })
    for i, discount in enumerate(DISCOUNTS):
        result[f"prob_{discount}pct"] = probs[:, i]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_CSV, index=False)

    print("=" * 68)
    print("PPO STATE-SENSITIVITY DIAGNOSTIC")
    print("=" * 68)
    print(f"Checkpoint: {CHECKPOINT}")
    print(f"Test products: {len(result)}")
    print(f"Saved: {OUTPUT_CSV}")

    print("\nMean action probabilities:")
    for discount in DISCOUNTS:
        col = f"prob_{discount}pct"
        print(f"  {discount:>2}%: mean={result[col].mean():.6f}, std={result[col].std():.6f}")

    print("\nDeterministic selected discounts:")
    print(result["selected_discount"].value_counts().sort_index().to_string())

    print("\nPolicy confidence and entropy:")
    print(f"  Mean entropy: {result['policy_entropy'].mean():.6f} (maximum possible: {np.log(8):.6f})")
    print(f"  Median entropy: {result['policy_entropy'].median():.6f}")
    print(f"  Mean maximum probability: {result['max_probability'].mean():.6f}")

    probability_cols = [f"prob_{d}pct" for d in DISCOUNTS]
    probability_matrix = result[probability_cols].to_numpy()
    mean_policy = probability_matrix.mean(axis=0)
    distances = np.linalg.norm(probability_matrix - mean_policy, axis=1)
    print("\nState sensitivity:")
    print(f"  Mean per-action probability std: {probability_matrix.std(axis=0).mean():.6f}")
    print(f"  Mean distance from average policy: {distances.mean():.6f}")
    print(f"  Maximum distance from average policy: {distances.max():.6f}")

    print("\nValidity checks:")
    print(f"  Probabilities sum to 1: {np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)}")
    print(f"  All probabilities finite: {np.isfinite(probs).all()}")
    print(f"  All selected discounts valid: {set(result['selected_discount']).issubset(DISCOUNTS)}")

if __name__ == "__main__":
    main()
