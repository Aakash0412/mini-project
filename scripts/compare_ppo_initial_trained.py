from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path.cwd()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.environment.ppo_dataset import load_ppo_dataset
from src.models.ppo_agent import PPOAgent

INITIAL_PATH = ROOT / "models/transfer/ppo_warmstart/best_warmstart_actor.pth"
TRAINED_PATH = ROOT / "models/ppo_reward_experiments/profit_entropy_005/ppo_transfer_supported_8action_200_updates.pth"
OUTPUT_PATH = ROOT / "models/ppo_reward_experiments/profit_entropy_005/policy_diagnostics/initial_vs_trained.csv"
DISCOUNTS = np.array([5, 10, 15, 20, 25, 30, 35, 40])

def load_actor(path, key):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint[key]
    actor = PPOAgent(state_dim=111, action_dim=8).actor
    actor.load_state_dict(state)
    actor.eval()
    return actor

def get_probabilities(actor, states):
    with torch.no_grad():
        logits = actor(torch.as_tensor(states, dtype=torch.float32))
        return torch.softmax(logits, dim=-1).cpu().numpy()

def summarize(label, probs):
    actions = DISCOUNTS[probs.argmax(axis=1)]
    entropy = -(probs * np.log(np.clip(probs, 1e-12, 1.0))).sum(axis=1)

    print(f"\n{label}")
    print("-" * 52)
    print(f"Mean entropy: {entropy.mean():.6f}")
    print(f"Mean maximum probability: {probs.max(axis=1).mean():.6f}")
    for discount in DISCOUNTS:
        print(
            f"{discount:>2}%: "
            f"mean probability={probs[:, DISCOUNTS.tolist().index(discount)].mean():.6f}; "
            f"selected={np.count_nonzero(actions == discount)} "
            f"({np.mean(actions == discount) * 100:.2f}%)"
        )

def main():
    for path in (INITIAL_PATH, TRAINED_PATH):
        if not path.exists():
            raise FileNotFoundError(path)

    dataset = load_ppo_dataset()
    df = pd.read_parquet(ROOT / "data/processed/products_clean.parquet")
    source_indices = np.asarray(dataset.source_row_indices)
    aligned = df.iloc[source_indices]
    positions = np.flatnonzero(aligned["MonthNum"].between(9, 10).to_numpy())
    states = dataset.states[positions]

    initial_actor = load_actor(INITIAL_PATH, "model_state_dict")
    trained_actor = load_actor(TRAINED_PATH, "actor_state_dict")

    initial_probs = get_probabilities(initial_actor, states)
    trained_probs = get_probabilities(trained_actor, states)

    print("=" * 68)
    print("INITIAL VS TRAINED PPO POLICY")
    print("=" * 68)
    print(f"Test products: {len(states)}")

    summarize("INITIAL WARM-START ACTOR", initial_probs)
    summarize("TRAINED PPO ACTOR (ENTROPY 0.05)", trained_probs)

    print("\nChange from initial to trained actor")
    print("-" * 52)
    print(f"Mean absolute probability change: {np.abs(trained_probs - initial_probs).mean():.6f}")
    print(f"Mean L1 distance per product: {np.abs(trained_probs - initial_probs).sum(axis=1).mean():.6f}")
    initial_actions = DISCOUNTS[initial_probs.argmax(axis=1)]
    trained_actions = DISCOUNTS[trained_probs.argmax(axis=1)]
    print(f"Products whose deterministic action changed: {np.count_nonzero(initial_actions != trained_actions)} ({np.mean(initial_actions != trained_actions)*100:.2f}%)")

    output = pd.DataFrame({
        "source_row": source_indices[positions],
        "initial_action": initial_actions,
        "trained_action": trained_actions,
    })
    for i, discount in enumerate(DISCOUNTS):
        output[f"initial_prob_{discount}pct"] = initial_probs[:, i]
        output[f"trained_prob_{discount}pct"] = trained_probs[:, i]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved comparison: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
