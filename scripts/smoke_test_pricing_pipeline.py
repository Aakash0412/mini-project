"""Read-only smoke test for the dynamic-pricing inference pipeline."""

from pathlib import Path
import json
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT))

from src.environment.counterfactual_simulator import CounterfactualSimulator
from src.models.ppo_agent import PPOAgent


STATE_DIM = 111
ACTION_DIM = 8
SUPPORTED_DISCOUNTS = [5, 10, 15, 20, 25, 30, 35, 40]
PAPER_DISCOUNTS = [0, 5, 10, 15, 20, 25, 30, 35, 40]

STATE_PATH = ROOT / "data" / "processed" / "state" / "state.npy"
INDICES_PATH = (
    ROOT / "data" / "processed" / "state" / "state_row_indices.npy"
)
PPO_PATH = (
    ROOT / "models" / "ppo_transfer"
    / "ppo_transfer_supported_8action_1000_updates.pth"
)
PREDICTOR_DIR = ROOT / "models" / "sales_predictor"


def check(condition, message):
    if not condition:
        raise RuntimeError(message)
    print(f"[PASS] {message}")


def main():
    print("=" * 68)
    print("DYNAMIC PRICING — LIVE INFERENCE SMOKE TEST")
    print("=" * 68)
    print("This test does not train or modify model checkpoints.\n")

    # 1. Confirm required files exist.
    for path in [STATE_PATH, INDICES_PATH, PPO_PATH]:
        check(path.is_file(), f"Found {path.relative_to(ROOT)}")

    # 2. Load one valid state from the existing final state array.
    states = np.load(STATE_PATH, mmap_mode="r", allow_pickle=False)
    source_indices = np.load(
        INDICES_PATH, mmap_mode="r", allow_pickle=False
    )

    check(states.ndim == 2, "State array is two-dimensional")
    check(states.shape[1] == STATE_DIM, "State dimension is 111")
    check(
        len(states) == len(source_indices),
        "State rows and source-index mapping have matching lengths",
    )

    # Find a finite row. The saved state is expected to be entirely finite.
    finite_rows = np.isfinite(states).all(axis=1)
    valid_positions = np.flatnonzero(finite_rows)
    check(len(valid_positions) > 0, "At least one finite product state exists")

    position = int(valid_positions[0])
    state = np.asarray(states[position], dtype=np.float32).copy()
    source_row = int(source_indices[position])

    check(
        state.shape == (STATE_DIM,) and np.isfinite(state).all(),
        "Selected product state has 111 finite features",
    )
    print(f"Selected state position: {position}")
    print(f"Mapped source row index: {source_row}")

    # 3. Load the frozen sales predictor through the simulator.
    simulator = CounterfactualSimulator(
        checkpoint_dir=str(PREDICTOR_DIR),
        device=torch.device("cpu"),
    )

    check(
        simulator.config.get("input_dim") == STATE_DIM + 1,
        "Frozen predictor expects 112 inputs (111 state features + discount)",
    )
    check(
        all(not p.requires_grad for p in simulator._model.parameters()),
        "Predictor parameters are frozen",
    )

    # 4. Evaluate candidate discounts.
    predictions = simulator.predict_all_discounts(state)

    check(
        set(predictions.keys()) == set(PAPER_DISCOUNTS),
        "Simulator returned predictions for all nine paper discount bins",
    )
    check(
        all(np.isfinite(v) for v in predictions.values()),
        "All candidate-discount predictions are finite",
    )

    print("\nPredicted sales growth by discount:")
    for discount in PAPER_DISCOUNTS:
        print(f"  {discount:>2}% discount -> {predictions[discount]:.6f}")

    # 5. Load the saved 8-action PPO policy.
    checkpoint = torch.load(
        PPO_PATH,
        map_location="cpu",
        weights_only=False,
    )

    saved_action_dim = checkpoint.get("action_dim")
    if saved_action_dim is not None:
        check(
            int(saved_action_dim) == ACTION_DIM,
            f"Checkpoint action dimension is {ACTION_DIM}",
        )

    agent = PPOAgent(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        actor_lr=8e-4,
        critic_lr=8e-4,
    )
    agent.actor.load_state_dict(checkpoint["actor_state_dict"])
    agent.critic.load_state_dict(checkpoint["critic_state_dict"])
    agent.actor.eval()
    agent.critic.eval()

    # 6. Select the deterministic action: highest actor probability.
    state_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        distribution = agent.actor.get_distribution(state_tensor)
        probabilities = distribution.probs.squeeze(0).cpu().numpy()
        action = int(np.argmax(probabilities))
        value = float(agent.critic(state_tensor).item())

    check(
        probabilities.shape == (ACTION_DIM,),
        "PPO actor returned eight action probabilities",
    )
    check(
        np.isfinite(probabilities).all()
        and np.isfinite(value),
        "PPO probabilities and critic value are finite",
    )
    check(
        np.all(probabilities >= 0)
        and np.isclose(probabilities.sum(), 1.0, atol=1e-5),
        "PPO probabilities are valid and sum to one",
    )
    check(
        0 <= action < ACTION_DIM,
        "PPO action index is within the supported action range",
    )

    recommended_discount = SUPPORTED_DISCOUNTS[action]

    check(
        recommended_discount in SUPPORTED_DISCOUNTS,
        "Recommended discount belongs to the supported 5%-40% bins",
    )

    # 7. Report the inference result.
    print("\n" + "=" * 68)
    print("SMOKE-TEST RESULT")
    print("=" * 68)
    print(f"Source row index: {source_row}")
    print(f"PPO action index: {action}")
    print(f"Recommended discount: {recommended_discount}%")
    print(f"Critic value estimate: {value:.6f}")
    print(f"Predicted growth at recommended discount: "
          f"{predictions[recommended_discount]:.6f}")
    print("\nAction probabilities:")
    for discount, probability in zip(
        SUPPORTED_DISCOUNTS, probabilities
    ):
        print(f"  {discount:>2}%: {probability:.6f}")

    result = {
        "status": "PASS",
        "state_position": position,
        "source_row_index": source_row,
        "state_dim": STATE_DIM,
        "predictor_input_dim": simulator.config.get("input_dim"),
        "ppo_action_dim": ACTION_DIM,
        "recommended_discount_pct": recommended_discount,
        "predicted_growth_at_recommended_discount":
            predictions[recommended_discount],
        "critic_value": value,
        "all_predictions_finite": True,
    }

    print("\n" + json.dumps(result, indent=2))
    print("\n[PASS] Live inference smoke test completed.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\n[FAIL] Smoke test failed: {exc}")
        raise