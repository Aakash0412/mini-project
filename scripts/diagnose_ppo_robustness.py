
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.environment.ppo_dataset import load_ppo_dataset
from src.models.ppo_agent import PPOAgent

CHECKPOINT = ROOT / "models/ppo/ppo_batched_economic_50_updates.pth"
PRODUCTS = ROOT / "data/processed/products_clean.parquet"
OUTPUT_DIR = ROOT / "reports/base_paper_reproduction/state_robustness"

DISCOUNTS = np.array([0, 5, 10, 15, 20, 25, 30, 35, 40])
NOISE_LEVELS = [0.00, 0.01, 0.05, 0.10]
SEED = 42
BATCH_SIZE = 1024


def main():
    if not CHECKPOINT.exists():
        raise FileNotFoundError(CHECKPOINT)

    checkpoint = torch.load(
        CHECKPOINT, map_location="cpu", weights_only=False
    )
    if checkpoint.get("state_dim") != 111 or checkpoint.get("action_dim") != 9:
        raise ValueError("Checkpoint is not the expected 111-input, 9-action PPO.")

    dataset = load_ppo_dataset()
    products = pd.read_parquet(PRODUCTS)
    source_indices = np.asarray(dataset.source_row_indices)
    aligned = products.iloc[source_indices].reset_index(drop=True)

    test_positions = np.flatnonzero(
        aligned["MonthNum"].between(9, 10).to_numpy()
    )
    states = np.asarray(dataset.states[test_positions], dtype=np.float32)

    if states.shape[1] != 111 or not np.isfinite(states).all():
        raise ValueError(f"Unexpected or invalid state array: {states.shape}")

    agent = PPOAgent(state_dim=111, action_dim=9)
    agent.actor.load_state_dict(checkpoint["actor_state_dict"])
    agent.actor.eval()

    # Scale perturbations per feature using training-state standard deviations.
    # This prevents one common absolute noise magnitude from dominating
    # features with small natural scales.
    train_positions = np.flatnonzero(
        aligned["MonthNum"].between(1, 6).to_numpy()
    )
    train_states = np.asarray(dataset.states[train_positions], dtype=np.float64)
    feature_std = train_states.std(axis=0)
    feature_std = np.maximum(feature_std, 1e-8)

    rng = np.random.default_rng(SEED)
    shared_noise = rng.standard_normal(states.shape).astype(np.float32)

    baseline_probs = []
    with torch.no_grad():
        for start in range(0, len(states), BATCH_SIZE):
            batch = torch.as_tensor(states[start:start + BATCH_SIZE])
            baseline_probs.append(torch.softmax(agent.actor(batch), dim=-1).numpy())

    baseline_probs = np.concatenate(baseline_probs)
    baseline_actions = baseline_probs.argmax(axis=1)
    baseline_discounts = DISCOUNTS[baseline_actions]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    product_rows = []

    for level in NOISE_LEVELS:
        perturbed_states = states + (
            level * feature_std.astype(np.float32) * shared_noise
        )

        probability_batches = []
        with torch.no_grad():
            for start in range(0, len(states), BATCH_SIZE):
                batch = torch.as_tensor(
                    perturbed_states[start:start + BATCH_SIZE]
                )
                probability_batches.append(
                    torch.softmax(agent.actor(batch), dim=-1).numpy()
                )

        probs = np.concatenate(probability_batches)
        actions = probs.argmax(axis=1)
        selected_discounts = DISCOUNTS[actions]

        flips = actions != baseline_actions
        abs_discount_change = np.abs(
            selected_discounts - baseline_discounts
        )
        probability_shift = np.abs(probs - baseline_probs).sum(axis=1) / 2

        entropy = -(
            probs * np.log(np.clip(probs, 1e-12, 1.0))
        ).sum(axis=1)

        summary_rows.append({
            "noise_fraction_of_train_std": level,
            "test_products": len(states),
            "action_flip_rate": float(flips.mean()),
            "mean_absolute_discount_change_pp": float(
                abs_discount_change.mean()
            ),
            "median_absolute_discount_change_pp": float(
                np.median(abs_discount_change)
            ),
            "mean_total_variation_distance": float(
                probability_shift.mean()
            ),
            "mean_max_action_probability": float(probs.max(axis=1).mean()),
            "mean_policy_entropy": float(entropy.mean()),
            "baseline_mean_discount": float(baseline_discounts.mean()),
            "perturbed_mean_discount": float(selected_discounts.mean()),
        })

        if level > 0:
            product_rows.extend({
                "source_row": int(source_indices[test_positions[i]]),
                "noise_fraction_of_train_std": level,
                "baseline_discount": int(baseline_discounts[i]),
                "perturbed_discount": int(selected_discounts[i]),
                "action_flipped": bool(flips[i]),
                "absolute_discount_change_pp": int(abs_discount_change[i]),
                "probability_total_variation": float(probability_shift[i]),
            } for i in range(len(states)))

        print(
            f"Noise {level:.0%} | "
            f"flip rate={flips.mean():.4f} | "
            f"mean discount change="
            f"{abs_discount_change.mean():.3f} percentage points | "
            f"mean probability shift={probability_shift.mean():.4f}"
        )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUTPUT_DIR / "state_robustness_summary.csv", index=False)
    pd.DataFrame(product_rows).to_csv(
        OUTPUT_DIR / "state_robustness_by_product.csv", index=False
    )

    metadata = {
        "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
        "checkpoint_updates": checkpoint.get("updates"),
        "state_dim": 111,
        "action_dim": 9,
        "discount_actions": DISCOUNTS.tolist(),
        "test_months": [9, 10],
        "test_products": int(len(states)),
        "noise_levels": NOISE_LEVELS,
        "noise_method": (
            "Shared seeded standard-normal perturbations, scaled by each "
            "feature's training-split standard deviation."
        ),
        "seed": SEED,
        "interpretation_note": (
            "Synthetic input-sensitivity analysis, not an empirically "
            "calibrated measurement-error model. Reward retention is not "
            "estimated by this script."
        ),
    }
    (OUTPUT_DIR / "state_robustness_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("\nSaved reports to:", OUTPUT_DIR)
    print("\nSummary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
