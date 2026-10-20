from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path.cwd().resolve()
sys.path.insert(0, str(ROOT))

from src.environment.ppo_dataset import load_ppo_dataset
from src.models.ppo_agent import PPOAgent

CHECKPOINT = ROOT / "models/ppo_transfer/ppo_transfer_supported_8action_1000_updates.pth"
PRODUCTS = ROOT / "data/processed/products_clean.parquet"
OUT = ROOT / "reports/robustness/ppo_transfer_state_sensitivity"
DISCOUNTS = np.array([5, 10, 15, 20, 25, 30, 35, 40])
LEVELS = [0.00, 0.01, 0.05, 0.10]
SEED = 42
BATCH_SIZE = 1024

def main():
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(CHECKPOINT)
    if not PRODUCTS.is_file():
        raise FileNotFoundError(PRODUCTS)
    if OUT.exists():
        raise FileExistsError(
            f"Output directory already exists; refusing to overwrite: {OUT}"
        )

    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    if ckpt.get("state_dim") != 111 or ckpt.get("action_dim") != 8:
        raise ValueError(
            f"Expected 111 states / 8 actions; got "
            f"{ckpt.get('state_dim')} / {ckpt.get('action_dim')}"
        )

    dataset = load_ppo_dataset()
    products = pd.read_parquet(PRODUCTS)
    source_indices = np.asarray(dataset.source_row_indices, dtype=int)
    aligned = products.iloc[source_indices].reset_index(drop=True)
    states_all = np.asarray(dataset.states, dtype=np.float32)

    if states_all.shape != (len(aligned), 111):
        raise ValueError(
            f"State/data alignment mismatch: {states_all.shape}, {len(aligned)}"
        )
    if not np.isfinite(states_all).all():
        raise ValueError("Dataset states contain non-finite values")

    train_mask = aligned["MonthNum"].between(1, 6).to_numpy()
    test_mask = aligned["MonthNum"].between(9, 10).to_numpy()
    train_states = states_all[train_mask]
    states = states_all[test_mask]
    test_source_indices = source_indices[test_mask]

    if not len(train_states) or not len(states):
        raise ValueError("Training or held-out test split is empty")

    feature_std = np.maximum(train_states.astype(np.float64).std(axis=0), 1e-8)
    feature_std = feature_std.astype(np.float32)

    agent = PPOAgent(state_dim=111, action_dim=8)
    agent.actor.load_state_dict(ckpt["actor_state_dict"])
    agent.actor.eval()

    def predict_probs(x):
        batches = []
        with torch.no_grad():
            for start in range(0, len(x), BATCH_SIZE):
                batch = torch.as_tensor(x[start:start + BATCH_SIZE])
                batches.append(torch.softmax(agent.actor(batch), dim=-1).numpy())
        return np.concatenate(batches, axis=0)

    baseline_probs = predict_probs(states)
    baseline_actions = baseline_probs.argmax(axis=1)
    baseline_discounts = DISCOUNTS[baseline_actions]

    rng = np.random.default_rng(SEED)
    # Use the same standard-normal noise realization across noise levels.
    shared_noise = rng.standard_normal(states.shape).astype(np.float32)

    summaries, detail_rows = [], []

    for level in LEVELS:
        perturbed = states + level * feature_std * shared_noise
        probs = predict_probs(perturbed)
        actions = probs.argmax(axis=1)
        discounts = DISCOUNTS[actions]

        flips = actions != baseline_actions
        abs_change = np.abs(discounts - baseline_discounts)
        tv = np.abs(probs - baseline_probs).sum(axis=1) / 2
        entropy = -(probs * np.log(np.clip(probs, 1e-12, 1.0))).sum(axis=1)

        summaries.append({
            "noise_fraction_of_train_std": level,
            "test_samples": len(states),
            "action_flip_rate": float(flips.mean()),
            "mean_absolute_discount_change_pp": float(abs_change.mean()),
            "median_absolute_discount_change_pp": float(np.median(abs_change)),
            "p95_absolute_discount_change_pp": float(np.percentile(abs_change, 95)),
            "mean_total_variation_distance": float(tv.mean()),
            "mean_max_action_probability": float(probs.max(axis=1).mean()),
            "mean_policy_entropy": float(entropy.mean()),
            "baseline_mean_discount": float(baseline_discounts.mean()),
            "perturbed_mean_discount": float(discounts.mean()),
        })

        if level > 0:
            for i in range(len(states)):
                detail_rows.append({
                    "source_row": int(test_source_indices[i]),
                    "noise_fraction_of_train_std": level,
                    "baseline_discount": int(baseline_discounts[i]),
                    "perturbed_discount": int(discounts[i]),
                    "action_flipped": bool(flips[i]),
                    "absolute_discount_change_pp": int(abs_change[i]),
                    "probability_total_variation": float(tv[i]),
                })

        print(
            f"Noise {level:.0%} | flip rate={flips.mean():.4%} | "
            f"mean absolute discount change={abs_change.mean():.4f} pp | "
            f"mean TV distance={tv.mean():.6f}"
        )

    OUT.mkdir(parents=True)
    pd.DataFrame(summaries).to_csv(OUT / "summary.csv", index=False)
    pd.DataFrame(detail_rows).to_csv(OUT / "by_product.csv", index=False)

    metadata = {
        "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
        "checkpoint_updates": ckpt.get("updates"),
        "state_dim": 111,
        "action_dim": 8,
        "discount_actions": DISCOUNTS.tolist(),
        "train_months": [1, 2, 3, 4, 5, 6],
        "test_months": [9, 10],
        "test_samples": int(len(states)),
        "noise_levels": LEVELS,
        "noise_method": (
            "Seeded standard-normal input perturbations scaled by each "
            "feature's standard deviation on training months 1-6; "
            "same noise realization reused across levels."
        ),
        "seed": SEED,
        "interpretation": (
            "Measures policy action and probability sensitivity to synthetic "
            "input perturbations. It does not measure reward retention, "
            "real-world measurement error, or causal business impact."
        ),
    }
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"\nSaved new robustness results to: {OUT.relative_to(ROOT)}")
    print(pd.DataFrame(summaries).to_string(index=False))

if __name__ == "__main__":
    main()
