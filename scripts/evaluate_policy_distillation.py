from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.transfer.policy_distillation import (
    PolicyDistillationConfig,
    TargetPricingPolicy,
    discount_from_index,
)


DEVICE = torch.device("cpu")


def load_states():
    state_dir = PROJECT_ROOT / "data/processed/state"

    states = np.load(state_dir / "state.npy").astype(np.float32)
    rows = np.load(state_dir / "state_row_indices.npy").astype(np.int64)

    return states, rows


def load_products():
    return pd.read_parquet(
        PROJECT_ROOT / "data/processed/products_clean.parquet"
    )


def load_policy():
    checkpoint = (
        PROJECT_ROOT
        / "models/transfer/distillation/best_model.pth"
    )

    payload = torch.load(
        checkpoint,
        map_location=DEVICE,
        weights_only=False,
    )

    config_data = payload.get("config", {})

    config = PolicyDistillationConfig(
        state_dim=int(config_data.get("state_dim", 111)),
        action_dim=int(config_data.get("action_dim", 9)),
        hidden_dims=tuple(
            config_data.get(
                "hidden_dims",
                (256, 128, 64),
            )
        ),
        temperature=float(
            config_data.get("temperature", 1.0)
        ),
        seed=int(config_data.get("seed", 42)),
    )

    model = TargetPricingPolicy(config).to(DEVICE)

    state_dict = payload["model_state_dict"]
    model.load_state_dict(state_dict)

    model.eval()

    return model


def split_by_month(products, rows):
    aligned = products.iloc[rows]

    months = pd.to_numeric(
        aligned["MonthNum"],
        errors="coerce",
    ).to_numpy()

    valid = np.isfinite(months)

    train_mask = (
        valid
        & (months >= 1)
        & (months <= 6)
    )

    val_mask = (
        valid
        & (months >= 7)
        & (months <= 8)
    )

    test_mask = (
        valid
        & (months >= 9)
        & (months <= 10)
    )

    return train_mask, val_mask, test_mask


@torch.no_grad()
def evaluate_split(model, states, mask, name):
    x = torch.from_numpy(states[mask]).to(DEVICE)

    logits = model(x)

    probabilities = torch.softmax(
        logits,
        dim=-1,
    )

    actions = torch.argmax(
        probabilities,
        dim=-1,
    )

    discounts = discount_from_index(actions)

    probs = probabilities.cpu().numpy()
    actions_np = actions.cpu().numpy()
    discounts_np = discounts.cpu().numpy()

    mean_probs = probs.mean(axis=0)

    entropy = -np.sum(
        probs * np.log(
            np.clip(probs, 1e-12, 1.0)
        ),
        axis=1,
    )

    print()
    print("=" * 70)
    print(f"{name.upper()} DISTILLATION POLICY")
    print("=" * 70)

    print(f"Samples: {len(x)}")

    print("\nDeterministic discount distribution:")

    for discount in range(0, 45, 5):
        count = int(
            np.sum(discounts_np == discount)
        )

        percentage = (
            100.0 * count / len(discounts_np)
            if len(discounts_np)
            else 0.0
        )

        print(
            f"  {discount:2d}% : "
            f"{count:6d} "
            f"({percentage:6.2f}%)"
        )

    print("\nMean action probabilities:")

    for i, probability in enumerate(mean_probs):
        print(
            f"  {i * 5:2d}% : "
            f"{probability:.6f}"
        )

    print("\nPolicy statistics:")
    print(
        f"  Mean discount: "
        f"{discounts_np.mean():.4f}%"
    )
    print(
        f"  Std discount:  "
        f"{discounts_np.std():.4f}%"
    )
    print(
        f"  Min discount:  "
        f"{discounts_np.min():.2f}%"
    )
    print(
        f"  Max discount:  "
        f"{discounts_np.max():.2f}%"
    )

    print(
        f"  Mean entropy:   "
        f"{entropy.mean():.6f}"
    )

    unique_actions = np.unique(actions_np)

    print(
        f"  Actions used:   "
        f"{len(unique_actions)}/9"
    )

    print(
        "  Action levels:  "
        + ", ".join(
            f"{int(a) * 5}%"
            for a in unique_actions
        )
    )

    return {
        "samples": int(len(x)),
        "discount_distribution": {
            str(d): int(np.sum(discounts_np == d))
            for d in range(0, 45, 5)
        },
        "discount_distribution_percent": {
            str(d): float(
                100.0
                * np.sum(discounts_np == d)
                / len(discounts_np)
            )
            for d in range(0, 45, 5)
        },
        "mean_action_probabilities": {
            str(i * 5): float(mean_probs[i])
            for i in range(9)
        },
        "mean_discount": float(discounts_np.mean()),
        "std_discount": float(discounts_np.std()),
        "min_discount": float(discounts_np.min()),
        "max_discount": float(discounts_np.max()),
        "mean_entropy": float(entropy.mean()),
        "actions_used": int(len(unique_actions)),
        "action_levels_used": [
            int(a) * 5
            for a in unique_actions
        ],
    }


def state_sensitivity(model, states):
    """
    Measure whether the policy changes its action probabilities
    across different product states.
    """

    rng = np.random.default_rng(42)

    n = min(2000, len(states))

    indices = rng.choice(
        len(states),
        size=n,
        replace=False,
    )

    x = torch.from_numpy(
        states[indices]
    ).to(DEVICE)

    with torch.no_grad():
        probabilities = (
            model.probabilities(x)
            .cpu()
            .numpy()
        )

    std_per_action = probabilities.std(axis=0)

    vector_distances = np.linalg.norm(
        probabilities[1:] - probabilities[:-1],
        axis=1,
    )

    mean_vector_distance = float(
        vector_distances.mean()
    )

    print()
    print("=" * 70)
    print("STATE SENSITIVITY")
    print("=" * 70)

    print(
        f"Samples analysed: {n}"
    )

    print(
        "\nProbability standard deviation by action:"
    )

    for i, value in enumerate(std_per_action):
        print(
            f"  {i * 5:2d}% : "
            f"{value:.6f}"
        )

    print(
        f"\nMean probability-vector distance: "
        f"{mean_vector_distance:.6f}"
    )

    return {
        "samples": n,
        "probability_std_by_action": {
            str(i * 5): float(std_per_action[i])
            for i in range(9)
        },
        "mean_probability_vector_distance": (
            mean_vector_distance
        ),
    }


def main():
    print("=" * 70)
    print("NEORL -> ITANET POLICY DISTILLATION EVALUATION")
    print("=" * 70)

    states, rows = load_states()
    products = load_products()
    model = load_policy()

    print(
        f"\nState matrix: {states.shape}"
    )

    train_mask, val_mask, test_mask = split_by_month(
        products,
        rows,
    )

    results = {}

    results["train"] = evaluate_split(
        model,
        states,
        train_mask,
        "Train",
    )

    results["validation"] = evaluate_split(
        model,
        states,
        val_mask,
        "Validation",
    )

    results["test"] = evaluate_split(
        model,
        states,
        test_mask,
        "Test",
    )

    results["state_sensitivity"] = state_sensitivity(
        model,
        states,
    )

    results["checkpoint"] = str(
        PROJECT_ROOT
        / "models/transfer/distillation/best_model.pth"
    )

    results["interpretation"] = (
        "This evaluates behavioral distillation from the "
        "NeoRL source policy through the existing 111D->4D "
        "transfer adapter. It does not establish real-world "
        "pricing improvement or exact reproduction of the "
        "paper's paired-state transfer equation."
    )

    output_dir = (
        PROJECT_ROOT
        / "reports/transfer"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / "policy_distillation_diagnostic.json"
    )

    output_path.write_text(
        json.dumps(
            results,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    print(
        f"Diagnostic: {output_path}"
    )


if __name__ == "__main__":
    main()