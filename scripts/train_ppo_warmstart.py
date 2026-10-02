from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.environment.ppo_dataset import load_ppo_dataset

from src.transfer.ppo_warmstart import (
    PPOWarmStartConfig,
    load_adapter,
    load_source_policy,
    train_warm_start_actor,
    expand_to_nine_actions,
)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ADAPTER_CHECKPOINT = (
    PROJECT_ROOT
    / "models"
    / "transfer"
    / "adapter"
    / "best_adapter.pth"
)

SOURCE_CHECKPOINT = (
    PROJECT_ROOT
    / "models"
    / "transfer"
    / "neorl_source_policy"
    / "best_model.pth"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "models"
    / "transfer"
    / "ppo_warmstart"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

WARMSTART_CHECKPOINT = (
    OUTPUT_DIR
    / "best_warmstart_actor.pth"
)

PPO_INITIALIZATION_CHECKPOINT = (
    OUTPUT_DIR
    / "ppo_warmstart_initialization.pth"
)

HISTORY_PATH = (
    OUTPUT_DIR
    / "training_history.json"
)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:

    print("=" * 72)
    print("PPO TRANSFER WARM-START")
    print("=" * 72)

    device = torch.device(
        "cpu"
    )

    # -----------------------------------------------------------------------
    # Load PPO dataset
    # -----------------------------------------------------------------------

    print("\n[1] Loading Amazon target states...")

    dataset = load_ppo_dataset()

    states = dataset.states.astype(
        np.float32
    )

    print(
        f"    states = {states.shape}"
    )

    if states.shape[1] != 111:
        raise RuntimeError(
            f"Expected 111-D states, got {states.shape}"
        )

    # -----------------------------------------------------------------------
    # Chronological training data only
    # -----------------------------------------------------------------------

    print(
        "\n[2] Selecting chronological training states..."
    )

    import pandas as pd

    products_df = pd.read_parquet(
        PROJECT_ROOT
        / "data"
        / "processed"
        / "products_clean.parquet"
    )

    aligned = products_df.iloc[
        dataset.source_row_indices
    ].copy()

    train_mask = aligned[
        "MonthNum"
    ].between(
        1,
        6,
    ).to_numpy()

    train_states = states[
        train_mask
    ]

    print(
        f"    training states = "
        f"{train_states.shape}"
    )

    # -----------------------------------------------------------------------
    # Load frozen transfer models
    # -----------------------------------------------------------------------

    print(
        "\n[3] Loading frozen transfer models..."
    )

    adapter = load_adapter(
        ADAPTER_CHECKPOINT,
        device,
    )

    source_policy = load_source_policy(
        SOURCE_CHECKPOINT,
        device,
    )

    print(
        "    adapter: loaded"
    )

    print(
        "    NeoRL source policy: loaded"
    )

    # -----------------------------------------------------------------------
    # Warm-start configuration
    # -----------------------------------------------------------------------

    config = PPOWarmStartConfig(
        state_dim=111,
        action_dim=8,
        hidden_dims=(128, 128),
        learning_rate=8e-4,
        weight_decay=1e-5,
        batch_size=512,
        epochs=10,
        temperature=1.0,
        seed=42,
    )

    print(
        "\n[4] Warm-start configuration"
    )

    print(
        f"    state dimension = "
        f"{config.state_dim}"
    )

    print(
        f"    transfer actions = "
        f"{config.action_dim}"
    )

    print(
        "    target actions = "
        "[5, 10, 15, 20, 25, 30, 35, 40]"
    )

    print(
        f"    batch size = "
        f"{config.batch_size}"
    )

    print(
        f"    epochs = "
        f"{config.epochs}"
    )

    # -----------------------------------------------------------------------
    # Train soft transfer actor
    # -----------------------------------------------------------------------

    print(
        "\n[5] Training target actor from source-policy distribution..."
    )

    warm_actor, history = train_warm_start_actor(
        states=train_states,
        adapter=adapter,
        source_policy=source_policy,
        config=config,
        device=device,
    )

    # -----------------------------------------------------------------------
    # Save warm actor
    # -----------------------------------------------------------------------

    print(
        "\n[6] Saving 8-action warm-start actor..."
    )

    torch.save(
        {
            "model_state_dict": warm_actor.state_dict(),
            "config": {
                "state_dim": config.state_dim,
                "action_dim": config.action_dim,
                "hidden_dims": tuple(
                    config.hidden_dims
                ),
                "learning_rate": config.learning_rate,
                "weight_decay": config.weight_decay,
                "batch_size": config.batch_size,
                "epochs": config.epochs,
                "temperature": config.temperature,
                "seed": config.seed,
            },
            "target_discounts": [
                5,
                10,
                15,
                20,
                25,
                30,
                35,
                40,
            ],
        },
        WARMSTART_CHECKPOINT,
    )

    print(
        f"    saved: {WARMSTART_CHECKPOINT}"
    )

    # -----------------------------------------------------------------------
    # Expand into the project's existing 9-action PPO actor
    # -----------------------------------------------------------------------

    print(
        "\n[7] Expanding to existing 9-action PPO..."
    )

    ppo_agent = expand_to_nine_actions(
        warm_actor,
        device,
    )

    # -----------------------------------------------------------------------
    # Verify initial policy
    # -----------------------------------------------------------------------

    print(
        "\n[8] Verifying initialized PPO actor..."
    )

    verification_states = torch.as_tensor(
        train_states[:5000],
        dtype=torch.float32,
    )

    with torch.no_grad():

        logits = ppo_agent.actor(
            verification_states
        )

        probabilities = torch.softmax(
            logits,
            dim=1,
        )

        selected = probabilities.argmax(
            dim=1
        ).cpu().numpy()

        entropy = -(
            probabilities
            * torch.log(
                probabilities + 1e-12
            )
        ).sum(
            dim=1
        )

    action_counts = np.bincount(
        selected,
        minlength=9,
    )

    discounts = [
        0,
        5,
        10,
        15,
        20,
        25,
        30,
        35,
        40,
    ]

    print(
        "\n    Deterministic action distribution:"
    )

    for index, discount in enumerate(
        discounts
    ):

        count = int(
            action_counts[index]
        )

        percentage = (
            100.0
            * count
            / len(selected)
        )

        print(
            f"      {discount:2d}% : "
            f"{count:5d} "
            f"({percentage:6.2f}%)"
        )

    print(
        "\n    Mean policy entropy = "
        f"{entropy.mean().item():.6f}"
    )

    # -----------------------------------------------------------------------
    # Save complete PPO initialization
    # -----------------------------------------------------------------------

    torch.save(
        {
            "actor_state_dict":
                ppo_agent.actor.state_dict(),

            "critic_state_dict":
                ppo_agent.critic.state_dict(),

            "state_dim": 111,

            "action_dim": 9,

            "discount_bins": discounts,

            "transfer_source":
                "NeoRL -> TransferAdapter -> "
                "Gaussian-to-categorical soft distillation",

            "transfer_action_bins": [
                5,
                10,
                15,
                20,
                25,
                30,
                35,
                40,
            ],

            "zero_percent_status":
                "initialized separately; "
                "out-of-training-support",

            "config": {
                "actor_lr": 8e-4,
                "critic_lr": 8e-4,
                "seed": 42,
            },
        },
        PPO_INITIALIZATION_CHECKPOINT,
    )

    print(
        f"\n    saved PPO initialization: "
        f"{PPO_INITIALIZATION_CHECKPOINT}"
    )

    # -----------------------------------------------------------------------
    # Save history
    # -----------------------------------------------------------------------

    with open(
        HISTORY_PATH,
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            history,
            handle,
            indent=2,
        )

    print(
        f"    saved history: {HISTORY_PATH}"
    )

    # -----------------------------------------------------------------------
    # Final validation
    # -----------------------------------------------------------------------

    assert logits.shape[1] == 9

    assert torch.all(
        torch.isfinite(logits)
    )

    assert torch.all(
        torch.isfinite(probabilities)
    )

    assert torch.allclose(
        probabilities.sum(dim=1),
        torch.ones(
            len(probabilities)
        ),
        atol=1e-5,
    )

    print(
        "\n[9] Validation"
    )

    print(
        "    state dimension: PASS"
    )

    print(
        "    PPO action dimension: PASS"
    )

    print(
        "    finite logits: PASS"
    )

    print(
        "    finite probabilities: PASS"
    )

    print(
        "    probabilities sum to 1: PASS"
    )

    print(
        "\n" + "=" * 72
    )

    print(
        "PPO TRANSFER WARM-START COMPLETE"
    )

    print(
        "=" * 72
    )


if __name__ == "__main__":
    main()