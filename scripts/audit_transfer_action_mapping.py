from __future__ import annotations

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Project import path
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import numpy as np
import torch

from src.transfer.transfer_adapter import (
    TransferAdapter,
    TransferAdapterConfig,
)

from src.transfer.neorl_source_policy import (
    NeoRLSourcePolicy,
    NeoRLSourcePolicyConfig,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEVICE = torch.device("cpu")

ADAPTER_CKPT = (
    PROJECT_ROOT
    / "models"
    / "transfer"
    / "adapter"
    / "best_adapter.pth"
)

SOURCE_CKPT = (
    PROJECT_ROOT
    / "models"
    / "transfer"
    / "neorl_source_policy"
    / "best_model.pth"
)

STATE_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "state"
    / "state_test.npy"
)


# Supported Amazon discount levels.
#
# 0% exists in the paper's action space but is outside the observed
# Amazon training support, so this transfer audit uses the supported
# 5%-40% action space.
TARGET_DISCOUNTS = np.array(
    [5, 10, 15, 20, 25, 30, 35, 40],
    dtype=np.float32,
)


# NeoRL stores the second continuous action in normalized form.
#
# From the NeoRL source environment:
#
#   discount factor:
#       0.95 -> 5%
#       0.90 -> 10%
#       0.85 -> 15%
#       0.80 -> 20%
#       0.75 -> 25%
#       0.70 -> 30%
#       0.65 -> 35%
#       0.60 -> 40%
#
# The dataset action was normalized by dividing by the action-space
# high value of 5.
#
# Therefore the corresponding normalized values are:
#
#   0.19 -> 5%
#   0.18 -> 10%
#   ...
#   0.12 -> 40%
#
SOURCE_LEVELS = np.array(
    [0.19, 0.18, 0.17, 0.16, 0.15, 0.14, 0.13, 0.12],
    dtype=np.float32,
)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def continuous_discount(normalized_action: np.ndarray) -> np.ndarray:
    """
    Convert NeoRL's normalized discount-factor action into percentage
    discount.

    NeoRL action dimension 1 represents the discount factor.

    factor = 0.95 -> 5% discount
    factor = 0.90 -> 10% discount
    ...
    factor = 0.60 -> 40% discount

    Dataset normalization divides the action by 5.

    Therefore:

        factor = normalized_action * 5

        discount = (1 - factor) * 100

    This function intentionally does not clip the input. The audit should
    reveal whether the stochastic policy generates values outside the
    supported source range.
    """

    normalized_action = np.asarray(
        normalized_action,
        dtype=np.float32,
    )

    return (1.0 - normalized_action * 5.0) * 100.0


def nearest_target_discount(
    normalized_action: np.ndarray,
) -> np.ndarray:
    """
    Map a normalized NeoRL action to the nearest supported Amazon
    discount level.

    Supported target actions:

        5, 10, 15, 20, 25, 30, 35, 40%

    The source action is clipped to the observed NeoRL discount-action
    range [0.12, 0.19] before nearest-neighbour mapping.

    Clipping is used only for categorical mapping. The raw continuous
    distribution is reported separately by the audit.
    """

    normalized_action = np.asarray(
        normalized_action,
        dtype=np.float32,
    )

    clipped = np.clip(
        normalized_action,
        SOURCE_LEVELS.min(),
        SOURCE_LEVELS.max(),
    )

    distances = np.abs(
        clipped[..., None] - SOURCE_LEVELS[None, :],
    )

    indices = np.argmin(
        distances,
        axis=-1,
    )

    return TARGET_DISCOUNTS[indices]


def print_distribution(
    discounts: np.ndarray,
    title: str,
) -> None:
    """
    Print the distribution over the supported target discounts.
    """

    print(f"\n{title}")

    total = len(discounts)

    for discount in TARGET_DISCOUNTS:
        count = int(np.sum(discounts == discount))

        percentage = (
            100.0 * count / total
            if total > 0
            else 0.0
        )

        print(
            f"    {int(discount):2d}% : "
            f"{count:6d} "
            f"({percentage:6.2f}%)"
        )


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_adapter() -> TransferAdapter:
    """
    Load the previously trained 111-D ItaNet -> 4-D NeoRL adapter.

    The adapter checkpoint uses an older configuration serialization
    format. Its saved config contains:

        target_state_dim
        source_state_dim
        source_min
        source_max
        behavior_weight
        source_prior_weight
        seed

    It does not explicitly store hidden_dims/dropout.

    The actual checkpoint architecture is:

        111 -> 128 -> 64 -> 4

    This is verified from the saved weight shapes:

        network.0.weight = [128, 111]
        network.2.weight = [64, 128]
        network.4.weight = [4, 64]

    Therefore we reconstruct the architecture explicitly as
    hidden_dims=(128, 64), dropout=0.0.
    """

    if not ADAPTER_CKPT.exists():
        raise FileNotFoundError(
            f"Adapter checkpoint not found:\n{ADAPTER_CKPT}"
        )

    checkpoint = torch.load(
        ADAPTER_CKPT,
        map_location=DEVICE,
        weights_only=False,
    )

    config_dict = checkpoint["config"]
    state_dict = checkpoint["model_state_dict"]

    config = TransferAdapterConfig(
        target_state_dim=int(
            config_dict["target_state_dim"]
        ),
        source_state_dim=int(
            config_dict["source_state_dim"]
        ),
        hidden_dims=(128, 64),
        dropout=0.0,
    )

    # Use the exact source-state bounds stored in the trained checkpoint.
    source_min = state_dict["source_min"]
    source_max = state_dict["source_max"]

    model = TransferAdapter(
        config=config,
        source_min=source_min,
        source_max=source_max,
    ).to(DEVICE)

    model.load_state_dict(state_dict)

    model.eval()

    return model


def load_source_policy() -> NeoRLSourcePolicy:
    """
    Load the previously trained NeoRL source policy.

    The checkpoint configuration uses:

        learning_rate

    rather than:

        lr

    so we reconstruct the current NeoRLSourcePolicyConfig using the
    exact keys saved by the checkpoint.
    """

    if not SOURCE_CKPT.exists():
        raise FileNotFoundError(
            f"NeoRL source-policy checkpoint not found:\n{SOURCE_CKPT}"
        )

    checkpoint = torch.load(
        SOURCE_CKPT,
        map_location=DEVICE,
        weights_only=False,
    )

    config_dict = checkpoint["config"]

    config = NeoRLSourcePolicyConfig(
        state_dim=int(
            config_dict["state_dim"]
        ),
        action_dim=int(
            config_dict["action_dim"]
        ),
        hidden_dims=tuple(
            config_dict["hidden_dims"]
        ),
        learning_rate=float(
            config_dict["learning_rate"]
        ),
        weight_decay=float(
            config_dict["weight_decay"]
        ),
        seed=int(
            config_dict["seed"]
        ),
    )

    model = NeoRLSourcePolicy(
        config
    ).to(DEVICE)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model


# ---------------------------------------------------------------------------
# Main audit
# ---------------------------------------------------------------------------

def main() -> None:

    print("=" * 72)
    print("TRANSFER ACTION-MAPPING AUDIT")
    print("=" * 72)

    # -----------------------------------------------------------------------
    # 1. Load adapter
    # -----------------------------------------------------------------------

    print("\n[1] Loading adapter...")

    adapter = load_adapter()

    print(
        f"    target_state_dim = "
        f"{adapter.config.target_state_dim}"
    )

    print(
        f"    source_state_dim = "
        f"{adapter.config.source_state_dim}"
    )

    print(
        "    source_min = "
        f"{adapter.source_min.cpu().numpy()}"
    )

    print(
        "    source_max = "
        f"{adapter.source_max.cpu().numpy()}"
    )

    # -----------------------------------------------------------------------
    # 2. Load source policy
    # -----------------------------------------------------------------------

    print("\n[2] Loading NeoRL source policy...")

    source_policy = load_source_policy()

    print(
        f"    state_dim = "
        f"{source_policy.config.state_dim}"
    )

    print(
        f"    action_dim = "
        f"{source_policy.config.action_dim}"
    )

    # -----------------------------------------------------------------------
    # 3. Load Amazon test states
    # -----------------------------------------------------------------------

    print("\n[3] Loading target states...")

    if not STATE_PATH.exists():
        raise FileNotFoundError(
            f"Target state file not found:\n{STATE_PATH}"
        )

    states = np.load(
        STATE_PATH
    ).astype(np.float32)

    # Keep the audit lightweight.
    n = min(
        len(states),
        5000,
    )

    states = states[:n]

    print(
        f"    states used = {len(states)}"
    )

    print(
        f"    state shape = {states.shape}"
    )

    # -----------------------------------------------------------------------
    # 4. ItaNet state -> NeoRL pseudo-state
    # -----------------------------------------------------------------------

    print(
        "\n[4] Adapter -> NeoRL pseudo-state..."
    )

    state_tensor = torch.from_numpy(
        states
    ).to(DEVICE)

    with torch.no_grad():
        pseudo_states = adapter.predict_source_state(
            state_tensor
        )

    pseudo_np = (
        pseudo_states
        .cpu()
        .numpy()
    )

    print(
        f"    pseudo shape = {pseudo_np.shape}"
    )

    print(
        "    finite = "
        f"{bool(np.all(np.isfinite(pseudo_np)))}"
    )

    print(
        "    pseudo-state means = "
        f"{pseudo_np.mean(axis=0)}"
    )

    print(
        "    pseudo-state std = "
        f"{pseudo_np.std(axis=0)}"
    )

    print(
        "    pseudo-state min = "
        f"{pseudo_np.min(axis=0)}"
    )

    print(
        "    pseudo-state max = "
        f"{pseudo_np.max(axis=0)}"
    )

    # -----------------------------------------------------------------------
    # 5. Source policy deterministic action
    # -----------------------------------------------------------------------

    print(
        "\n[5] Source-policy deterministic action..."
    )

    with torch.no_grad():

        # NeoRLSourcePolicy.forward() returns:
        #
        #   mean_action
        #   std_action
        #   log_std
        #
        mean_action, std_action, log_std = (
            source_policy.forward(
                pseudo_states
            )
        )

    mean_np = (
        mean_action
        .cpu()
        .numpy()
    )

    std_np = (
        std_action
        .cpu()
        .numpy()
    )

    log_std_np = (
        log_std
        .cpu()
        .numpy()
    )

    print(
        "    mean action-0: "
        f"mean={mean_np[:, 0].mean():.6f}, "
        f"std={mean_np[:, 0].std():.6f}"
    )

    print(
        "    mean action-1: "
        f"mean={mean_np[:, 1].mean():.6f}, "
        f"std={mean_np[:, 1].std():.6f}"
    )

    print(
        "    std action-0: "
        f"mean={std_np[:, 0].mean():.6f}, "
        f"std={std_np[:, 0].std():.6f}"
    )

    print(
        "    std action-1: "
        f"mean={std_np[:, 1].mean():.6f}, "
        f"std={std_np[:, 1].std():.6f}"
    )

    print(
        "    log-std action-0: "
        f"mean={log_std_np[:, 0].mean():.6f}"
    )

    print(
        "    log-std action-1: "
        f"mean={log_std_np[:, 1].mean():.6f}"
    )

    # -----------------------------------------------------------------------
    # 6. Deterministic continuous discount
    # -----------------------------------------------------------------------

    print(
        "\n[6] Deterministic continuous discount"
    )

    # The second source action dimension is the discount-factor action.
    normalized_mean = mean_np[:, 1]

    continuous = continuous_discount(
        normalized_mean
    )

    print(
        f"    mean = {continuous.mean():.4f}%"
    )

    print(
        f"    std  = {continuous.std():.4f}%"
    )

    print(
        f"    min  = {continuous.min():.4f}%"
    )

    print(
        f"    max  = {continuous.max():.4f}%"
    )

    print(
        f"    p01  = {np.percentile(continuous, 1):.4f}%"
    )

    print(
        f"    p50  = {np.percentile(continuous, 50):.4f}%"
    )

    print(
        f"    p99  = {np.percentile(continuous, 99):.4f}%"
    )

    # -----------------------------------------------------------------------
    # 7. Deterministic categorical mapping
    # -----------------------------------------------------------------------

    print(
        "\n[7] Deterministic mapped target discount"
    )

    mapped = nearest_target_discount(
        normalized_mean
    )

    print_distribution(
        mapped,
        "    Distribution:"
    )

    # -----------------------------------------------------------------------
    # 8. Stochastic source-policy audit
    # -----------------------------------------------------------------------

    print(
        "\n[8] Stochastic source-policy audit"
    )

    repeats = 20

    pseudo_repeat = (
        pseudo_states
        .unsqueeze(1)
        .repeat(
            1,
            repeats,
            1,
        )
        .reshape(
            -1,
            pseudo_states.shape[1],
        )
    )

    with torch.no_grad():

        sample_mean, sample_std, _ = (
            source_policy.forward(
                pseudo_repeat
            )
        )

        distribution = torch.distributions.Normal(
            sample_mean,
            sample_std,
        )

        samples = distribution.sample()

    sample_action1 = (
        samples[:, 1]
        .cpu()
        .numpy()
    )

    sample_continuous = continuous_discount(
        sample_action1
    )

    sample_mapped = nearest_target_discount(
        sample_action1
    )

    print(
        f"    samples = {len(sample_action1)}"
    )

    print(
        "    raw normalized action-1:"
    )

    print(
        f"        mean = {sample_action1.mean():.6f}"
    )

    print(
        f"        std  = {sample_action1.std():.6f}"
    )

    print(
        f"        min  = {sample_action1.min():.6f}"
    )

    print(
        f"        max  = {sample_action1.max():.6f}"
    )

    print(
        "\n    Continuous discount:"
    )

    print(
        f"        mean = "
        f"{sample_continuous.mean():.4f}%"
    )

    print(
        f"        std  = "
        f"{sample_continuous.std():.4f}%"
    )

    print(
        f"        min  = "
        f"{sample_continuous.min():.4f}%"
    )

    print(
        f"        max  = "
        f"{sample_continuous.max():.4f}%"
    )

    print_distribution(
        sample_mapped,
        "    Stochastic mapped distribution:"
    )

    # -----------------------------------------------------------------------
    # 9. Mapping sanity checks
    # -----------------------------------------------------------------------

    print(
        "\n[9] Mapping sanity checks"
    )

    # Adapter output dimensions.
    assert (
        pseudo_np.ndim == 2
    )

    assert (
        pseudo_np.shape[1] == 4
    )

    # No invalid numerical values.
    assert np.all(
        np.isfinite(pseudo_np)
    )

    assert np.all(
        np.isfinite(mean_np)
    )

    assert np.all(
        np.isfinite(std_np)
    )

    # Adapter output should stay inside its trained source range.
    source_min = (
        adapter.source_min
        .cpu()
        .numpy()
    )

    source_max = (
        adapter.source_max
        .cpu()
        .numpy()
    )

    assert np.all(
        pseudo_np >= source_min - 1e-5
    )

    assert np.all(
        pseudo_np <= source_max + 1e-5
    )

    # Deterministic target mapping must use only supported actions.
    assert np.all(
        mapped >= TARGET_DISCOUNTS.min()
    )

    assert np.all(
        mapped <= TARGET_DISCOUNTS.max()
    )

    assert np.all(
        np.isin(
            mapped,
            TARGET_DISCOUNTS,
        )
    )

    # Stochastic mapping must also produce only supported actions.
    assert np.all(
        np.isin(
            sample_mapped,
            TARGET_DISCOUNTS,
        )
    )

    print(
        "    pseudo-state dimension: PASS"
    )

    print(
        "    finite pseudo-state: PASS"
    )

    print(
        "    pseudo-state range: PASS"
    )

    print(
        "    finite source-policy output: PASS"
    )

    print(
        "    valid deterministic target discounts: PASS"
    )

    print(
        "    valid stochastic target discounts: PASS"
    )

    # -----------------------------------------------------------------------
    # Final
    # -----------------------------------------------------------------------

    print(
        "\n" + "=" * 72
    )

    print(
        "AUDIT COMPLETE"
    )

    print(
        "=" * 72
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()