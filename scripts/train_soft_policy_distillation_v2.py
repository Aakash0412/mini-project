from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.transfer.neorl_source_policy import (
    NeoRLSourcePolicy,
    NeoRLSourcePolicyConfig,
)
from src.transfer.transfer_adapter import (
    TransferAdapter,
    TransferAdapterConfig,
)
from src.transfer.soft_policy_distillation_v2 import (
    SoftPolicyDistillationV2Config,
    SupportedTargetPricingPolicyV2,
    continuous_to_soft_bins,
)


DEVICE = torch.device("cpu")


def load_source_policy():
    cfg = NeoRLSourcePolicyConfig()
    model = NeoRLSourcePolicy(cfg).to(DEVICE)

    path = (
        PROJECT_ROOT
        / "models/transfer/neorl_source_policy/best_model.pth"
    )
    payload = torch.load(
        path,
        map_location=DEVICE,
        weights_only=False,
    )
    model.load_state_dict(
        payload.get("model_state_dict", payload)
    )
    model.eval()
    return model


def load_adapter():
    cfg = TransferAdapterConfig()
    model = TransferAdapter(cfg).to(DEVICE)

    path = (
        PROJECT_ROOT
        / "models/transfer/adapter/best_adapter.pth"
    )
    payload = torch.load(
        path,
        map_location=DEVICE,
        weights_only=False,
    )
    model.load_state_dict(
        payload.get("model_state_dict", payload)
    )
    model.eval()
    return model


def load_data():
    state_dir = PROJECT_ROOT / "data/processed/state"

    states = np.load(
        state_dir / "state.npy"
    ).astype(np.float32)

    rows = np.load(
        state_dir / "state_row_indices.npy"
    ).astype(np.int64)

    products = pd.read_parquet(
        PROJECT_ROOT
        / "data/processed/products_clean.parquet"
    )

    months = pd.to_numeric(
        products.iloc[rows]["MonthNum"],
        errors="coerce",
    ).to_numpy()

    train_mask = (
        np.isfinite(months)
        & (months >= 1)
        & (months <= 6)
    )

    val_mask = (
        np.isfinite(months)
        & (months >= 7)
        & (months <= 8)
    )

    return states, train_mask, val_mask


@torch.no_grad()
def generate_soft_targets(
    states: np.ndarray,
    adapter,
    source_policy,
    batch_size: int,
    samples: int,
    bandwidth: float,
):
    """
    Generate one 8-class soft target per Amazon product state.

    Source-policy samples are converted to discount percentages and then
    mapped to nearby supported target discounts using Gaussian RBF
    assignment. The output shape is [N, 8].
    """

    outputs = []

    for start in range(0, len(states), batch_size):
        x = torch.from_numpy(
            states[start:start + batch_size]
        ).to(DEVICE)

        pseudo_state = adapter(x)

        output = source_policy.forward(
            pseudo_state
        )

        if not isinstance(output, tuple):
            raise RuntimeError(
                "NeoRLSourcePolicy.forward() must return "
                "(mean, log_std)."
            )

        mean, log_std = output[:2]

        std = log_std.exp().clamp(
            min=1e-5,
            max=10.0,
        )

        distribution = torch.distributions.Normal(
            mean,
            std,
        )

        # [samples, batch, 2]
        sampled_actions = distribution.sample(
            (samples,)
        )

        # NeoRL source action dimension 1 is the coupon factor.
        normalized_factor = sampled_actions[..., 1]

        # NeoRL dataset normalization uses high[0] = 5.
        raw_factor = (
            normalized_factor * 5.0
        )

        # Convert factor to percentage discount:
        # 0.95 -> 5%, ..., 0.60 -> 40%.
        discount = (
            1.0 - raw_factor
        ) * 100.0

        # Keep within the source-supported target range.
        discount = discount.clamp(
            5.0,
            40.0,
        )

        # Convert every sampled continuous action independently
        # into an 8-way soft distribution.
        sampled_targets = continuous_to_soft_bins(
            discount.reshape(-1),
            bandwidth=bandwidth,
        )

        sampled_targets = sampled_targets.reshape(
            samples,
            -1,
            8,
        )

        # Average only over Monte-Carlo samples.
        target = sampled_targets.mean(dim=0)

        # Numerical safety.
        target = target.clamp_min(1e-8)
        target = target / target.sum(
            dim=-1,
            keepdim=True,
        )

        outputs.append(
            target.cpu()
        )

    return torch.cat(outputs, dim=0).numpy()


def validate_target_distribution(
    targets: np.ndarray,
    name: str,
):
    mean_distribution = targets.mean(axis=0)

    print(
        f"{name} target mean distribution:"
    )

    for i, value in enumerate(mean_distribution):
        print(
            f"  {5 + 5 * i:2d}% : "
            f"{value:.6f}"
        )

    row_sums = targets.sum(axis=1)

    print(
        f"{name} row-sum range: "
        f"{row_sums.min():.6f} - "
        f"{row_sums.max():.6f}"
    )

    entropy = (
        -targets
        * np.log(np.clip(targets, 1e-12, 1.0))
    ).sum(axis=1).mean()

    print(
        f"{name} mean entropy: {entropy:.6f}"
    )

def train(
    epochs: int,
    batch_size: int,
    samples: int,
    bandwidth: float,
):
    torch.manual_seed(42)
    np.random.seed(42)

    states, train_mask, val_mask = load_data()

    train_states = states[train_mask]
    val_states = states[val_mask]

    adapter = load_adapter()
    source_policy = load_source_policy()

    print("=" * 70)
    print("NEORL -> ITANET CORRECTED SOFT POLICY DISTILLATION")
    print("=" * 70)
    print(
        f"Train states: {train_states.shape}"
    )
    print(
        f"Val states:   {val_states.shape}"
    )
    print(
        "Target action space: "
        "5,10,15,20,25,30,35,40%"
    )
    print(
        f"Monte-Carlo samples/state: {samples}"
    )
    print(
        f"Soft-bin bandwidth: {bandwidth:.2f} percentage points"
    )

    print("\nGenerating per-state soft targets...")

    train_targets = generate_soft_targets(
        train_states,
        adapter,
        source_policy,
        batch_size,
        samples,
        bandwidth,
    )

    val_targets = generate_soft_targets(
        val_states,
        adapter,
        source_policy,
        batch_size,
        samples,
        bandwidth,
    )

    print()
    validate_target_distribution(
        train_targets,
        "Train",
    )
    validate_target_distribution(
        val_targets,
        "Val",
    )

    config = SoftPolicyDistillationV2Config()

    model = SupportedTargetPricingPolicyV2(
        config
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(train_states),
            torch.from_numpy(train_targets).float(),
        ),
        batch_size=batch_size,
        shuffle=True,
    )

    val_x = torch.from_numpy(val_states)
    val_y = torch.from_numpy(val_targets).float()

    best_val = float("inf")
    best_epoch = -1
    history = []

    output_dir = (
        PROJECT_ROOT
        / "models/transfer/soft_distillation_v2"
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []

        for x, target in loader:
            x = x.to(DEVICE)
            target = target.to(DEVICE)

            logits = model(x)

            temperature = config.temperature

            log_probs = F.log_softmax(
                logits / temperature,
                dim=-1,
            )

            loss = (
                F.kl_div(
                    log_probs,
                    target,
                    reduction="batchmean",
                )
                * temperature**2
            )

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                0.5,
            )

            optimizer.step()

            train_losses.append(
                float(loss.item())
            )

        model.eval()

        with torch.no_grad():
            val_logits = model(val_x)

            val_log_probs = F.log_softmax(
                val_logits / config.temperature,
                dim=-1,
            )

            val_loss = (
                F.kl_div(
                    val_log_probs,
                    val_y,
                    reduction="batchmean",
                )
                * config.temperature**2
            )

            student_probs = torch.softmax(
                val_logits,
                dim=-1,
            )

            student_argmax = torch.argmax(
                student_probs,
                dim=-1,
            )

            target_argmax = torch.argmax(
                val_y,
                dim=-1,
            )

            argmax_accuracy = (
                student_argmax == target_argmax
            ).float().mean().item()

        train_loss = float(
            np.mean(train_losses)
        )

        print(
            f"Epoch {epoch:03d} | "
            f"Train KL {train_loss:.6f} | "
            f"Val KL {val_loss.item():.6f} | "
            f"Val Argmax Acc {argmax_accuracy * 100:.2f}%"
        )

        history.append(
            {
                "epoch": epoch,
                "train_kl": train_loss,
                "val_kl": float(val_loss.item()),
                "val_argmax_accuracy": argmax_accuracy,
            }
        )

        if val_loss.item() < best_val:
            best_val = float(val_loss.item())
            best_epoch = epoch

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config.__dict__,
                    "best_val_kl": best_val,
                    "epoch": epoch,
                    "action_space": [
                        5, 10, 15, 20,
                        25, 30, 35, 40
                    ],
                    "soft_target_bandwidth": bandwidth,
                },
                output_dir / "best_model.pth",
            )

    report = {
        "epochs": epochs,
        "batch_size": batch_size,
        "source_samples_per_state": samples,
        "soft_target_bandwidth": bandwidth,
        "train_states": int(len(train_states)),
        "val_states": int(len(val_states)),
        "best_epoch": best_epoch,
        "best_val_kl": best_val,
        "train_target_mean_distribution": (
            train_targets.mean(axis=0).tolist()
        ),
        "val_target_mean_distribution": (
            val_targets.mean(axis=0).tolist()
        ),
        "history": history,
        "action_space": [
            5, 10, 15, 20,
            25, 30, 35, 40
        ],
        "interpretation": (
            "Corrected behavioral soft distillation. "
            "Each Amazon state receives its own normalized "
            "8-action target distribution generated from "
            "Monte-Carlo samples of the frozen NeoRL source "
            "policy after the existing 111D->4D adapter. "
            "This remains an adaptation experiment rather "
            "than an exact paired-state implementation."
        ),
    }

    report_path = (
        PROJECT_ROOT
        / "reports/transfer/"
        / "soft_policy_distillation_v2_training.json"
    )
    report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("=" * 70)
    print("CORRECTED SOFT DISTILLATION COMPLETE")
    print(
        f"Best validation KL: {best_val:.6f}"
    )
    print(
        f"Checkpoint: "
        f"{output_dir / 'best_model.pth'}"
    )
    print(
        f"Report: {report_path}"
    )
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--bandwidth",
        type=float,
        default=3.0,
    )

    args = parser.parse_args()

    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        samples=args.samples,
        bandwidth=args.bandwidth,
    )


if __name__ == "__main__":
    main()
