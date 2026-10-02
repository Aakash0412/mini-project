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
from src.transfer.soft_policy_distillation import (
    SoftPolicyDistillationConfig,
    SupportedTargetPricingPolicy,
)


DEVICE = torch.device("cpu")


def load_source_policy():
    cfg = NeoRLSourcePolicyConfig()
    model = NeoRLSourcePolicy(cfg).to(DEVICE)

    path = (
        PROJECT_ROOT
        / "models/transfer/neorl_source_policy/best_model.pth"
    )
    payload = torch.load(path, map_location=DEVICE, weights_only=False)
    state = payload.get("model_state_dict", payload)
    model.load_state_dict(state)
    model.eval()

    return model


def load_adapter():
    cfg = TransferAdapterConfig()
    model = TransferAdapter(cfg).to(DEVICE)

    path = (
        PROJECT_ROOT
        / "models/transfer/adapter/best_adapter.pth"
    )
    payload = torch.load(path, map_location=DEVICE, weights_only=False)
    state = payload.get("model_state_dict", payload)
    model.load_state_dict(state)
    model.eval()

    return model


def load_data():
    state_dir = PROJECT_ROOT / "data/processed/state"

    states = np.load(
        state_dir / "state.npy"
    ).astype(np.float32)

    row_indices = np.load(
        state_dir / "state_row_indices.npy"
    ).astype(np.int64)

    products = pd.read_parquet(
        PROJECT_ROOT / "data/processed/products_clean.parquet"
    )

    aligned = products.iloc[row_indices]
    months = pd.to_numeric(
        aligned["MonthNum"],
        errors="coerce",
    ).to_numpy()

    train_mask = np.isfinite(months) & (months <= 6)
    val_mask = np.isfinite(months) & (months >= 7) & (months <= 8)

    return states, train_mask, val_mask


@torch.no_grad()
def source_soft_targets(
    states: np.ndarray,
    adapter,
    source_policy,
    batch_size: int,
    samples: int,
):
    """
    Generate soft 8-action targets from the frozen NeoRL policy.

    The source dataset stores the second action normalized by 5:
        normalized_factor = raw_factor / 5

    Therefore:
        raw_factor = normalized_factor * 5
        discount = (1 - raw_factor) * 100

    We Monte-Carlo sample the source Normal policy, clamp to the
    supported NeoRL action range, map samples to 5..40%, and form
    an 8-bin distribution.
    """

    all_targets = []

    for start in range(0, len(states), batch_size):
        x = torch.from_numpy(
            states[start:start + batch_size]
        ).to(DEVICE)

        pseudo_state = adapter(x)

        # The source policy implementation exposes forward().
        output = source_policy.forward(pseudo_state)

        if isinstance(output, tuple):
            mean, log_std = output[:2]
        else:
            raise RuntimeError(
                "Expected NeoRLSourcePolicy.forward() to return "
                "(mean, log_std)."
            )

        std = log_std.exp().clamp(min=1e-5, max=10.0)

        dist = torch.distributions.Normal(mean, std)

        sampled = dist.rsample(
            (samples,)
        )

        # Source action dimension 1 is the coupon/discount factor.
        normalized_factor = sampled[..., 1]

        # NeoRL dataset action is normalized by high[0] = 5.
        normalized_factor = normalized_factor.clamp(
            0.12, 0.20
        )

        raw_factor = normalized_factor * 5.0

        discount = (
            1.0 - raw_factor
        ) * 100.0

        # Map to nearest supported discount: 5..40.
        index = torch.round(
            (discount - 5.0) / 5.0
        ).clamp(0, 7).long()

        one_hot = F.one_hot(
            index,
            num_classes=8,
        ).float()

        target = one_hot.mean(dim=0)

        # Avoid exactly zero probabilities.
        target = (
            target + 1e-4
        )
        target = target / target.sum()

        all_targets.append(
            target.cpu()
        )

    return torch.cat(
        [
            t.unsqueeze(0)
            if t.ndim == 1
            else t
            for t in all_targets
        ],
        dim=0,
    ).numpy()


def train(
    epochs: int,
    batch_size: int,
    samples: int,
    output_dir: Path,
):
    torch.manual_seed(42)
    np.random.seed(42)

    states, train_mask, val_mask = load_data()

    adapter = load_adapter()
    source_policy = load_source_policy()

    train_states = states[train_mask]
    val_states = states[val_mask]

    print("=" * 70)
    print("NEORL -> ITANET SOFT POLICY DISTILLATION")
    print("=" * 70)
    print(f"Train states: {train_states.shape}")
    print(f"Val states:   {val_states.shape}")
    print(
        "Targets preserve the source-policy action distribution "
        "instead of using deterministic hard labels."
    )
    print(
        "Target action space: 5,10,15,20,25,30,35,40%"
    )
    print(f"Monte-Carlo source samples/state: {samples}")

    print("\nGenerating soft source targets...")
    train_targets = source_soft_targets(
        train_states,
        adapter,
        source_policy,
        batch_size,
        samples,
    )
    val_targets = source_soft_targets(
        val_states,
        adapter,
        source_policy,
        batch_size,
        samples,
    )

    print(
        f"Train target mean distribution: "
        f"{train_targets.mean(axis=0)}"
    )
    print(
        f"Val target mean distribution:   "
        f"{val_targets.mean(axis=0)}"
    )

    config = SoftPolicyDistillationConfig()
    model = SupportedTargetPricingPolicy(config).to(DEVICE)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    train_loader = DataLoader(
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

    output_dir.mkdir(parents=True, exist_ok=True)

    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []

        for x, target in train_loader:
            x = x.to(DEVICE)
            target = target.to(DEVICE)

            logits = model(x)

            temperature = config.temperature

            student_log_probs = F.log_softmax(
                logits / temperature,
                dim=-1,
            )

            loss = F.kl_div(
                student_log_probs,
                target,
                reduction="batchmean",
            ) * (temperature ** 2)

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

            val_loss = F.kl_div(
                val_log_probs,
                val_y,
                reduction="batchmean",
            ) * (config.temperature ** 2)

            deterministic = torch.argmax(
                val_logits,
                dim=-1,
            )

            target_argmax = torch.argmax(
                val_y,
                dim=-1,
            )

            val_acc = (
                deterministic == target_argmax
            ).float().mean().item()

        train_loss = float(
            np.mean(train_losses)
        )

        print(
            f"Epoch {epoch:03d} | "
            f"Train KL {train_loss:.6f} | "
            f"Val KL {val_loss.item():.6f} | "
            f"Val Argmax Acc {val_acc * 100:.2f}%"
        )

        history.append(
            {
                "epoch": epoch,
                "train_kl": train_loss,
                "val_kl": float(val_loss.item()),
                "val_argmax_accuracy": val_acc,
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
                },
                output_dir / "best_model.pth",
            )

    report = {
        "epochs": epochs,
        "batch_size": batch_size,
        "source_samples_per_state": samples,
        "train_states": int(len(train_states)),
        "val_states": int(len(val_states)),
        "best_epoch": best_epoch,
        "best_val_kl": best_val,
        "train_target_mean_distribution": train_targets.mean(axis=0).tolist(),
        "val_target_mean_distribution": val_targets.mean(axis=0).tolist(),
        "history": history,
        "action_space": [5, 10, 15, 20, 25, 30, 35, 40],
        "interpretation": (
            "Behavioral soft distillation from the frozen NeoRL source "
            "policy through the existing Amazon 111D->4D adapter. "
            "This is an adaptation experiment and not an exact "
            "implementation of the paper's paired-state transfer equation."
        ),
    }

    report_path = (
        PROJECT_ROOT
        / "reports/transfer/soft_policy_distillation_training.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("=" * 70)
    print("SOFT DISTILLATION COMPLETE")
    print(f"Best validation KL: {best_val:.6f}")
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
        help="Monte-Carlo source-policy samples per state.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(
            PROJECT_ROOT
            / "models/transfer/soft_distillation"
        ),
    )

    args = parser.parse_args()

    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        samples=args.samples,
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
