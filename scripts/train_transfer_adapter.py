"""Train the ItaNet -> NeoRL pseudo-state transfer adapter.

Important methodological note:
The paper's Eq. (21) assumes matched target/source state pairs. The available
Amazon and NeoRL datasets do not contain genuine paired trajectories. We do
not fabricate such pairs.

This implementation therefore performs a local transfer adaptation using
the target-domain observed discount as the behavioral signal. The adapter
maps the 111-D Amazon state into the valid 4-D NeoRL state range, and the
frozen NeoRL source policy is used as the teacher. The objective minimizes
the difference between the source-policy-implied discount and the observed
Amazon discount, with a small source-state distribution regularizer.

This is explicitly an adaptation necessitated by the available public data,
not a claim of exact reproduction of the paper's paired-state Eq. (21).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.transfer.neorl_source_policy import (
    NeoRLSourcePolicy,
    NeoRLSourcePolicyConfig,
)
from src.transfer.transfer_adapter import TransferAdapter, TransferAdapterConfig


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def find_products_file() -> Path:
    candidates = [
        PROJECT_ROOT / "data" / "processed" / "products_clean.parquet",
        PROJECT_ROOT / "data" / "processed" / "products_clean.csv",
        PROJECT_ROOT / "data" / "raw" / "products.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Could not find products_clean.parquet/csv or data/raw/products.csv"
    )


def load_products(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    try:
        return pd.read_csv(path, encoding="cp1252")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin1")


def find_column(df: pd.DataFrame, candidates: list[str]) -> str:
    normalized = {str(c).strip().lower(): str(c) for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in normalized:
            return normalized[candidate.lower()]
    raise KeyError(f"Could not find any of {candidates}")


def load_target_training_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    state_dir = PROJECT_ROOT / "data" / "processed" / "state"
    state_path = state_dir / "state.npy"
    row_idx_path = state_dir / "state_row_indices.npy"

    if not state_path.exists() or not row_idx_path.exists():
        raise FileNotFoundError(
            "Expected data/processed/state/state.npy and state_row_indices.npy"
        )

    state = np.load(state_path).astype(np.float32)
    row_indices = np.load(row_idx_path).astype(np.int64)

    if state.ndim != 2 or state.shape[1] != 111:
        raise ValueError(f"Expected state shape [N,111], got {state.shape}")
    if len(state) != len(row_indices):
        raise ValueError("State rows and row-index rows do not match")

    products_path = find_products_file()
    df = load_products(products_path)

    if len(df) < int(row_indices.max()) + 1:
        raise ValueError("Product table is shorter than state row-index references")

    discount_col = find_column(df, ["Discount"])
    month_col = find_column(df, ["MonthNum", "Deal Month", "Month"])

    discounts = pd.to_numeric(
        df.iloc[row_indices][discount_col].astype(str).str.replace("%", "", regex=False),
        errors="coerce",
    ).to_numpy(dtype=np.float32)

    months = pd.to_numeric(
        df.iloc[row_indices][month_col], errors="coerce"
    ).to_numpy(dtype=np.float32)

    valid = (
        np.isfinite(state).all(axis=1)
        & np.isfinite(discounts)
        & np.isfinite(months)
        & (discounts >= 0.0)
        & (discounts <= 40.0)
    )

    # Chronological training split: Jan-Jun, matching the existing target PPO setup.
    train_mask = valid & (months >= 1) & (months <= 6)
    val_mask = valid & (months >= 7) & (months <= 8)

    return (
        state[train_mask],
        discounts[train_mask],
        state[val_mask],
        discounts[val_mask],
    )


def load_neorl_source_policy(
    checkpoint: Path, device: torch.device
) -> NeoRLSourcePolicy:
    config = NeoRLSourcePolicyConfig()
    policy = NeoRLSourcePolicy(config).to(device)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)

    if isinstance(payload, dict) and "model_state_dict" in payload:
        state_dict = payload["model_state_dict"]
    else:
        state_dict = payload

    policy.load_state_dict(state_dict)
    policy.eval()
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    return policy


@torch.no_grad()
def source_discount_percent(
    source_policy: NeoRLSourcePolicy,
    source_state: torch.Tensor,
) -> torch.Tensor:
    # NeoRL's wrapper divides both stored action dimensions by 5.
    # The second action therefore represents the coupon factor / 5.
    mean_action = source_policy.deterministic_action(source_state)

    coupon_factor = torch.clamp(mean_action[:, 1] * 5.0, 0.60, 0.95)
    discount_percent = (1.0 - coupon_factor) * 100.0
    return discount_percent


def make_source_stats(
    source_obs_path: Path | None,
) -> tuple[np.ndarray, np.ndarray]:
    if source_obs_path is not None and source_obs_path.exists():
        obs = np.load(source_obs_path).astype(np.float32)
    else:
        # These are the observed min/max values from the locally downloaded
        # NeoRL SP training dataset. The script can also derive them from a
        # supplied obs.npy if desired.
        obs_min = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        obs_max = np.array([168.0, 3.2142856, 51.14402, 6.0], dtype=np.float32)
        return obs_min, obs_max

    return obs.min(axis=0), obs.max(axis=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--source-policy", type=str,
                        default="models/transfer/neorl_source_policy/best_model.pth")
    parser.add_argument("--output-dir", type=str,
                        default="models/transfer/adapter")
    parser.add_argument("--behavior-weight", type=float, default=1.0)
    parser.add_argument("--source-prior-weight", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cpu")

    print("=" * 70)
    print("NeoRL -> ItaNet Transfer Adapter Training")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")

    train_state, train_discount, val_state, val_discount = load_target_training_data()

    print("\nTarget data:")
    print("Train states: ", train_state.shape)
    print("Val states:   ", val_state.shape)

    source_policy_path = PROJECT_ROOT / args.source_policy
    if not source_policy_path.exists():
        raise FileNotFoundError(source_policy_path)

    source_policy = load_neorl_source_policy(source_policy_path, device)

    source_min_np, source_max_np = make_source_stats(None)
    source_min = torch.from_numpy(source_min_np)
    source_max = torch.from_numpy(source_max_np)

    adapter = TransferAdapter(
        TransferAdapterConfig(target_state_dim=111, source_state_dim=4),
        source_min=source_min,
        source_max=source_max,
    ).to(device)

    optimizer = torch.optim.Adam(
        adapter.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(train_state),
            torch.from_numpy(train_discount),
        ),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )

    history = []
    best_val = float("inf")
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    source_mean = torch.from_numpy(np.array(
        [16.086088, 0.45454767, 7.5501084, 2.94], dtype=np.float32
    )).to(device)
    source_std = torch.from_numpy(np.array(
        [16.833755, 0.3890327, 8.238263, 2.0232835], dtype=np.float32
    )).to(device)
    source_std = torch.clamp(source_std, min=1e-6)

    val_x = torch.from_numpy(val_state).to(device)
    val_y = torch.from_numpy(val_discount).to(device)

    for epoch in range(1, args.epochs + 1):
        adapter.train()
        train_total = 0.0
        train_behavior = 0.0
        train_prior = 0.0
        count = 0

        for target_x, observed_discount in train_loader:
            target_x = target_x.to(device)
            observed_discount = observed_discount.to(device)

            pseudo_source = adapter(target_x)
            predicted_discount = source_discount_percent(source_policy, pseudo_source)

            behavior_loss = F.mse_loss(
                predicted_discount / 40.0,
                observed_discount / 40.0,
            )

            prior_z = (pseudo_source - source_mean) / source_std
            prior_loss = torch.mean(prior_z.pow(2))

            loss = (
                args.behavior_weight * behavior_loss
                + args.source_prior_weight * prior_loss
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
            optimizer.step()

            n = target_x.shape[0]
            train_total += float(loss.item()) * n
            train_behavior += float(behavior_loss.item()) * n
            train_prior += float(prior_loss.item()) * n
            count += n

        adapter.eval()
        with torch.no_grad():
            val_pseudo = adapter(val_x)
            val_pred = source_discount_percent(source_policy, val_pseudo)
            val_behavior = F.mse_loss(
                val_pred / 40.0,
                val_y / 40.0,
            )
            val_prior_z = (val_pseudo - source_mean) / source_std
            val_prior = torch.mean(val_prior_z.pow(2))
            val_loss = (
                args.behavior_weight * val_behavior
                + args.source_prior_weight * val_prior
            )
            val_mae = torch.mean(torch.abs(val_pred - val_y))
            val_rmse = torch.sqrt(torch.mean((val_pred - val_y) ** 2))

        row = {
            "epoch": epoch,
            "train_loss": train_total / count,
            "train_behavior_loss": train_behavior / count,
            "train_source_prior_loss": train_prior / count,
            "val_loss": float(val_loss.item()),
            "val_behavior_loss": float(val_behavior.item()),
            "val_source_prior_loss": float(val_prior.item()),
            "val_discount_mae": float(val_mae.item()),
            "val_discount_rmse": float(val_rmse.item()),
        }
        history.append(row)

        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss {row['train_loss']:.6f} | "
            f"Val Loss {row['val_loss']:.6f} | "
            f"Val Discount MAE {row['val_discount_mae']:.4f}%"
        )

        if row["val_loss"] < best_val:
            best_val = row["val_loss"]
            torch.save(
                {
                    "model_state_dict": adapter.state_dict(),
                    "config": {
                        "target_state_dim": 111,
                        "source_state_dim": 4,
                        "source_min": source_min_np.tolist(),
                        "source_max": source_max_np.tolist(),
                        "behavior_weight": args.behavior_weight,
                        "source_prior_weight": args.source_prior_weight,
                        "seed": args.seed,
                    },
                    "best_val_loss": best_val,
                },
                output_dir / "best_adapter.pth",
            )

    with open(output_dir / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    print("\n" + "=" * 70)
    print("Adapter training complete")
    print("=" * 70)
    print(f"Best validation loss: {best_val:.6f}")
    print(f"Checkpoint: {output_dir / 'best_adapter.pth'}")


if __name__ == "__main__":
    main()
