"""
Evaluate the NeoRL -> ItaNet transfer adapter.

Run from the project root:
    python scripts\evaluate_transfer_adapter.py

This diagnostic evaluates the adapter against the observed Amazon discount
behavior used during adapter training. It does NOT claim exact reproduction
of the paper's paired-state Eq. (21), because no genuine Amazon/NeoRL state
pairs exist in the public data.
"""

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

from src.transfer.neorl_source_policy import (
    NeoRLSourcePolicy,
    NeoRLSourcePolicyConfig,
)
from src.transfer.transfer_adapter import TransferAdapter, TransferAdapterConfig


def find_products_file() -> Path:
    candidates = [
        PROJECT_ROOT / "data" / "processed" / "products_clean.parquet",
        PROJECT_ROOT / "data" / "processed" / "products_clean.csv",
        PROJECT_ROOT / "data" / "raw" / "products.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("Could not find the processed/raw products file.")


def load_products(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    try:
        return pd.read_csv(path, encoding="cp1252")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin1")


def find_column(df: pd.DataFrame, candidates: list[str]) -> str:
    normalized = {str(c).strip().lower(): str(c) for c in df.columns}
    for c in candidates:
        if c.lower() in normalized:
            return normalized[c.lower()]
    raise KeyError(f"Could not find any of {candidates}")


def load_split() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    state_dir = PROJECT_ROOT / "data" / "processed" / "state"
    states = np.load(state_dir / "state.npy").astype(np.float32)
    row_indices = np.load(state_dir / "state_row_indices.npy").astype(np.int64)

    df = load_products(find_products_file())
    discount_col = find_column(df, ["Discount"])
    month_col = find_column(df, ["MonthNum", "Deal Month", "Month"])

    selected = df.iloc[row_indices]
    discounts = pd.to_numeric(
        selected[discount_col].astype(str).str.replace("%", "", regex=False),
        errors="coerce",
    ).to_numpy(np.float32)
    months = pd.to_numeric(selected[month_col], errors="coerce").to_numpy(np.float32)

    valid = (
        np.isfinite(states).all(axis=1)
        & np.isfinite(discounts)
        & np.isfinite(months)
        & (discounts >= 0)
        & (discounts <= 40)
    )

    train = valid & (months >= 1) & (months <= 6)
    val = valid & (months >= 7) & (months <= 8)

    return states[train], discounts[train], states[val], discounts[val]


def load_source_policy(device: torch.device) -> NeoRLSourcePolicy:
    path = PROJECT_ROOT / "models/transfer/neorl_source_policy/best_model.pth"
    policy = NeoRLSourcePolicy(NeoRLSourcePolicyConfig()).to(device)
    payload = torch.load(path, map_location=device, weights_only=False)
    state_dict = payload["model_state_dict"] if isinstance(payload, dict) and "model_state_dict" in payload else payload
    policy.load_state_dict(state_dict)
    policy.eval()
    for p in policy.parameters():
        p.requires_grad_(False)
    return policy


def load_adapter(device: torch.device) -> TransferAdapter:
    path = PROJECT_ROOT / "models/transfer/adapter/best_adapter.pth"
    payload = torch.load(path, map_location=device, weights_only=False)

    cfg = payload.get("config", {})
    source_min = torch.tensor(cfg.get("source_min", [0, 0, 0, 0]), dtype=torch.float32)
    source_max = torch.tensor(
        cfg.get("source_max", [168.0, 3.2142856, 51.14402, 6.0]),
        dtype=torch.float32,
    )

    adapter = TransferAdapter(
        TransferAdapterConfig(
            target_state_dim=int(cfg.get("target_state_dim", 111)),
            source_state_dim=int(cfg.get("source_state_dim", 4)),
        ),
        source_min=source_min,
        source_max=source_max,
    ).to(device)

    state_dict = payload["model_state_dict"] if isinstance(payload, dict) and "model_state_dict" in payload else payload
    adapter.load_state_dict(state_dict)
    adapter.eval()
    return adapter


def source_discount(policy: NeoRLSourcePolicy, source_state: torch.Tensor) -> torch.Tensor:
    action = policy.deterministic_action(source_state)
    # NeoRL wrapper stores both action dimensions divided by 5.
    coupon_factor = torch.clamp(action[:, 1] * 5.0, 0.60, 0.95)
    return (1.0 - coupon_factor) * 100.0


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))

    # Constant baseline: most common observed discount in the split.
    values, counts = np.unique(y_true, return_counts=True)
    mode = float(values[np.argmax(counts)])
    baseline_mae = float(np.mean(np.abs(y_true - mode)))

    return {
        "mae_percentage_points": mae,
        "rmse_percentage_points": rmse,
        "constant_mode_discount": mode,
        "constant_mode_mae_percentage_points": baseline_mae,
        "mae_improvement_vs_mode_percent": (
            float(100.0 * (baseline_mae - mae) / baseline_mae)
            if baseline_mae > 0 else 0.0
        ),
    }


def distribution(values: np.ndarray) -> dict:
    rounded = np.clip(np.rint(values / 5.0) * 5.0, 0, 40).astype(int)
    unique, counts = np.unique(rounded, return_counts=True)
    total = len(rounded)
    return {
        str(int(k)): round(float(v) / total * 100.0, 3)
        for k, v in zip(unique, counts)
    }


def main() -> None:
    device = torch.device("cpu")
    train_x, train_y, val_x, val_y = load_split()

    policy = load_source_policy(device)
    adapter = load_adapter(device)

    print("=" * 72)
    print("NeoRL -> ItaNet Transfer Adapter Diagnostic")
    print("=" * 72)
    print(f"Train: {train_x.shape}")
    print(f"Val:   {val_x.shape}")
    print()

    results = {}

    for name, x_np, y_np in [
        ("train", train_x, train_y),
        ("validation", val_x, val_y),
    ]:
        x = torch.from_numpy(x_np).to(device)

        with torch.no_grad():
            pseudo = adapter(x)
            pred = source_discount(policy, pseudo).cpu().numpy()

        m = metrics(y_np, pred)

        print(f"[{name.upper()}]")
        print(f"Observed mean discount: {np.mean(y_np):.4f}%")
        print(f"Predicted mean discount: {np.mean(pred):.4f}%")
        print(f"Observed std: {np.std(y_np):.4f}")
        print(f"Predicted std: {np.std(pred):.4f}")
        print(f"MAE: {m['mae_percentage_points']:.4f} percentage points")
        print(f"RMSE: {m['rmse_percentage_points']:.4f} percentage points")
        print(f"Mode baseline: {m['constant_mode_discount']:.1f}%")
        print(f"Mode baseline MAE: {m['constant_mode_mae_percentage_points']:.4f}")
        print(f"MAE improvement vs mode: {m['mae_improvement_vs_mode_percent']:.2f}%")
        print(f"Observed action distribution: {distribution(y_np)}")
        print(f"Adapter action distribution:  {distribution(pred)}")

        pseudo_np = pseudo.cpu().numpy()
        print("Pseudo-state mean:", np.round(pseudo_np.mean(axis=0), 4))
        print("Pseudo-state std: ", np.round(pseudo_np.std(axis=0), 4))
        print()

        results[name] = {
            "n": int(len(y_np)),
            "metrics": m,
            "observed_distribution": distribution(y_np),
            "adapter_distribution": distribution(pred),
            "observed_mean": float(np.mean(y_np)),
            "predicted_mean": float(np.mean(pred)),
            "observed_std": float(np.std(y_np)),
            "predicted_std": float(np.std(pred)),
            "pseudo_state_mean": pseudo_np.mean(axis=0).tolist(),
            "pseudo_state_std": pseudo_np.std(axis=0).tolist(),
        }

    # State sensitivity: compare predictions for nearby batches and
    # report the spread of adapter output probabilities/discounts.
    with torch.no_grad():
        sample = torch.from_numpy(val_x[: min(2048, len(val_x))]).to(device)
        pred_sample = source_discount(policy, adapter(sample)).cpu().numpy()

    print("[STATE SENSITIVITY]")
    print(f"Prediction min: {pred_sample.min():.4f}%")
    print(f"Prediction max: {pred_sample.max():.4f}%")
    print(f"Prediction std: {pred_sample.std():.4f} percentage points")
    print(f"Unique rounded discount levels: {sorted(set(np.clip(np.rint(pred_sample/5)*5, 0, 40).astype(int).tolist()))}")

    results["state_sensitivity"] = {
        "sample_n": int(len(pred_sample)),
        "prediction_min": float(pred_sample.min()),
        "prediction_max": float(pred_sample.max()),
        "prediction_std": float(pred_sample.std()),
        "unique_rounded_levels": sorted(
            set(np.clip(np.rint(pred_sample / 5) * 5, 0, 40).astype(int).tolist())
        ),
    }

    out = PROJECT_ROOT / "reports" / "transfer" / "transfer_adapter_diagnostic.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print()
    print("=" * 72)
    print(f"Saved diagnostic: {out}")
    print("=" * 72)


if __name__ == "__main__":
    main()
