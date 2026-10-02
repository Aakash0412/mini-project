from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.transfer.neorl_source_policy import NeoRLSourcePolicy, NeoRLSourcePolicyConfig
from src.transfer.policy_distillation import (
    PolicyDistillationConfig,
    TargetPricingPolicy,
    discount_index,
)


def parse_args():
    p = argparse.ArgumentParser(description="Distill NeoRL source behavior into the 111-D target pricing policy.")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--learning-rate", type=float, default=8e-4)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_products():
    path = PROJECT_ROOT / "data/processed/products_clean.parquet"
    return pd.read_parquet(path)


def load_states():
    d = PROJECT_ROOT / "data/processed/state"
    states = np.load(d / "state.npy").astype(np.float32)
    rows = np.load(d / "state_row_indices.npy").astype(np.int64)
    return states, rows


def load_adapter(device):
    from src.transfer.transfer_adapter import TransferAdapter, TransferAdapterConfig

    path = PROJECT_ROOT / "models/transfer/adapter/best_adapter.pth"
    payload = torch.load(path, map_location=device, weights_only=False)
    cfg = payload.get("config", {}) if isinstance(payload, dict) else {}
    adapter = TransferAdapter(
        TransferAdapterConfig(
            target_state_dim=int(cfg.get("target_state_dim", 111)),
            source_state_dim=int(cfg.get("source_state_dim", 4)),
        ),
        source_min=torch.tensor(cfg.get("source_min", [0, 0, 0, 0]), dtype=torch.float32),
        source_max=torch.tensor(cfg.get("source_max", [168.0, 3.2142856, 51.14402, 6.0]), dtype=torch.float32),
    ).to(device)
    state_dict = payload["model_state_dict"] if isinstance(payload, dict) and "model_state_dict" in payload else payload
    adapter.load_state_dict(state_dict)
    adapter.eval()
    for p in adapter.parameters():
        p.requires_grad_(False)
    return adapter


def load_source_policy(device):
    path = PROJECT_ROOT / "models/transfer/neorl_source_policy/best_model.pth"
    payload = torch.load(path, map_location=device, weights_only=False)
    policy = NeoRLSourcePolicy(NeoRLSourcePolicyConfig()).to(device)
    state_dict = payload["model_state_dict"] if isinstance(payload, dict) and "model_state_dict" in payload else payload
    policy.load_state_dict(state_dict)
    policy.eval()
    for p in policy.parameters():
        p.requires_grad_(False)
    return policy


def source_action_distribution(policy, source_state):
    """Return the source policy's 2-D Normal means when available."""
    with torch.no_grad():
        out = policy.forward(source_state)
        if isinstance(out, tuple):
            mean = out[0]
        elif hasattr(out, "loc"):
            mean = out.loc
        else:
            mean = out
    return mean


def source_discount(policy, source_state):
    mean = source_action_distribution(policy, source_state)
    # NeoRL stores both action dimensions divided by 5.
    coupon_factor = torch.clamp(mean[:, 1] * 5.0, 0.60, 0.95)
    return (1.0 - coupon_factor) * 100.0


def build_targets(states, rows, products, device):
    aligned = products.iloc[rows]
    months = pd.to_numeric(aligned["MonthNum"], errors="coerce").to_numpy()
    valid = np.isfinite(months)

    train = valid & (months >= 1) & (months <= 6)
    val = valid & (months >= 7) & (months <= 8)

    source = load_source_policy(device)
    adapter = load_adapter(device)

    def make(mask):
        x = torch.from_numpy(states[mask]).to(device)
        with torch.no_grad():
            pseudo = adapter(x)
            discounts = source_discount(source, pseudo)
        y = discount_index(discounts).cpu().numpy()
        return states[mask], y, discounts.cpu().numpy()

    return make(train), make(val)


def train_epoch(model, loader, optimizer, temperature):
    model.train()
    total = 0.0
    n = 0
    ce = nn.CrossEntropyLoss()
    for x, y in loader:
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = ce(logits / max(temperature, 1e-6), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        total += float(loss.item()) * len(x)
        n += len(x)
    return total / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, temperature):
    model.eval()
    ce = nn.CrossEntropyLoss()
    losses, correct, n = 0.0, 0, 0
    for x, y in loader:
        logits = model(x)
        loss = ce(logits / max(temperature, 1e-6), y)
        pred = torch.argmax(logits, dim=-1)
        losses += float(loss.item()) * len(x)
        correct += int((pred == y).sum())
        n += len(x)
    return losses / max(n, 1), correct / max(n, 1)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cpu")

    states, rows = load_states()
    products = load_products()
    (train_x, train_y, train_disc), (val_x, val_y, val_disc) = build_targets(
        states, rows, products, device
    )

    print("=" * 70)
    print("NeoRL -> ItaNet POLICY DISTILLATION")
    print("=" * 70)
    print(f"Train states: {train_x.shape}")
    print(f"Val states:   {val_x.shape}")
    print("Source-derived target discounts are generated from the frozen NeoRL policy.")
    print("0% is retained as a target action but is not produced by the source mapping.")

    train_ds = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y).long())
    val_ds = TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y).long())
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    cfg = PolicyDistillationConfig(
        learning_rate=args.learning_rate,
        temperature=args.temperature,
        seed=args.seed,
    )
    model = TargetPricingPolicy(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    best = float("inf")
    out_dir = PROJECT_ROOT / "models/transfer/distillation"
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "best_model.pth"

    history = []
    for epoch in range(1, args.epochs + 1):
        tr = train_epoch(model, train_loader, optimizer, cfg.temperature)
        vl, acc = evaluate(model, val_loader, cfg.temperature)
        print(f"Epoch {epoch:03d} | Train CE {tr:.6f} | Val CE {vl:.6f} | Val Acc {acc*100:.2f}%")
        history.append({"epoch": epoch, "train_ce": tr, "val_ce": vl, "val_accuracy": acc})
        if vl < best:
            best = vl
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": vars(cfg),
                    "best_val_loss": best,
                    "source": "NeoRL frozen source policy + existing 111D->4D adapter",
                    "note": "Behavioral distillation adaptation; not exact paper Eq.21 paired-state reproduction.",
                },
                best_path,
            )

    report = {
        "best_val_loss": best,
        "train_n": len(train_y),
        "val_n": len(val_y),
        "source_discount_train_mean": float(train_disc.mean()),
        "source_discount_val_mean": float(val_disc.mean()),
        "source_discount_train_std": float(train_disc.std()),
        "source_discount_val_std": float(val_disc.std()),
        "history": history,
    }
    report_path = PROJECT_ROOT / "reports/transfer/policy_distillation_training.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("=" * 70)
    print("Distillation complete")
    print(f"Best validation loss: {best:.6f}")
    print(f"Checkpoint: {best_path}")
    print(f"Report: {report_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
