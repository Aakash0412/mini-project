"""
Phase 5: Sales Predictor — Training, Evaluation, and Freeze

Architecture (IMPLEMENTATION DECISION — not fully specified in paper):
    Input (80-D: 79-D state + 1-D discount action)
    → Linear(80, 256) → BatchNorm → ReLU → Dropout(0.2)
    → Linear(256, 128) → BatchNorm → ReLU → Dropout(0.1)
    → Linear(128, 64)  → ReLU
    → Linear(64, 1)

Target: Estimated Monthly Sales Growth Rate (%)
After training, the model is frozen and saved. PPO will use it as the
transition/simulation engine during policy optimisation.

Paper principle: Sales predictor is trained first, then frozen before DRL.
"""

import sys
import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── Paper-defined discount action space ──────────────────────
DISCOUNT_BINS = [0, 5, 10, 15, 20, 25, 30, 35, 40]  # percent


# ── Model definition ─────────────────────────────────────────
class SalesPredictorMLP(nn.Module):
    """
    Lightweight MLP sales predictor.

    IMPLEMENTATION DECISION: Architecture not fully specified in paper.
    Chosen to be the smallest viable network for 8 GB Apple Silicon.
    Labelled as IMPLEMENTATION DECISION in all reports.
    """

    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ── Data prep ────────────────────────────────────────────────
def prepare_data(state: np.ndarray, df: pd.DataFrame):
    target = df["Estimated Monthly Sales Growth Rate"].fillna(0.0).values.astype(np.float32)
    discount = df["Discount"].fillna(0.0).values.astype(np.float32).reshape(-1, 1)
    X = np.concatenate([state, discount], axis=1).astype(np.float32)
    return X, target


# ── Training loop ────────────────────────────────────────────
def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 60,
    patience: int = 8,
    lr: float = 1e-3,
    save_path: str = "models/sales_predictor/best_model.pth",
):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=4)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    patience_counter = 0
    train_losses, val_losses = [], []

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for epoch in range(1, epochs + 1):
        # ── train ──
        model.train()
        running = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running += loss.item() * bx.size(0)
        train_loss = running / len(train_loader.dataset)

        # ── val ──
        model.eval()
        val_running = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                val_running += criterion(model(bx), by).item() * bx.size(0)
        val_loss = val_running / len(val_loader.dataset)

        scheduler.step(val_loss)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if epoch % 5 == 0:
            print(f"  Epoch {epoch:3d} | train_MSE={train_loss:.4f} | val_MSE={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    return train_losses, val_losses


# ── Evaluation ───────────────────────────────────────────────
def evaluate(model, loader, device):
    model.eval()
    preds, actuals = [], []
    with torch.no_grad():
        for bx, by in loader:
            bx = bx.to(device)
            preds.extend(model(bx).cpu().numpy())
            actuals.extend(by.numpy())
    return np.array(preds), np.array(actuals)


# ── Counterfactual sanity check ──────────────────────────────
def counterfactual_sanity_check(
    model, state_test, df_test, device, n_products=10, out_dir="reports/sales_response_curves"
):
    """
    For a sample of products, predict sales growth under each discount tier.
    Detects: NaN, constant predictions, extreme values, instability.
    """
    os.makedirs(out_dir, exist_ok=True)
    model.eval()

    sample_idx = np.random.default_rng(42).choice(len(state_test), size=min(n_products, len(state_test)), replace=False)

    issues = []

    fig, axes = plt.subplots(2, 5, figsize=(18, 6))
    axes = axes.flatten()

    for plot_i, idx in enumerate(sample_idx):
        s = state_test[idx]
        responses = []

        for disc in DISCOUNT_BINS:
            inp = np.concatenate([s, [disc]], axis=0).astype(np.float32)
            with torch.no_grad():
                inp_t = torch.tensor(inp).unsqueeze(0).to(device)
                pred = model(inp_t).item()
            responses.append(pred)

        responses = np.array(responses)

        # Checks
        if np.isnan(responses).any():
            issues.append(f"Product idx {idx}: NaN in predictions")
        if np.all(responses == responses[0]):
            issues.append(f"Product idx {idx}: Identical predictions for all discounts")
        if np.abs(responses).max() > 1000:
            issues.append(f"Product idx {idx}: Extreme predictions (max={np.abs(responses).max():.1f})")

        ax = axes[plot_i]
        ax.plot(DISCOUNT_BINS, responses, marker="o", linewidth=2)
        ax.set_title(f"Product {idx}", fontsize=9)
        ax.set_xlabel("Discount %")
        ax.set_ylabel("Pred. Sales Growth %")
        ax.grid(True, alpha=0.3)

    plt.suptitle("Counterfactual Sales Response Curves (Sample)", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "sample_response_curves.png"), dpi=120)
    plt.close()

    with open(os.path.join(out_dir, "sanity_check_issues.json"), "w") as f:
        json.dump({"issues": issues, "n_checked": int(n_products)}, f, indent=4)

    if issues:
        print(f"\n  ⚠ Sanity check flagged {len(issues)} issue(s):")
        for issue in issues:
            print(f"    {issue}")
    else:
        print(f"\n  Counterfactual sanity check PASSED ({n_products} products).")

    return issues


# ── Main ─────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("PHASE 5 — Sales Predictor Training")
    print("=" * 55)

    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load state and dataset ────────────────────────────────
    state = np.load("data/processed/state/state.npy")
    df = pd.read_parquet("data/processed/products_clean.parquet").reset_index(drop=True)
    assert len(state) == len(df), "State / dataset length mismatch"

    X, y = prepare_data(state, df)

    train_mask = df["MonthNum"].isin([1, 2, 3, 4, 5, 6]).values
    val_mask   = df["MonthNum"].isin([7, 8]).values
    test_mask  = df["MonthNum"].isin([9, 10]).values

    X_train, y_train = X[train_mask], y[train_mask]
    X_val,   y_val   = X[val_mask],   y[val_mask]
    X_test,  y_test  = X[test_mask],  y[test_mask]

    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    # ── DataLoaders ───────────────────────────────────────────
    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
        batch_size=256, shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val), torch.tensor(y_val)),
        batch_size=512, shuffle=False,
    )
    test_loader = DataLoader(
        TensorDataset(torch.tensor(X_test), torch.tensor(y_test)),
        batch_size=512, shuffle=False,
    )

    # ── Model ─────────────────────────────────────────────────
    input_dim = X.shape[1]    # 79-D state + 1-D discount = 80
    model = SalesPredictorMLP(input_dim=input_dim).to(device)
    print(f"Input dim: {input_dim}  (state {state.shape[1]} + action 1)")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ── Train ─────────────────────────────────────────────────
    print("\nTraining...")
    train_losses, val_losses = train(
        model, train_loader, val_loader, device,
        epochs=60, patience=8, lr=1e-3,
        save_path="models/sales_predictor/best_model.pth",
    )

    # ── Load best ─────────────────────────────────────────────
    model.load_state_dict(torch.load("models/sales_predictor/best_model.pth", map_location=device))
    model.eval()

    # ── Evaluate on test set ──────────────────────────────────
    print("\nEvaluating on test set...")
    preds, actuals = evaluate(model, test_loader, device)

    mae  = float(mean_absolute_error(actuals, preds))
    rmse = float(np.sqrt(mean_squared_error(actuals, preds)))
    r2   = float(r2_score(actuals, preds))

    print(f"  MAE  : {mae:.4f}")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  R²   : {r2:.4f}")

    # ── Save loss curves ──────────────────────────────────────
    os.makedirs("reports", exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label="Train MSE")
    plt.plot(val_losses,   label="Val MSE")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Sales Predictor — Training Curves")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("reports/sales_predictor_training_curves.png", dpi=120)
    plt.close()

    # ── Residual plot ─────────────────────────────────────────
    residuals = actuals - preds
    plt.figure(figsize=(8, 4))
    plt.hist(residuals, bins=60, edgecolor="k", alpha=0.75)
    plt.xlabel("Residual (actual − predicted)")
    plt.ylabel("Count")
    plt.title("Sales Predictor — Residual Distribution")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("reports/sales_predictor_residuals.png", dpi=120)
    plt.close()

    # ── Per-discount-level performance ───────────────────────
    df_test_sub = df[test_mask].reset_index(drop=True)
    per_disc = {}
    for d in sorted(df_test_sub["Discount"].unique()):
        mask = (df_test_sub["Discount"] == d).values
        if mask.sum() == 0:
            continue
        per_disc[int(d)] = {
            "n": int(mask.sum()),
            "mae": float(mean_absolute_error(actuals[mask], preds[mask])),
            "rmse": float(np.sqrt(mean_squared_error(actuals[mask], preds[mask]))),
        }

    # ── Save metadata alongside weights ──────────────────────
    config = {
        "architecture": "MLP: 80→256→BN→128→BN→64→1  (IMPLEMENTATION DECISION)",
        "input_dim": input_dim,
        "state_dim": int(state.shape[1]),
        "action_dim": 1,
        "activation": "ReLU",
        "optimizer": "Adam",
        "lr": 1e-3,
        "weight_decay": 1e-5,
        "batch_size": 256,
        "epochs_run": len(train_losses),
        "early_stopping_patience": 8,
        "loss": "MSELoss",
        "random_seed": 42,
        "device": str(device),
        "train_samples": int(len(X_train)),
        "val_samples": int(len(X_val)),
        "test_samples": int(len(X_test)),
    }

    metrics = {
        "test_MAE": mae,
        "test_RMSE": rmse,
        "test_R2": r2,
        "per_discount_performance": per_disc,
    }

    with open("models/sales_predictor/config.json", "w") as f:
        json.dump(config, f, indent=4)
    with open("models/sales_predictor/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    with open("reports/sales_predictor_metrics.json", "w") as f:
        json.dump({**config, **metrics}, f, indent=4)

    print("\nSaved model weights and metadata to models/sales_predictor/")

    # ── FREEZE reminder ───────────────────────────────────────
    for param in model.parameters():
        param.requires_grad = False
    print("Model FROZEN (requires_grad=False). Ready for PPO phase.")

    # ── Counterfactual sanity check ───────────────────────────
    print("\nRunning counterfactual sanity check...")
    state_test_arr = state[test_mask]
    counterfactual_sanity_check(model, state_test_arr, df_test_sub, device, n_products=10)

    print("\nPhase 5 complete.")
    print(f"  Best model: models/sales_predictor/best_model.pth")
    print(f"  MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}")


if __name__ == "__main__":
    main()
