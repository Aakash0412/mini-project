"""
scripts/train_sales_predictor_v3.py — v3 Sales Predictor Training

Fixes the missing-input bug present in the v1/v2 predictors by adding
Last Monthly Sales (Q_t) as an explicit predictor input, yielding
input_dim = 113  (state_111 ‖ last_monthly_sales_1 ‖ discount_1).

This is a fresh script — it does NOT modify train_sales_predictor_train_only.py.

Why this fix?
-------------
The v1/v2 predictor was trained to predict
    (Q_{t+1} - Q_t) / Q_t
without ever seeing Q_t. The paper's Eq. 2 defines:
    ĝ = f(c_i, d_{t,i}, a_{t,i})
where d_{t,i} includes Q_t. Omitting Q_t produced test R² ≈ 0.025
(and the log-winsor retrain gave R² = -0.20). Adding Q_t is the fix.

Loss / transform
----------------
Plain MSE on the raw target. The log-winsor transform was tried in v2
and yielded R² = -0.20; it is NOT used here.

Output directories
------------------
    models/sales_predictor_v3/      — best_model.pth
    reports/sales_predictor_v3/     — metrics.json, training_history.csv, config.json

Refuses to run if either output directory already exists.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
STATE_DIR = ROOT / "data" / "processed" / "state_train_only"
DATA_PATH = ROOT / "data" / "processed" / "products_clean.parquet"
OUTPUT_DIR = ROOT / "models" / "sales_predictor_v3"
REPORT_DIR = ROOT / "reports" / "sales_predictor_v3"

TARGET = "Estimated Monthly Sales Growth Rate"
SALES_COL = "Last Monthly Sales"
DISCOUNT_COL = "Discount"

SEED = 42
EPOCHS = 100
PATIENCE = 12
BATCH_SIZE = 256
EVAL_BATCH_SIZE = 512
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class SalesPredictorMLP(nn.Module):
    """
    MLP Sales Predictor — 113-D input (v3).

    Architecture identical to train_sales_predictor_train_only.py
    but with input_dim=113 (state_111 ‖ last_monthly_sales_1 ‖ discount_1).

        Linear(in, 256) → BatchNorm1d(256) → ReLU → Dropout(0.2)
        Linear(256, 128) → BatchNorm1d(128) → ReLU → Dropout(0.1)
        Linear(128, 64) → ReLU
        Linear(64, 1)
    """

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
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
        return self.network(x)

    def predict(
        self,
        state: torch.Tensor,
        last_sales: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convenience method: concatenate state ‖ last_sales ‖ action and forward.

        Parameters
        ----------
        state : (B, 111) float tensor
        last_sales : (B,) or (B, 1) float tensor — last monthly sales Q_t
        action : (B,) or (B, 1) float tensor — discount percentage
        """
        if last_sales.ndim == 1:
            last_sales = last_sales.unsqueeze(1)
        if action.ndim == 1:
            action = action.unsqueeze(1)
        x = torch.cat([state, last_sales, action], dim=1)
        return self.forward(x)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute MAE, RMSE, R² from arrays of actuals and predictions."""
    residual = y_true - y_pred
    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(np.mean(residual ** 2)))

    denominator = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = (
        float(1.0 - np.sum(residual ** 2) / denominator)
        if denominator > 0
        else None
    )

    return {
        "n": int(len(y_true)),
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    # Limit CPU threads on Windows to avoid oversubscription.
    torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))

    set_seed(SEED)

    # Refuse to overwrite existing experiment outputs.
    if OUTPUT_DIR.exists():
        print(
            f"[ERROR] Output directory already exists: {OUTPUT_DIR}\n"
            "Remove it or choose a different directory before running."
        )
        raise SystemExit(1)
    if REPORT_DIR.exists():
        print(
            f"[ERROR] Report directory already exists: {REPORT_DIR}\n"
            "Remove it or choose a different directory before running."
        )
        raise SystemExit(1)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------
    state_path = STATE_DIR / "state.npy"
    index_path = STATE_DIR / "state_row_indices.npy"

    if not state_path.exists():
        raise FileNotFoundError(f"State file not found: {state_path}")
    if not index_path.exists():
        raise FileNotFoundError(f"Index file not found: {index_path}")
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Products parquet not found: {DATA_PATH}")

    state = np.load(state_path).astype(np.float32)          # (N, 111)
    indices = np.load(index_path).astype(np.int64)           # (N,)

    df_full = pd.read_parquet(DATA_PATH).reset_index(drop=True)
    df = df_full.iloc[indices].reset_index(drop=True)

    if len(state) != len(df):
        raise ValueError(
            f"State ({len(state)}) and dataframe ({len(df)}) row counts do not match."
        )

    # ------------------------------------------------------------------
    # Extract columns
    # ------------------------------------------------------------------
    months = df["MonthNum"].to_numpy()
    targets = df[TARGET].to_numpy(dtype=np.float32)
    sales = df[SALES_COL].to_numpy(dtype=np.float32)
    discounts = df[DISCOUNT_COL].to_numpy(dtype=np.float32)

    # Validate: do NOT impute; raise on NaN with row count.
    sales_nan_count = int(np.isnan(sales).sum())
    if sales_nan_count > 0:
        raise ValueError(
            f"{SALES_COL} contains {sales_nan_count} NaN rows. "
            "Resolve missing values explicitly before rerunning."
        )
    if not np.all(sales >= 0.0):
        neg_count = int((sales < 0.0).sum())
        raise ValueError(
            f"{SALES_COL} contains {neg_count} negative values. "
            "Sales quantities must be non-negative."
        )

    targets_nan_count = int(np.isnan(targets).sum())
    if targets_nan_count > 0:
        raise ValueError(
            f"{TARGET} contains {targets_nan_count} NaN rows. "
            "Resolve missing values explicitly before rerunning."
        )
    if not np.all(np.isfinite(targets)):
        raise ValueError(f"{TARGET} contains non-finite values.")

    disc_nan_count = int(np.isnan(discounts).sum())
    if disc_nan_count > 0:
        raise ValueError(
            f"{DISCOUNT_COL} contains {disc_nan_count} NaN rows. "
            "Resolve missing values explicitly before rerunning."
        )
    if not np.all(np.isfinite(discounts)):
        raise ValueError(f"{DISCOUNT_COL} contains non-finite values.")

    # ------------------------------------------------------------------
    # Chronological splits (months 1–6 train, 7–8 val, 9–10 test)
    # ------------------------------------------------------------------
    train_mask = np.isin(months, [1, 2, 3, 4, 5, 6])
    val_mask = np.isin(months, [7, 8])
    test_mask = np.isin(months, [9, 10])

    if np.any(train_mask & val_mask):
        raise ValueError("train_mask and val_mask overlap.")
    if np.any(train_mask & test_mask):
        raise ValueError("train_mask and test_mask overlap.")
    if np.any(val_mask & test_mask):
        raise ValueError("val_mask and test_mask overlap.")

    # ------------------------------------------------------------------
    # Input construction — THE FIX: include Last Monthly Sales
    # ------------------------------------------------------------------
    # x shape: (N, 113) = state_111 ‖ last_monthly_sales_1 ‖ discount_1
    x = np.concatenate(
        [state, sales[:, None], discounts[:, None]], axis=1
    ).astype(np.float32)
    y = targets[:, None].astype(np.float32)

    if x.shape[1] != 113:
        raise AssertionError(
            f"Expected x.shape[1] == 113, got {x.shape[1]}."
        )

    if not np.all(np.isfinite(x)):
        raise ValueError("Combined input matrix x contains non-finite values.")
    if not np.all(np.isfinite(y)):
        raise ValueError("Target matrix y contains non-finite values.")

    # ------------------------------------------------------------------
    # DataLoaders
    # ------------------------------------------------------------------
    def make_loader(mask: np.ndarray, shuffle: bool) -> DataLoader:
        dataset = TensorDataset(
            torch.from_numpy(x[mask]),
            torch.from_numpy(y[mask]),
        )
        return DataLoader(
            dataset,
            batch_size=BATCH_SIZE if shuffle else EVAL_BATCH_SIZE,
            shuffle=shuffle,
            drop_last=False,
        )

    train_loader = make_loader(train_mask, shuffle=True)
    val_loader = make_loader(val_mask, shuffle=False)
    test_loader = make_loader(test_mask, shuffle=False)

    # ------------------------------------------------------------------
    # Model, optimiser, scheduler, loss
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SalesPredictorMLP(input_dim=113).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )
    criterion = nn.MSELoss()

    print(f"Device            : {device}")
    print(f"State dimension   : {state.shape[1]}")
    print(f"Predictor input   : {x.shape[1]}  (state_111 || last_monthly_sales_1 || discount_1)")
    print(
        f"Rows — train: {int(train_mask.sum())}, "
        f"val: {int(val_mask.sum())}, "
        f"test: {int(test_mask.sum())}"
    )
    print(f"Target: {TARGET}")
    print(
        f"Training target mean/std: "
        f"{targets[train_mask].mean():.4f} / {targets[train_mask].std():.4f}"
    )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    history = []

    def evaluate_loss(loader: DataLoader) -> float:
        model.eval()
        total_loss = 0.0
        total_rows = 0
        with torch.no_grad():
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                pred = model(batch_x)
                loss = criterion(pred, batch_y)
                total_loss += loss.item() * len(batch_x)
                total_rows += len(batch_x)
        return total_loss / total_rows

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        total_rows = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad(set_to_none=True)
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * len(batch_x)
            total_rows += len(batch_x)

        train_loss = total_loss / total_rows
        val_loss = evaluate_loss(val_loader)
        scheduler.step(val_loss)

        history.append({
            "epoch": epoch,
            "train_mse": train_loss,
            "validation_mse": val_loss,
            "learning_rate": optimizer.param_groups[0]["lr"],
        })

        print(
            f"Epoch {epoch:03d}/{EPOCHS} | "
            f"train MSE {train_loss:.6f} | "
            f"val MSE {val_loss:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0

            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_dim": 113,
                    "state_dim": 111,
                    "seed": SEED,
                    "best_epoch": best_epoch,
                    "best_validation_mse": best_val_loss,
                    "input_contract": "state_111 || last_monthly_sales_1 || discount_1",
                },
                OUTPUT_DIR / "best_model.pth",
            )
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch}.")
            break

    # ------------------------------------------------------------------
    # Restore best checkpoint and evaluate
    # ------------------------------------------------------------------
    ckpt = torch.load(
        OUTPUT_DIR / "best_model.pth",
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(ckpt["model_state_dict"])

    def predict_loader(loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
        model.eval()
        predictions, actuals = [], []
        with torch.no_grad():
            for batch_x, batch_y in loader:
                pred = model(batch_x.to(device)).cpu().numpy().reshape(-1)
                predictions.extend(pred.tolist())
                actuals.extend(batch_y.numpy().reshape(-1).tolist())
        return np.asarray(actuals), np.asarray(predictions)

    train_actual, train_pred = predict_loader(train_loader)
    val_actual, val_pred = predict_loader(val_loader)
    test_actual, test_pred = predict_loader(test_loader)

    train_metrics = regression_metrics(train_actual, train_pred)
    val_metrics = regression_metrics(val_actual, val_pred)
    test_metrics = regression_metrics(test_actual, test_pred)

    # ------------------------------------------------------------------
    # Write config.json (what CounterfactualSimulatorV3 reads)
    # ------------------------------------------------------------------
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    config_payload = {
        "input_dim": 113,
        "state_dim": 111,
        "input_contract": "state_111 || last_monthly_sales_1 || discount_1",
        "target": TARGET,
        "architecture": "MLP(256,128,64,1)",
    }
    (OUTPUT_DIR / "config.json").write_text(
        json.dumps(config_payload, indent=2), encoding="utf-8"
    )
    # Mirror config into the report directory as well.
    (REPORT_DIR / "config.json").write_text(
        json.dumps(config_payload, indent=2), encoding="utf-8"
    )

    # ------------------------------------------------------------------
    # Write training_history.csv
    # ------------------------------------------------------------------
    pd.DataFrame(history).to_csv(REPORT_DIR / "training_history.csv", index=False)

    # ------------------------------------------------------------------
    # Write metrics.json
    # ------------------------------------------------------------------
    results = {
        "experiment": "sales_predictor_v3_last_sales_input",
        "target": TARGET,
        "state_dim": 111,
        "input_dim": 113,
        "input_contract": "state_111 || last_monthly_sales_1 || discount_1",
        "seed": SEED,
        "best_epoch": best_epoch,
        "best_validation_mse": float(best_val_loss),
        "split_months": {
            "train": [1, 2, 3, 4, 5, 6],
            "validation": [7, 8],
            "test": [9, 10],
        },
        "row_counts": {
            "train": int(train_mask.sum()),
            "validation": int(val_mask.sum()),
            "test": int(test_mask.sum()),
        },
        "metrics": {
            "train": train_metrics,
            "validation": val_metrics,
            "test": test_metrics,
        },
        "target_distribution": {
            "train": {
                "mean": float(targets[train_mask].mean()),
                "std": float(targets[train_mask].std()),
                "min": float(targets[train_mask].min()),
                "max": float(targets[train_mask].max()),
            },
            "validation": {
                "mean": float(targets[val_mask].mean()),
                "std": float(targets[val_mask].std()),
                "min": float(targets[val_mask].min()),
                "max": float(targets[val_mask].max()),
            },
            "test": {
                "mean": float(targets[test_mask].mean()),
                "std": float(targets[test_mask].std()),
                "min": float(targets[test_mask].min()),
                "max": float(targets[test_mask].max()),
            },
        },
        "prediction_distribution": {
            "train": {
                "mean": float(train_pred.mean()),
                "std": float(train_pred.std()),
                "min": float(train_pred.min()),
                "max": float(train_pred.max()),
            },
            "validation": {
                "mean": float(val_pred.mean()),
                "std": float(val_pred.std()),
                "min": float(val_pred.min()),
                "max": float(val_pred.max()),
            },
            "test": {
                "mean": float(test_pred.mean()),
                "std": float(test_pred.std()),
                "min": float(test_pred.min()),
                "max": float(test_pred.max()),
            },
        },
        "comparison_to_baseline": {
            "baseline_test_r2": 0.02497,
            "baseline_prediction_std": 61.07,
            "baseline_source": "reports/sales_predictor_train_only/metrics.json",
            "caveat": (
                "Input now includes last_monthly_sales (paper Eq. 2 d_{t,i}); "
                "0% discount is still out of observed training support (5-40%)."
            ),
        },
    }

    (REPORT_DIR / "metrics.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )

    # ------------------------------------------------------------------
    # Final verdict
    # ------------------------------------------------------------------
    test_r2 = test_metrics["r2"]
    test_mae = test_metrics["mae"]
    test_pred_std = float(test_pred.std())
    baseline_r2 = 0.025

    if test_r2 is not None and test_r2 > baseline_r2:
        verdict = "IMPROVED"
    elif test_r2 is not None and test_r2 > 0.01:
        verdict = "MARGINAL"
    else:
        verdict = "NO IMPROVEMENT — file as negative finding"

    print("\n" + "=" * 60)
    print("v3 Sales Predictor — Final Test Metrics")
    print("=" * 60)
    print(f"  test R²           : {test_r2}")
    print(f"  test MAE          : {test_mae:.4f}")
    print(f"  test pred std     : {test_pred_std:.4f}")
    print(f"  baseline test R²  : {baseline_r2}")
    print(f"  verdict           : {verdict}")
    print("=" * 60)
    print(f"\nCheckpoint : {OUTPUT_DIR / 'best_model.pth'}")
    print(f"Metrics    : {REPORT_DIR / 'metrics.json'}")
    print(f"History    : {REPORT_DIR / 'training_history.csv'}")


if __name__ == "__main__":
    main()
