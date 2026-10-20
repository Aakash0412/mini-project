
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
STATE_DIR = ROOT / "data" / "processed" / "state_train_only"
DATA_PATH = ROOT / "data" / "processed" / "products_clean.parquet"

OUTPUT_DIR = ROOT / "models" / "sales_predictor_train_only"
REPORT_DIR = ROOT / "reports" / "sales_predictor_train_only"

TARGET = "Estimated Monthly Sales Growth Rate"
DISCOUNT = "Discount"

SEED = 42
EPOCHS = 60
PATIENCE = 8
BATCH_SIZE = 256
EVAL_BATCH_SIZE = 512
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5


class SalesPredictorMLP(nn.Module):
    def __init__(self, input_dim: int):
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
        self, state: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        if action.ndim == 1:
            action = action.unsqueeze(1)
        return self.forward(torch.cat([state, action], dim=1))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    residual = y_true - y_pred
    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(np.mean(residual**2)))

    denominator = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = (
        float(1.0 - np.sum(residual**2) / denominator)
        if denominator > 0
        else None
    )

    return {
        "n": int(len(y_true)),
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
    }


def main() -> None:
    set_seed(SEED)

    # Refuse to overwrite existing experiment outputs.
    if OUTPUT_DIR.exists() or REPORT_DIR.exists():
        raise FileExistsError(
            "Output directory already exists. Choose a new experiment "
            "directory or inspect existing files before proceeding.\n"
            f"Model output: {OUTPUT_DIR}\n"
            f"Report output: {REPORT_DIR}"
        )

    state_path = STATE_DIR / "state.npy"
    index_path = STATE_DIR / "state_row_indices.npy"

    state = np.load(state_path).astype(np.float32)
    row_indices = np.load(index_path).astype(np.int64)

    df = pd.read_parquet(DATA_PATH).reset_index(drop=True)
    aligned = df.iloc[row_indices].reset_index(drop=True)

    if len(state) != len(aligned):
        raise ValueError("State and dataframe row counts do not match.")
    if not np.isfinite(state).all():
        raise ValueError("State contains non-finite values.")
    if aligned[TARGET].isna().any():
        raise ValueError(f"{TARGET} contains missing values.")
    if aligned[DISCOUNT].isna().any():
        raise ValueError(
            "Discount contains missing values; resolve explicitly "
            "rather than silently imputing them."
        )

    months = aligned["MonthNum"].to_numpy()
    train_mask = np.isin(months, [1, 2, 3, 4, 5, 6])
    val_mask = np.isin(months, [7, 8])
    test_mask = np.isin(months, [9, 10])

    if np.any(train_mask & val_mask) or np.any(train_mask & test_mask):
        raise ValueError("Chronological split masks overlap.")

    targets = aligned[TARGET].to_numpy(dtype=np.float32)
    discounts = aligned[DISCOUNT].to_numpy(dtype=np.float32)

    # Inputs are the 111-D corrected state plus observed discount.
    x = np.concatenate([state, discounts[:, None]], axis=1)
    y = targets[:, None]

    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Training inputs or targets contain non-finite values.")

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SalesPredictorMLP(input_dim=x.shape[1]).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    history = []

    print(f"Device: {device}")
    print(f"State dimension: {state.shape[1]}")
    print(f"Predictor input dimension: {x.shape[1]}")
    print(
        f"Rows — train: {int(train_mask.sum())}, "
        f"validation: {int(val_mask.sum())}, "
        f"test: {int(test_mask.sum())}"
    )
    print(f"Target: {TARGET}")
    print(f"Training target mean/std: "
          f"{targets[train_mask].mean():.4f} / "
          f"{targets[train_mask].std():.4f}")

    def evaluate_loss(loader: DataLoader) -> float:
        model.eval()
        total_loss = 0.0
        total_rows = 0
        with torch.no_grad():
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                prediction = model(batch_x)
                loss = criterion(prediction, batch_y)
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
            prediction = model(batch_x)
            loss = criterion(prediction, batch_y)
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
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train MSE {train_loss:.4f} | "
            f"validation MSE {val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0

            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_dim": x.shape[1],
                    "state_dim": state.shape[1],
                    "seed": SEED,
                    "best_epoch": best_epoch,
                    "best_validation_mse": best_val_loss,
                },
                OUTPUT_DIR / "best_model.pth",
            )
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch}.")
            break

    # Restore best validation checkpoint before final test evaluation.
    checkpoint = torch.load(
        OUTPUT_DIR / "best_model.pth",
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(checkpoint["model_state_dict"])

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

    results = {
        "experiment": "sales_predictor_train_only",
        "target": TARGET,
        "state_dim": int(state.shape[1]),
        "input_dim": int(x.shape[1]),
        "seed": SEED,
        "best_epoch": best_epoch,
        "best_validation_mse": best_val_loss,
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
            "train": regression_metrics(train_actual, train_pred),
            "validation": regression_metrics(val_actual, val_pred),
            "test": regression_metrics(test_actual, test_pred),
        },
        "target_distribution": {
            "train_mean": float(targets[train_mask].mean()),
            "train_std": float(targets[train_mask].std()),
            "train_min": float(targets[train_mask].min()),
            "train_max": float(targets[train_mask].max()),
        },
        "caveat": (
            "Predictor uses existing image and text embeddings, which have "
            "not been regenerated or independently audited in this experiment. "
            "Sales-growth target semantics and possible temporal leakage "
            "must be reviewed before interpreting counterfactual predictions."
        ),
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(REPORT_DIR / "training_history.csv", index=False)
    (REPORT_DIR / "metrics.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )

    print("\nFinal metrics (best validation checkpoint):")
    print(json.dumps(results["metrics"], indent=2))
    print(f"\nBest epoch: {best_epoch}")
    print(f"Checkpoint: {OUTPUT_DIR / 'best_model.pth'}")
    print(f"Metrics: {REPORT_DIR / 'metrics.json'}")
    print(f"History: {REPORT_DIR / 'training_history.csv'}")


if __name__ == "__main__":
    main()
