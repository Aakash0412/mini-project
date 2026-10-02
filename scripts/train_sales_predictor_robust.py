"""
Experimental Sales Predictor — Signed-Log Target

IMPORTANT:
- Does NOT replace models/sales_predictor/
- Current raw-target predictor remains the project baseline.
- This experiment changes only the target representation:

    y' = sign(y) * log1p(abs(y))

- Metrics are converted back to the original target units.
- This is an experimental diagnostic/adaptation, NOT an exact
  base-paper reproduction.
"""

import json
import os
import random
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


SEED = 42
OUT_DIR = "models/sales_predictor_robust"

DISCOUNT_BINS = [0, 5, 10, 15, 20, 25, 30, 35, 40]


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def signed_log1p(y):
    y = np.asarray(y, dtype=np.float32)
    return np.sign(y) * np.log1p(np.abs(y))


def signed_expm1(z):
    z = np.asarray(z, dtype=np.float32)
    return np.sign(z) * np.expm1(np.abs(z))


class SalesPredictorMLP(nn.Module):

    def __init__(self, input_dim):
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

    def forward(self, x):
        return self.net(x).squeeze(-1)


def regression_metrics(actual, pred):

    residual = actual - pred

    return {
        "MAE": float(mean_absolute_error(actual, pred)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, pred))),
        "R2": float(r2_score(actual, pred)),
        "median_absolute_error": float(
            np.median(np.abs(residual))
        ),
        "bias_actual_minus_pred": float(
            np.mean(residual)
        ),
        "pearson": float(
            np.corrcoef(actual, pred)[0, 1]
        ),
        "spearman": float(
            pd.Series(actual).corr(
                pd.Series(pred),
                method="spearman"
            )
        ),
    }


def evaluate(model, loader, device):

    model.eval()

    pred_transformed = []
    actual_transformed = []

    with torch.no_grad():

        for bx, by in loader:

            prediction = model(
                bx.to(device)
            ).cpu().numpy()

            pred_transformed.append(prediction)
            actual_transformed.append(
                by.numpy()
            )

    pred_transformed = np.concatenate(
        pred_transformed
    )

    actual_transformed = np.concatenate(
        actual_transformed
    )

    # Convert back to original percentage units.
    predictions = signed_expm1(
        pred_transformed
    )

    actual = signed_expm1(
        actual_transformed
    )

    return predictions, actual


def main():

    seed_everything()

    print("=" * 70)
    print("EXPERIMENTAL SALES PREDICTOR")
    print("SIGNED-LOG TARGET")
    print("=" * 70)

    print(
        "\nIMPORTANT:"
        "\nThe base predictor will NOT be modified."
    )

    device = torch.device("cpu")

    print(f"\nDevice: {device}")

    # ----------------------------------------------------------
    # Load state
    # ----------------------------------------------------------

    state = np.load(
        "data/processed/state/state.npy"
    ).astype(np.float32)

    row_indices = np.load(
        "data/processed/state/state_row_indices.npy"
    ).astype(np.int64)

    df = pd.read_parquet(
        "data/processed/products_clean.parquet"
    ).reset_index(drop=True)

    if len(state) != len(row_indices):
        raise ValueError(
            "state.npy and state_row_indices.npy "
            "length mismatch"
        )

    if state.shape[1] != 111:
        raise ValueError(
            f"Expected 111-D state, "
            f"got {state.shape[1]}"
        )

    if not np.isfinite(state).all():
        raise ValueError(
            "State contains NaN or Inf"
        )

    # ----------------------------------------------------------
    # Preserve source-row alignment
    # ----------------------------------------------------------

    if (
        row_indices.min() < 0
        or row_indices.max() >= len(df)
    ):
        raise ValueError(
            "Invalid source-row indices"
        )

    aligned_df = (
        df.iloc[row_indices]
        .reset_index(drop=True)
    )

    # ----------------------------------------------------------
    # Target
    # ----------------------------------------------------------

    target_column = (
        "Estimated Monthly Sales Growth Rate"
    )

    y_raw = (
        aligned_df[target_column]
        .fillna(0.0)
        .astype(np.float32)
        .to_numpy()
    )

    y = signed_log1p(y_raw)

    # ----------------------------------------------------------
    # Discount
    # ----------------------------------------------------------

    discount = (
        aligned_df["Discount"]
        .fillna(0.0)
        .astype(np.float32)
        .to_numpy()
        .reshape(-1, 1)
    )

    # ----------------------------------------------------------
    # Predictor input
    # ----------------------------------------------------------

    X = np.concatenate(
        [state, discount],
        axis=1
    ).astype(np.float32)

    if X.shape[1] != 112:
        raise ValueError(
            f"Expected 112-D input, "
            f"got {X.shape[1]}"
        )

    # ----------------------------------------------------------
    # Chronological split
    # ----------------------------------------------------------

    train_mask = aligned_df[
        "MonthNum"
    ].isin(
        [1, 2, 3, 4, 5, 6]
    ).to_numpy()

    val_mask = aligned_df[
        "MonthNum"
    ].isin(
        [7, 8]
    ).to_numpy()

    test_mask = aligned_df[
        "MonthNum"
    ].isin(
        [9, 10]
    ).to_numpy()

    X_train = X[train_mask]
    y_train = y[train_mask]

    X_val = X[val_mask]
    y_val = y[val_mask]

    X_test = X[test_mask]
    y_test = y[test_mask]

    print(
        f"\nAligned samples: {len(X)}"
    )

    print(
        f"Train: {len(X_train)}"
        f" | Val: {len(X_val)}"
        f" | Test: {len(X_test)}"
    )

    print(
        f"Input: {X.shape[1]}-D "
        "(111-D state + 1-D discount)"
    )

    print(
        "Target transform:"
        " sign(y) * log1p(abs(y))"
    )

    # ----------------------------------------------------------
    # DataLoaders
    # ----------------------------------------------------------

    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(X_train),
            torch.from_numpy(y_train)
        ),
        batch_size=256,
        shuffle=True
    )

    val_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(X_val),
            torch.from_numpy(y_val)
        ),
        batch_size=512,
        shuffle=False
    )

    test_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(X_test),
            torch.from_numpy(y_test)
        ),
        batch_size=512,
        shuffle=False
    )

    # ----------------------------------------------------------
    # Model
    # ----------------------------------------------------------

    model = SalesPredictorMLP(
        input_dim=112
    ).to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-5
    )

    scheduler = (
        optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=0.5,
            patience=4
        )
    )

    criterion = nn.MSELoss()

    os.makedirs(
        OUT_DIR,
        exist_ok=True
    )

    best_model_path = os.path.join(
        OUT_DIR,
        "best_model.pth"
    )

    best_val_loss = float("inf")
    patience_counter = 0

    train_losses = []
    val_losses = []

    # ----------------------------------------------------------
    # Training
    # ----------------------------------------------------------

    print("\nTraining...")

    for epoch in range(1, 61):

        model.train()

        running_loss = 0.0

        for bx, by in train_loader:

            optimizer.zero_grad()

            prediction = model(bx)

            loss = criterion(
                prediction,
                by
            )

            loss.backward()

            nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

            running_loss += (
                loss.item()
                * bx.size(0)
            )

        train_loss = (
            running_loss
            / len(train_loader.dataset)
        )

        # ------------------------------------------------------
        # Validation
        # ------------------------------------------------------

        model.eval()

        validation_loss = 0.0

        with torch.no_grad():

            for bx, by in val_loader:

                prediction = model(bx)

                validation_loss += (
                    criterion(
                        prediction,
                        by
                    ).item()
                    * bx.size(0)
                )

        val_loss = (
            validation_loss
            / len(val_loader.dataset)
        )

        scheduler.step(val_loss)

        train_losses.append(
            float(train_loss)
        )

        val_losses.append(
            float(val_loss)
        )

        if (
            epoch == 1
            or epoch % 5 == 0
        ):

            print(
                f"Epoch {epoch:3d} | "
                f"Train MSE={train_loss:.6f} | "
                f"Val MSE={val_loss:.6f}"
            )

        # ------------------------------------------------------
        # Checkpoint
        # ------------------------------------------------------

        if val_loss < best_val_loss:

            best_val_loss = val_loss
            patience_counter = 0

            torch.save(
                model.state_dict(),
                best_model_path
            )

        else:

            patience_counter += 1

            if patience_counter >= 8:

                print(
                    f"Early stopping at epoch "
                    f"{epoch}"
                )

                break

    # ----------------------------------------------------------
    # Load best model
    # ----------------------------------------------------------

    model.load_state_dict(
        torch.load(
            best_model_path,
            map_location=device
        )
    )

    model.eval()

    # ----------------------------------------------------------
    # Evaluation in ORIGINAL units
    # ----------------------------------------------------------

    train_eval_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(X_train),
            torch.from_numpy(y_train)
        ),
        batch_size=512,
        shuffle=False
    )

    train_pred, train_actual = evaluate(
        model,
        train_eval_loader,
        device
    )

    val_pred, val_actual = evaluate(
        model,
        val_loader,
        device
    )

    test_pred, test_actual = evaluate(
        model,
        test_loader,
        device
    )

    train_metrics = regression_metrics(
        train_actual,
        train_pred
    )

    val_metrics = regression_metrics(
        val_actual,
        val_pred
    )

    test_metrics = regression_metrics(
        test_actual,
        test_pred
    )

    # ----------------------------------------------------------
    # Print results
    # ----------------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "RESULTS — ORIGINAL TARGET UNITS"
    )

    print("=" * 70)

    for name, metrics in [
        ("TRAIN", train_metrics),
        ("VALIDATION", val_metrics),
        ("TEST", test_metrics),
    ]:

        print(
            f"\n{name}"
        )

        print(
            f"MAE      : "
            f"{metrics['MAE']:.4f}"
        )

        print(
            f"RMSE     : "
            f"{metrics['RMSE']:.4f}"
        )

        print(
            f"R²       : "
            f"{metrics['R2']:.4f}"
        )

        print(
            f"Median AE: "
            f"{metrics['median_absolute_error']:.4f}"
        )

        print(
            f"Bias     : "
            f"{metrics['bias_actual_minus_pred']:.4f}"
        )

    # ----------------------------------------------------------
    # Save metrics
    # ----------------------------------------------------------

    metrics = {
        "experiment":
            "signed_log_target",

        "baseline_preserved":
            True,

        "target":
            target_column,

        "target_transform":
            "sign(y) * log1p(abs(y))",

        "inverse_transform":
            "sign(z) * expm1(abs(z))",

        "input_dim":
            112,

        "state_dim":
            111,

        "action_dim":
            1,

        "discount_bins":
            DISCOUNT_BINS,

        "architecture":
            "MLP: 112->256->BN->128->BN->64->1",

        "optimizer":
            "Adam",

        "learning_rate":
            1e-3,

        "weight_decay":
            1e-5,

        "loss":
            "MSELoss on transformed target",

        "batch_size":
            256,

        "epochs_run":
            len(train_losses),

        "seed":
            SEED,

        "device":
            str(device),

        "samples": {
            "total":
                int(len(X)),

            "train":
                int(len(X_train)),

            "validation":
                int(len(X_val)),

            "test":
                int(len(X_test)),
        },

        "metrics_original_target_units": {

            "train":
                train_metrics,

            "validation":
                val_metrics,

            "test":
                test_metrics,
        },

        "training_curve": {

            "train_transformed_mse":
                train_losses,

            "validation_transformed_mse":
                val_losses,
        }
    }

    with open(
        os.path.join(
            OUT_DIR,
            "metrics.json"
        ),
        "w"
    ) as f:

        json.dump(
            metrics,
            f,
            indent=4
        )

    with open(
        os.path.join(
            OUT_DIR,
            "config.json"
        ),
        "w"
    ) as f:

        json.dump(
            {
                "experiment":
                    "signed_log_target",

                "input_dim":
                    112,

                "state_dim":
                    111,

                "action_dim":
                    1,

                "discount_bins":
                    DISCOUNT_BINS,

                "target":
                    target_column,

                "target_transform":
                    "sign(y) * log1p(abs(y))",

                "note":
                    "Experimental variant. "
                    "Do not present as the "
                    "base-paper raw-target "
                    "implementation."
            },
            f,
            indent=4
        )

    print(
        "\nSaved experimental model:"
    )

    print(
        f"  {OUT_DIR}/best_model.pth"
    )

    print(
        f"  {OUT_DIR}/config.json"
    )

    print(
        f"  {OUT_DIR}/metrics.json"
    )

    print(
        "\nBASELINE PRESERVED:"
        "\n  models/sales_predictor/"
    )


if __name__ == "__main__":
    main()