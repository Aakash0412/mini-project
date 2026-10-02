"""
Phase 14 — Multimodal Ablation Study

Ablation configurations:
    1. Attributes only
    2. Attributes + Text
    3. Attributes + Text + Image

Purpose:
    Measure whether adding available product modalities changes
    sales-prediction performance.

This is an ablation experiment. It does NOT modify the main
111-D sales predictor or any PPO checkpoint.

Current state ordering:
    Image       = 32-D
    Text        = 32-D
    Attributes  = 47-D
    Total       = 111-D

Target:
    Estimated Monthly Sales Growth Rate

Important:
    Sentiment is intentionally excluded because the required
    customer-review text is unavailable.
"""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

SEED = 42

STATE_DIM = 111

IMAGE_DIM = 32
TEXT_DIM = 32
ATTRIBUTE_DIM = 47

DISCOUNT_DIM = 1

TARGET_COLUMN = "Estimated Monthly Sales Growth Rate"
DISCOUNT_COLUMN = "Discount"

EPOCHS = 60
PATIENCE = 8
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5

MAX_GRAD_NORM = 1.0

OUTPUT_DIR = PROJECT_ROOT / "reports" / "ablation"
MODEL_DIR = OUTPUT_DIR / "models"

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "products_clean.parquet"
)

STATE_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "state"
    / "state.npy"
)

STATE_INDEX_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "state"
    / "state_row_indices.npy"
)


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------

def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------

class AblationSalesPredictor(nn.Module):
    """
    Same predictor architecture used by the current project,
    with a configurable input dimension.

    Input:
        state_dim + discount

    Architecture:
        input
          ↓
        256
          ↓
        128
          ↓
         64
          ↓
          1
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


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:

    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    absolute_error = np.abs(y_true - y_pred)

    metrics = {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(
            np.sqrt(
                mean_squared_error(y_true, y_pred)
            )
        ),
        "r2": float(
            r2_score(y_true, y_pred)
        ),
        "median_absolute_error": float(
            np.median(absolute_error)
        ),
        "bias": float(
            np.mean(y_pred - y_true)
        ),
    }

    # Correlation metrics are useful but can be undefined
    # for constant vectors, so handle safely.

    if (
        np.std(y_true) > 0
        and np.std(y_pred) > 0
    ):
        pearson = np.corrcoef(
            y_true,
            y_pred,
        )[0, 1]
    else:
        pearson = np.nan

    true_rank = pd.Series(y_true).rank().to_numpy()
    pred_rank = pd.Series(y_pred).rank().to_numpy()

    if (
        np.std(true_rank) > 0
        and np.std(pred_rank) > 0
    ):
        spearman = np.corrcoef(
            true_rank,
            pred_rank,
        )[0, 1]
    else:
        spearman = np.nan

    metrics["pearson"] = float(pearson)
    metrics["spearman"] = float(spearman)

    return metrics


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------

def load_data():

    print("=" * 72)
    print("PHASE 14 — MULTIMODAL ABLATION STUDY")
    print("=" * 72)

    print()
    print("Loading dataset...")
    df = pd.read_parquet(DATA_PATH)

    print(
        f"Dataset rows: {len(df)}"
    )

    print("Loading final 111-D state...")
    state = np.load(STATE_PATH)

    print(
        f"State shape: {state.shape}"
    )

    source_indices = np.load(
        STATE_INDEX_PATH
    )

    print(
        f"State source-index count: "
        f"{len(source_indices)}"
    )

    # -------------------------------------------------------------
    # Basic validation
    # -------------------------------------------------------------

    if state.ndim != 2:
        raise ValueError(
            f"State must be 2-D. "
            f"Found shape {state.shape}."
        )

    if state.shape[1] != STATE_DIM:
        raise ValueError(
            f"Expected 111-D state. "
            f"Found {state.shape[1]}."
        )

    if len(source_indices) != len(state):
        raise ValueError(
            "State/source-index length mismatch."
        )

    if not np.isfinite(state).all():
        raise ValueError(
            "State contains NaN or Inf values."
        )

    # -------------------------------------------------------------
    # Align dataframe to state rows
    # -------------------------------------------------------------

    aligned = df.iloc[
        source_indices
    ].copy()

    aligned = aligned.reset_index(
        drop=True
    )

    # -------------------------------------------------------------
    # Target
    # -------------------------------------------------------------

    target = (
        aligned[TARGET_COLUMN]
        .fillna(0.0)
        .to_numpy(dtype=np.float32)
    )

    # -------------------------------------------------------------
    # Discount
    # -------------------------------------------------------------

    discount = (
        aligned[DISCOUNT_COLUMN]
        .fillna(0.0)
        .to_numpy(dtype=np.float32)
        .reshape(-1, 1)
    )

    # -------------------------------------------------------------
    # Chronological split
    # -------------------------------------------------------------

    train_mask = (
        aligned["MonthNum"]
        .between(1, 6)
        .to_numpy()
    )

    validation_mask = (
        aligned["MonthNum"]
        .between(7, 8)
        .to_numpy()
    )

    test_mask = (
        aligned["MonthNum"]
        .between(9, 10)
        .to_numpy()
    )

    masks = {
        "train": train_mask,
        "validation": validation_mask,
        "test": test_mask,
    }

    print()
    print("Chronological split:")
    for name, mask in masks.items():
        print(
            f"  {name:<12}: "
            f"{int(mask.sum()):>6} rows"
        )

    return (
        state.astype(np.float32),
        discount,
        target,
        masks,
        source_indices,
    )


# ---------------------------------------------------------------------
# Feature configurations
# ---------------------------------------------------------------------

def build_configurations(state: np.ndarray):
    """
    State ordering:

        Image       0:32
        Text       32:64
        Attributes 64:111
    """

    configurations = {
        "attributes_only": {
            "description": "Attributes only",
            "state": state[:, 64:111],
            "modalities": [
                "attributes"
            ],
        },

        "attributes_text": {
            "description": "Attributes + Text",
            "state": state[:, 32:111],
            "modalities": [
                "attributes",
                "text",
            ],
        },

        "attributes_text_image": {
            "description": "Attributes + Text + Image",
            "state": state[:, :111],
            "modalities": [
                "attributes",
                "text",
                "image",
            ],
        },
    }

    return configurations


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------

def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    input_dim: int,
    model_path: Path,
):

    device = torch.device("cpu")

    model = AblationSalesPredictor(
        input_dim=input_dim
    ).to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=0.5,
        patience=4,
    )

    criterion = nn.MSELoss()

    train_dataset = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train),
    )

    val_dataset = TensorDataset(
        torch.from_numpy(X_val),
        torch.from_numpy(y_val),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0

    history = []

    for epoch in range(1, EPOCHS + 1):

        # ---------------------------------------------------------
        # Training
        # ---------------------------------------------------------

        model.train()

        train_loss_sum = 0.0

        for bx, by in train_loader:

            bx = bx.to(device)
            by = by.to(device)

            optimizer.zero_grad()

            prediction = model(bx)

            loss = criterion(
                prediction,
                by,
            )

            loss.backward()

            nn.utils.clip_grad_norm_(
                model.parameters(),
                MAX_GRAD_NORM,
            )

            optimizer.step()

            train_loss_sum += (
                loss.item()
                * bx.size(0)
            )

        train_loss = (
            train_loss_sum
            / len(train_loader.dataset)
        )

        # ---------------------------------------------------------
        # Validation
        # ---------------------------------------------------------

        model.eval()

        val_loss_sum = 0.0

        with torch.no_grad():

            for bx, by in val_loader:

                bx = bx.to(device)
                by = by.to(device)

                prediction = model(bx)

                loss = criterion(
                    prediction,
                    by,
                )

                val_loss_sum += (
                    loss.item()
                    * bx.size(0)
                )

        val_loss = (
            val_loss_sum
            / len(val_loader.dataset)
        )

        scheduler.step(val_loss)

        history.append(
            {
                "epoch": epoch,
                "train_mse": float(train_loss),
                "val_mse": float(val_loss),
                "learning_rate": float(
                    optimizer.param_groups[0]["lr"]
                ),
            }
        )

        if epoch == 1 or epoch % 5 == 0:
            print(
                f"  Epoch {epoch:3d} | "
                f"train_MSE={train_loss:.6f} | "
                f"val_MSE={val_loss:.6f}"
            )

        # ---------------------------------------------------------
        # Checkpoint
        # ---------------------------------------------------------

        if val_loss < best_val_loss:

            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0

            torch.save(
                model.state_dict(),
                model_path,
            )

        else:

            patience_counter += 1

            if patience_counter >= PATIENCE:

                print(
                    f"  Early stopping at epoch "
                    f"{epoch}."
                )

                break

    # -------------------------------------------------------------
    # Reload best checkpoint
    # -------------------------------------------------------------

    model.load_state_dict(
        torch.load(
            model_path,
            map_location="cpu",
            weights_only=True,
        )
    )

    model.eval()

    return (
        model,
        history,
        best_epoch,
        best_val_loss,
    )


# ---------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------

def predict(
    model: nn.Module,
    X: np.ndarray,
) -> np.ndarray:

    model.eval()

    dataset = TensorDataset(
        torch.from_numpy(X)
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    predictions = []

    with torch.no_grad():

        for (bx,) in loader:

            output = model(
                bx
            )

            predictions.append(
                output.cpu().numpy()
            )

    return np.concatenate(
        predictions
    ).astype(np.float32)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    set_seed(SEED)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        state,
        discount,
        target,
        masks,
        source_indices,
    ) = load_data()

    configurations = build_configurations(
        state
    )

    all_results = []

    experiment_metadata = {
        "phase": 14,
        "description": (
            "Multimodal sales-predictor ablation"
        ),
        "target": TARGET_COLUMN,
        "seed": SEED,
        "epochs": EPOCHS,
        "patience": PATIENCE,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "state_order": [
            "image_32",
            "text_32",
            "attributes_47",
        ],
        "sentiment": (
            "excluded: review text unavailable"
        ),
        "configurations": {},
    }

    # -------------------------------------------------------------
    # Run each ablation
    # -------------------------------------------------------------

    for name, config in configurations.items():

        print()
        print("=" * 72)
        print(
            f"ABLATION: {config['description']}"
        )
        print("=" * 72)

        ablation_state = config["state"]

        state_dim = ablation_state.shape[1]

        input_dim = (
            state_dim
            + DISCOUNT_DIM
        )

        print(
            f"State dimension: {state_dim}"
        )

        print(
            f"Predictor input dimension: "
            f"{input_dim}"
        )

        print(
            "Modalities: "
            + ", ".join(
                config["modalities"]
            )
        )

        # ---------------------------------------------------------
        # Build predictor input
        # ---------------------------------------------------------

        X = np.concatenate(
            [
                ablation_state,
                discount,
            ],
            axis=1,
        ).astype(np.float32)

        if X.shape[1] != input_dim:
            raise RuntimeError(
                "Unexpected predictor input "
                f"shape: {X.shape}"
            )

        # ---------------------------------------------------------
        # Split
        # ---------------------------------------------------------

        train_mask = masks["train"]
        val_mask = masks["validation"]
        test_mask = masks["test"]

        X_train = X[train_mask]
        y_train = target[train_mask]

        X_val = X[val_mask]
        y_val = target[val_mask]

        X_test = X[test_mask]
        y_test = target[test_mask]

        print(
            f"Train: {X_train.shape}"
        )
        print(
            f"Validation: {X_val.shape}"
        )
        print(
            f"Test: {X_test.shape}"
        )

        # ---------------------------------------------------------
        # Train
        # ---------------------------------------------------------

        model_path = (
            MODEL_DIR
            / f"{name}_best_model.pth"
        )

        model, history, best_epoch, best_val_loss = (
            train_model(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                input_dim=input_dim,
                model_path=model_path,
            )
        )

        # ---------------------------------------------------------
        # Predictions
        # ---------------------------------------------------------

        train_pred = predict(
            model,
            X_train,
        )

        val_pred = predict(
            model,
            X_val,
        )

        test_pred = predict(
            model,
            X_test,
        )

        # ---------------------------------------------------------
        # Metrics
        # ---------------------------------------------------------

        split_predictions = {
            "train": (
                y_train,
                train_pred,
            ),
            "validation": (
                y_val,
                val_pred,
            ),
            "test": (
                y_test,
                test_pred,
            ),
        }

        metrics_by_split = {}

        for split_name, (
            y_true,
            y_pred,
        ) in split_predictions.items():

            metrics = calculate_metrics(
                y_true,
                y_pred,
            )

            metrics_by_split[
                split_name
            ] = metrics

            print()
            print(
                f"{split_name.upper()} RESULTS"
            )

            print(
                f"  MAE:                 "
                f"{metrics['mae']:.6f}"
            )

            print(
                f"  RMSE:                "
                f"{metrics['rmse']:.6f}"
            )

            print(
                f"  R²:                  "
                f"{metrics['r2']:.6f}"
            )

            print(
                f"  Median AE:           "
                f"{metrics['median_absolute_error']:.6f}"
            )

            print(
                f"  Bias:                "
                f"{metrics['bias']:.6f}"
            )

            print(
                f"  Pearson:             "
                f"{metrics['pearson']:.6f}"
            )

            print(
                f"  Spearman:            "
                f"{metrics['spearman']:.6f}"
            )

        # ---------------------------------------------------------
        # Save history
        # ---------------------------------------------------------

        history_path = (
            OUTPUT_DIR
            / f"{name}_training_history.csv"
        )

        pd.DataFrame(
            history
        ).to_csv(
            history_path,
            index=False,
        )

        # ---------------------------------------------------------
        # Save configuration/results
        # ---------------------------------------------------------

        config_result = {
            "name": name,
            "description": config[
                "description"
            ],
            "modalities": config[
                "modalities"
            ],
            "state_dim": state_dim,
            "input_dim": input_dim,
            "best_epoch": best_epoch,
            "best_validation_mse": float(
                best_val_loss
            ),
            "metrics": metrics_by_split,
            "model_path": str(
                model_path
            ),
            "history_path": str(
                history_path
            ),
        }

        experiment_metadata[
            "configurations"
        ][name] = config_result

        all_results.append(
            {
                "configuration": name,
                "description": config[
                    "description"
                ],
                "state_dim": state_dim,
                "input_dim": input_dim,
                "best_epoch": best_epoch,

                "train_mae":
                    metrics_by_split[
                        "train"
                    ]["mae"],

                "train_rmse":
                    metrics_by_split[
                        "train"
                    ]["rmse"],

                "train_r2":
                    metrics_by_split[
                        "train"
                    ]["r2"],

                "validation_mae":
                    metrics_by_split[
                        "validation"
                    ]["mae"],

                "validation_rmse":
                    metrics_by_split[
                        "validation"
                    ]["rmse"],

                "validation_r2":
                    metrics_by_split[
                        "validation"
                    ]["r2"],

                "test_mae":
                    metrics_by_split[
                        "test"
                    ]["mae"],

                "test_rmse":
                    metrics_by_split[
                        "test"
                    ]["rmse"],

                "test_r2":
                    metrics_by_split[
                        "test"
                    ]["r2"],

                "test_median_ae":
                    metrics_by_split[
                        "test"
                    ][
                        "median_absolute_error"
                    ],

                "test_bias":
                    metrics_by_split[
                        "test"
                    ]["bias"],

                "test_pearson":
                    metrics_by_split[
                        "test"
                    ]["pearson"],

                "test_spearman":
                    metrics_by_split[
                        "test"
                    ]["spearman"],
            }
        )

    # -------------------------------------------------------------
    # Save summary
    # -------------------------------------------------------------

    summary_df = pd.DataFrame(
        all_results
    )

    summary_path = (
        OUTPUT_DIR
        / "ablation_results.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    metadata_path = (
        OUTPUT_DIR
        / "ablation_metadata.json"
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            experiment_metadata,
            f,
            indent=2,
        )

    # -------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------

    print()
    print("=" * 72)
    print("PHASE 14 ABLATION SUMMARY")
    print("=" * 72)

    print()

    print(
        summary_df[
            [
                "configuration",
                "state_dim",
                "input_dim",
                "test_mae",
                "test_rmse",
                "test_r2",
                "test_median_ae",
                "test_pearson",
                "test_spearman",
            ]
        ].to_string(
            index=False
        )
    )

    # -------------------------------------------------------------
    # Required checks
    # -------------------------------------------------------------

    print()
    print("VALIDATION")

    state_valid = (
        state.shape[1]
        == STATE_DIM
    )

    finite_state = np.isfinite(
        state
    ).all()

    source_alignment_valid = (
        len(source_indices)
        == len(state)
    )

    summary_finite = np.isfinite(
        summary_df.select_dtypes(
            include=[np.number]
        ).to_numpy()
    ).all()

    dimensions_valid = (
        configurations[
            "attributes_only"
        ]["state"].shape[1] == 47
        and
        configurations[
            "attributes_text"
        ]["state"].shape[1] == 79
        and
        configurations[
            "attributes_text_image"
        ]["state"].shape[1] == 111
    )

    print(
        f"111-D state:           "
        f"{'PASS' if state_valid else 'FAIL'}"
    )

    print(
        f"Finite state:          "
        f"{'PASS' if finite_state else 'FAIL'}"
    )

    print(
        f"Source alignment:      "
        f"{'PASS' if source_alignment_valid else 'FAIL'}"
    )

    print(
        f"Ablation dimensions:   "
        f"{'PASS' if dimensions_valid else 'FAIL'}"
    )

    print(
        f"Finite results:        "
        f"{'PASS' if summary_finite else 'FAIL'}"
    )

    if not all(
        [
            state_valid,
            finite_state,
            source_alignment_valid,
            dimensions_valid,
            summary_finite,
        ]
    ):
        raise RuntimeError(
            "Phase 14 validation failed."
        )

    print()
    print(
        "Phase 14 ablation validation: PASS"
    )

    print()
    print(
        f"Saved summary: {summary_path}"
    )

    print(
        f"Saved metadata: {metadata_path}"
    )

    print(
        f"Saved models: {MODEL_DIR}"
    )


if __name__ == "__main__":
    main()