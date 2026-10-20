"""Sales predictor, target clipped at train-time quantiles.

Hypothesis: the baseline predictor collapses to predict-the-mean because
the target has std/mean = 38 and 14% of |g| > 100%. Under MSE, the model
ignores inputs and predicts the mean.

Fix: clip the target to [q_lo, q_hi] percentiles DURING TRAINING ONLY.
Metrics are evaluated on the raw, unclipped target. Units and sign are
preserved (unlike log-winsor), so the model still fits the same function
shape.

Everything else (state, split, architecture, seed) matches the baseline.
"""

from __future__ import annotations
import json, random, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
STATE_DIR = ROOT / "data/processed/state_train_only"
DATA_PATH = ROOT / "data/processed/products_clean.parquet"
OUTPUT_DIR = ROOT / "models/sales_predictor_clip"
REPORT_DIR = ROOT / "reports/sales_predictor_clip"

TARGET = "Estimated Monthly Sales Growth Rate"
DISCOUNT = "Discount"
SEED, EPOCHS, PATIENCE = 42, 100, 12
BATCH_SIZE, EVAL_BATCH_SIZE = 256, 512
LR, WEIGHT_DECAY = 1e-3, 1e-5
CLIP_LO, CLIP_HI = 1, 99


class SalesPredictorMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1),
        )
    def forward(self, x): return self.network(x)
    def predict(self, s, a):
        if a.ndim == 1: a = a.unsqueeze(1)
        return self.forward(torch.cat([s, a], dim=1))


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)


def metrics(y, p):
    r = y - p
    mae = float(np.mean(np.abs(r)))
    rmse = float(np.sqrt(np.mean(r**2)))
    den = float(np.sum((y - y.mean())**2))
    r2 = float(1 - np.sum(r**2)/den) if den > 0 else None
    return {"n": int(len(y)), "mae": mae, "rmse": rmse, "r2": r2}


def main():
    set_seed(SEED)
    torch.set_num_threads(max(1, (__import__("os").cpu_count() or 2) // 2))

    if OUTPUT_DIR.exists() or REPORT_DIR.exists():
        raise FileExistsError(f"Refuse to overwrite: {OUTPUT_DIR} or {REPORT_DIR}")

    state = np.load(STATE_DIR / "state.npy").astype(np.float32)
    idx = np.load(STATE_DIR / "state_row_indices.npy").astype(np.int64)
    df = pd.read_parquet(DATA_PATH).reset_index(drop=True).iloc[idx].reset_index(drop=True)
    assert len(state) == len(df)

    months = df["MonthNum"].to_numpy()
    targets = df[TARGET].to_numpy(dtype=np.float32)
    discounts = df[DISCOUNT].to_numpy(dtype=np.float32)
    assert np.isfinite(targets).all() and np.isfinite(discounts).all()

    train = np.isin(months, [1, 2, 3, 4, 5, 6])
    val   = np.isin(months, [7, 8])
    test  = np.isin(months, [9, 10])

    # Clip bounds fit on TRAIN ONLY, in target's native units.
    lo, hi = np.percentile(targets[train], [CLIP_LO, CLIP_HI])
    targets_train_clipped = np.clip(targets, lo, hi).astype(np.float32)
    print(f"train target clip bounds: [{lo:.4f}, {hi:.4f}]")
    print(f"train target original  : min={targets[train].min():.2f} max={targets[train].max():.2f} std={targets[train].std():.2f}")

    x = np.concatenate([state, discounts[:, None]], axis=1).astype(np.float32)
    assert x.shape[1] == 112

    def loader(mask, shuffle, clip_targets):
        y = targets_train_clipped[mask] if clip_targets else targets[mask]
        ds = TensorDataset(torch.from_numpy(x[mask]), torch.from_numpy(y)[:, None])
        return DataLoader(ds, batch_size=BATCH_SIZE if shuffle else EVAL_BATCH_SIZE,
                          shuffle=shuffle, drop_last=False)

    # Train/val use clipped targets; test metrics use RAW targets.
    tl = loader(train, True,  True)
    vl = loader(val,   False, True)
    tl_raw = loader(train, False, False)  # for raw-target metrics
    vl_raw = loader(val,   False, False)
    te_raw = loader(test,  False, False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    model = SalesPredictorMLP(x.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3)
    crit = nn.MSELoss()
    best, best_epoch, patience = float("inf"), 0, 0

    def eval_loss(l):
        model.eval(); tot, n = 0.0, 0
        with torch.no_grad():
            for bx, by in l:
                bx, by = bx.to(device), by.to(device)
                tot += crit(model(bx), by).item() * len(bx); n += len(bx)
        return tot / n

    for ep in range(1, EPOCHS + 1):
        model.train(); tot, n = 0.0, 0
        for bx, by in tl:
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(bx), by)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(bx); n += len(bx)
        tr = tot / n; va = eval_loss(vl)
        sched.step(va)
        print(f"ep {ep:03d} | train {tr:.4f} | val {va:.4f}")
        if va < best:
            best, best_epoch, patience = va, ep, 0
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(),
                "input_dim": 112, "state_dim": 111,
                "seed": SEED, "best_epoch": best_epoch,
                "best_validation_mse": best,
                "clip_lo": float(lo), "clip_hi": float(hi),
                "target_transform": "clip_1_99_train_only",
            }, OUTPUT_DIR / "best_model.pth")
        else:
            patience += 1
        if patience >= PATIENCE:
            print(f"early stop at ep {ep}"); break

    ck = torch.load(OUTPUT_DIR / "best_model.pth", map_location=device, weights_only=True)
    model.load_state_dict(ck["model_state_dict"])

    def predict(l):
        model.eval(); preds, actuals = [], []
        with torch.no_grad():
            for bx, by in l:
                preds.extend(model(bx.to(device)).cpu().numpy().reshape(-1).tolist())
                actuals.extend(by.numpy().reshape(-1).tolist())
        return np.asarray(actuals), np.asarray(preds)

    tr_a, tr_p = predict(tl_raw)
    va_a, va_p = predict(vl_raw)
    te_a, te_p = predict(te_raw)

    results = {
        "experiment": "sales_predictor_clip",
        "target": TARGET,
        "state_dim": 111, "input_dim": 112,
        "target_transform": "clip to [q_lo,q_hi] fit on train, train-time only; metrics on raw",
        "clip_lo_percentile": CLIP_LO, "clip_hi_percentile": CLIP_HI,
        "clip_bounds_raw_units": {"lo": float(lo), "hi": float(hi)},
        "seed": SEED, "best_epoch": best_epoch, "best_validation_mse": best,
        "split_months": {"train": [1,2,3,4,5,6], "validation": [7,8], "test": [9,10]},
        "row_counts": {"train": int(train.sum()), "val": int(val.sum()), "test": int(test.sum())},
        "metrics": {
            "train": metrics(tr_a, tr_p),
            "validation": metrics(va_a, va_p),
            "test": metrics(te_a, te_p),
        },
        "prediction_std_test": float(np.std(te_p)),
        "target_distribution": {
            "train_mean": float(targets[train].mean()),
            "train_std": float(targets[train].std()),
            "train_min": float(targets[train].min()),
            "train_max": float(targets[train].max()),
            "frac_abs_gt_100": float((np.abs(targets[train]) > 100).mean()),
        },
        "comparison_to_baseline": {
            "baseline_test_r2": 0.02497,
            "baseline_prediction_std": 61.07,
            "baseline_source": "reports/sales_predictor_train_only/metrics.json",
        },
        "caveat": "Target clipped at train-time quantiles. Metrics on raw, unclipped target.",
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    r2 = results["metrics"]["test"]["r2"]
    std = results["prediction_std_test"]
    print("\n" + "=" * 60)
    print("Final Test Metrics (evaluated on RAW, unclipped target)")
    print("=" * 60)
    print(f"  test R^2        : {r2:.6f}")
    print(f"  test MAE        : {results['metrics']['test']['mae']:.4f}")
    print(f"  test pred std   : {std:.4f}")
    print(f"  baseline R^2    : 0.025")
    print(f"  baseline predstd: 61.07")
    if r2 > 0.10: verdict = "IMPROVED"
    elif r2 > 0.025: verdict = "MARGINAL"
    else: verdict = "NO IMPROVEMENT"
    print(f"  verdict         : {verdict}")
    print(f"checkpoint      : {OUTPUT_DIR / 'best_model.pth'}")
    print(f"metrics         : {REPORT_DIR / 'metrics.json'}")


if __name__ == "__main__":
    main()