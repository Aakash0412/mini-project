"""Retrain sales predictor with log-winsorized target.

Target transform (fit on TRAIN only):
    z = clip(sign(g) * log1p(|g|), q_1, q_99)
Inverse at inference:
    g = sign(z) * expm1(|z|)

Architecture, split, and input contract match
train_sales_predictor_train_only.py, so the resulting checkpoint is a
drop-in replacement for CounterfactualSimulator.
"""

from __future__ import annotations
import json, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
STATE_DIR = ROOT / "data/processed/state_train_only"
DATA_PATH = ROOT / "data/processed/products_clean.parquet"
OUTPUT_DIR = ROOT / "models/sales_predictor_log_winsor"
REPORT_DIR = ROOT / "reports/sales_predictor_log_winsor"

TARGET = "Estimated Monthly Sales Growth Rate"
DISCOUNT = "Discount"
SEED, EPOCHS, PATIENCE = 42, 80, 12
BATCH_SIZE, EVAL_BATCH_SIZE = 256, 512
LR, WEIGHT_DECAY = 1e-3, 1e-5
WINSOR_LO, WINSOR_HI = 1, 99


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


def transform(g, lo, hi):
    z = np.sign(g) * np.log1p(np.abs(g))
    return np.clip(z, lo, hi).astype(np.float32)


def inverse(z):
    return np.sign(z) * np.expm1(np.abs(z))


def main():
    set_seed(SEED)
    if OUTPUT_DIR.exists() or REPORT_DIR.exists():
        raise FileExistsError(f"Refuse to overwrite: {OUTPUT_DIR} or {REPORT_DIR}")

    state = np.load(STATE_DIR / "state.npy").astype(np.float32)
    idx   = np.load(STATE_DIR / "state_row_indices.npy").astype(np.int64)
    df = pd.read_parquet(DATA_PATH).reset_index(drop=True).iloc[idx].reset_index(drop=True)
    if len(state) != len(df): raise ValueError("row mismatch")

    months    = df["MonthNum"].to_numpy()
    targets   = df[TARGET].to_numpy(dtype=np.float32)
    discounts = df[DISCOUNT].to_numpy(dtype=np.float32)
    train = np.isin(months, [1,2,3,4,5,6])
    val   = np.isin(months, [7,8])
    test  = np.isin(months, [9,10])

    z_train = np.sign(targets[train]) * np.log1p(np.abs(targets[train]))
    lo, hi = np.percentile(z_train, [WINSOR_LO, WINSOR_HI])
    print(f"winsor z bounds: [{lo:.4f}, {hi:.4f}]")

    x = np.concatenate([state, discounts[:, None]], axis=1).astype(np.float32)
    z_all = transform(targets, lo, hi)

    def loader(mask, shuffle):
        ds = TensorDataset(torch.from_numpy(x[mask]), torch.from_numpy(z_all[mask])[:, None])
        return DataLoader(ds, batch_size=BATCH_SIZE if shuffle else EVAL_BATCH_SIZE,
                          shuffle=shuffle, drop_last=False)

    tl, vl, te = loader(train, True), loader(val, False), loader(test, False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    model = SalesPredictorMLP(x.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3)
    crit = nn.MSELoss()
    best, best_epoch, patience = float("inf"), 0, 0
    history = []

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
        tr_loss = tot / n; va_loss = eval_loss(vl)
        sched.step(va_loss)
        history.append({"epoch": ep, "train_mse_z": tr_loss,
                        "val_mse_z": va_loss, "lr": opt.param_groups[0]["lr"]})
        print(f"ep {ep:02d} | train_z {tr_loss:.4f} | val_z {va_loss:.4f}")
        if va_loss < best:
            best, best_epoch, patience = va_loss, ep, 0
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            torch.save({"model_state_dict": model.state_dict(),
                        "input_dim": x.shape[1], "state_dim": state.shape[1],
                        "seed": SEED, "best_epoch": best_epoch,
                        "best_validation_mse_z": best,
                        "winsor_lo": float(lo), "winsor_hi": float(hi),
                        "target_transform": "sign_log1p_winsor_z"},
                       OUTPUT_DIR / "best_model.pth")
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

    tr_a, tr_p = predict(tl); va_a, va_p = predict(vl); te_a, te_p = predict(te)

    results = {
        "experiment": "sales_predictor_log_winsor",
        "target": TARGET,
        "state_dim": int(state.shape[1]),
        "input_dim": int(x.shape[1]),
        "target_transform": "sign*log1p, winsor 1/99 fit on train",
        "winsor_z_lo": float(lo), "winsor_z_hi": float(hi),
        "seed": SEED, "best_epoch": best_epoch,
        "best_validation_mse_z": best,
        "split_months": {"train":[1,2,3,4,5,6], "validation":[7,8], "test":[9,10]},
        "row_counts": {"train":int(train.sum()), "validation":int(val.sum()),
                       "test":int(test.sum())},
        "metrics_z_space": {
            "train": metrics(tr_a, tr_p),
            "validation": metrics(va_a, va_p),
            "test": metrics(te_a, te_p),
        },
        "metrics_original_space": {
            "train": metrics(inverse(tr_a), inverse(tr_p)),
            "validation": metrics(inverse(va_a), inverse(va_p)),
            "test": metrics(inverse(te_a), inverse(te_p)),
        },
        "prediction_std_test": float(np.std(inverse(te_p))),
        "caveat": "Compare prediction_std_test to baseline ~61 pp. Lower is better.",
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(REPORT_DIR / "training_history.csv", index=False)
    (REPORT_DIR / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\nORIGINAL-SPACE TEST METRICS:")
    print(json.dumps(results["metrics_original_space"]["test"], indent=2))
    print(f"prediction Std on test = {results['prediction_std_test']:.4f}")
    print(f"(baseline Std was ~61; lower is better)")
    print(f"checkpoint: {OUTPUT_DIR/'best_model.pth'}")
    print(f"metrics:    {REPORT_DIR/'metrics.json'}")


if __name__ == "__main__":
    main()