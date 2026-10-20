"""Recompute robustness_v2.csv from existing v2 CSVs with the /100 fix.
Uses chosen_PPO_v2.csv as the source of truth for baseline, sales, price,
gmr so no re-derivation from raw parquet is needed."""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.paper_metrics import (
    delta_margin, delta_quantity, paper_reward, reward_retention,
)

OUT = ROOT / "reports/base_paper_reproduction_v2"

cf = pd.read_csv(OUT / "all_action_counterfactuals_v2.csv")
ppo = (pd.read_csv(OUT / "chosen_PPO_v2.csv")
       .sort_values("test_position").reset_index(drop=True))

# Reconstruct q_prev, m_prev, price, gmr, g_hat from the CSVs.
# q_prev = last_monthly_sales. We can recover it because
# estimated_sales = q_prev * (1 + g_hat/100) clipped at 0.
# For any row with g_hat > -100, q_prev = estimated_sales / (1 + g_hat/100).
g_hat = ppo["predicted_growth_percent"].to_numpy(dtype=np.float64)
q_t   = ppo["estimated_sales"].to_numpy(dtype=np.float64)
m_t   = ppo["gross_profit_proxy"].to_numpy(dtype=np.float64)
price = ppo["candidate_price"].to_numpy(dtype=np.float64)

# Recover gmr from m_t = price * q_t * gmr / 100
with np.errstate(divide="ignore", invalid="ignore"):
    gmr = np.where(price * q_t > 0,
                   m_t * 100.0 / (price * q_t),
                   np.nan)

# Recover q_prev
with np.errstate(divide="ignore", invalid="ignore"):
    q_prev = np.where(1.0 + g_hat / 100.0 > 0,
                      q_t / (1.0 + g_hat / 100.0),
                      np.nan)

# Baseline m_prev comes from OPS row of chosen_OPS_v2.csv because OPS
# uses the observed discount and hence its gross_profit_proxy is the
# baseline margin at observed discount. Fall back to PPO baseline if
# the file is missing.
ops_path = OUT / "chosen_OPS_v2.csv"
if ops_path.exists():
    ops = (pd.read_csv(ops_path)
           .sort_values("test_position").reset_index(drop=True))
    # OPS g_hat is not zero; baseline needs g=0. Recompute baseline from
    # observed price/discount using the same algebra the evaluator uses:
    # baseline_sales = last_monthly_sales, baseline_price = candidate_price
    # at observed discount. But we already have q_prev and the observed
    # discount, so:
    m_prev = q_prev * price * gmr / 100.0
else:
    m_prev = np.full_like(m_t, np.nan)

# The above recomputes m_prev as if the "baseline" were at the *chosen*
# price, not the observed price. That is not what paper Eq. 9 measures.
# The correct m_prev is M at the observed discount with zero growth,
# i.e. price_obs * q_prev * gmr / 100. Reconstruct price_obs from the
# observed discount in chosen_PPO_v2.csv.
obs_disc = ppo["chosen_discount_percent"].to_numpy(dtype=np.float64)
# reference_price = candidate_price / (1 - d/100)
with np.errstate(divide="ignore", invalid="ignore"):
    ref_price = np.where(1.0 - obs_disc / 100.0 > 0,
                         price / (1.0 - obs_disc / 100.0),
                         np.nan)
# Baseline margin at observed discount uses the *observed* discount, not
# the chosen one. The chosen_PPO_v2.csv stores the chosen discount, so we
# cannot recover the observed discount from it. Use the all-actions CSV:
# for each test_position, the row at the observed discount is the one
# whose candidate_discount_percent equals the observed discount. But the
# observed discount is not in the CSV either. Simplest defensible
# baseline: use the minimum-discount row (candidate_discount=0) which is
# the closest to a no-promo baseline and is present for every product.
min_disc_row = (cf[cf["candidate_discount_percent"] == 0]
                .sort_values("test_position")
                .reset_index(drop=True))
# But 0 is out-of-support. Better: use chosen_OPS_v2 which uses the
# observed discount with the predictor at observed discount. Its
# gross_profit_proxy is *not* a zero-growth baseline, but the delta
# against it is still a valid per-product growth measure. To match the
# evaluator's macro computation, we need the evaluator's own m_prev.

# Simplest correct approach: reload the evaluator's baseline directly
# from the raw parquet with the same filter ppo_dataset applies.
prod = pd.read_parquet(ROOT / "data/processed/products_clean.parquet")
idx = np.load(ROOT / "data/processed/state/state_row_indices.npy")
aligned = prod.iloc[idx].reset_index(drop=True)
# ppo_dataset drops rows with NaN Gross Margin Rate
keep = aligned["Gross Margin Rate"].notna().to_numpy()
aligned = aligned.loc[keep].reset_index(drop=True)
# test months
test_mask = aligned["MonthNum"].isin([9, 10]).to_numpy()
aligned = aligned.loc[test_mask].reset_index(drop=True)

def _baseline_margin(row):
    denom = 1.0 - float(row["Discount"]) / 100.0
    if denom <= 0:
        return np.nan
    ref = float(row["Buybox Price (USD)"]) / denom
    price_obs = ref * (1.0 - float(row["Discount"]) / 100.0)
    sales_obs = float(row["Last Monthly Sales"])
    gmr_row = float(row["Gross Margin Rate"])
    return price_obs * sales_obs * gmr_row / 100.0

m_prev = np.array(
    [_baseline_margin(row) for _, row in aligned.iterrows()],
    dtype=np.float64)

# Realign lengths: ppo rows must match aligned rows 1:1
n = len(ppo)
if len(m_prev) != n:
    raise RuntimeError(
        f"row count mismatch: ppo={n}, aligned={len(m_prev)}")

q_prev = aligned["Last Monthly Sales"].to_numpy(dtype=np.float64)
gmr    = aligned["Gross Margin Rate"].to_numpy(dtype=np.float64)

alpha, beta = 0.5, 0.5
R_orig = paper_reward(
    delta_margin(m_t, m_prev),
    delta_quantity(q_t, q_prev), alpha, beta)
std_g = float(np.std(cf["predicted_growth_percent"].to_numpy(), ddof=1))

rows = []
for frac in [0.0, 0.05, 0.10, 0.15]:
    sigma = frac * std_g
    retained = []
    for seed in range(10):
        rng = np.random.default_rng(seed)
        eps = (rng.normal(0.0, sigma, size=n)
               if sigma > 0 else np.zeros(n))
        pq = np.maximum(0.0, q_prev * (1.0 + (g_hat + eps) / 100.0))
        pm = price * pq * gmr / 100.0
        R_n = paper_reward(
            delta_margin(pm, m_prev),
            delta_quantity(pq, q_prev), alpha, beta)
        retained.append(reward_retention(R_n, R_orig))
    rows.append({
        "noise_fraction": frac,
        "sigma": sigma,
        "std_g_hat": std_g,
        "R_original": R_orig,
        "retention_mean_percent": float(np.mean(retained)),
        "retention_std_percent": (
            float(np.std(retained, ddof=1))
            if len(retained) > 1 else 0.0),
        "retention_min_percent": float(np.min(retained)),
        "retention_max_percent": float(np.max(retained)),
        "n_seeds": 10,
    })

df = pd.DataFrame(rows)
df.to_csv(OUT / "robustness_v2.csv", index=False)
print(df.to_string(index=False))

frac0 = df.loc[df["noise_fraction"] == 0.0, "retention_mean_percent"].iloc[0]
if abs(frac0 - 100.0) > 1e-6:
    print(f"\nWARNING: retention at frac=0 is {frac0:.6f}, expected 100.0")
    sys.exit(1)
print("\nOK: retention at frac=0 is 100.0 (health check passed)")