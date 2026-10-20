"""v2 paper-aligned evaluator. Reads frozen artifacts; writes only into
reports/base_paper_reproduction_v2. Never overwrites v1."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.ppo_agent import PPOActor
from src.environment.ppo_dataset import load_ppo_dataset
from src.environment.counterfactual_simulator import (
    CounterfactualSimulator, DISCOUNT_BINS,
)
from src.environment.pricing_economics import calculate_pricing_outcome
from src.evaluation.paper_metrics import (
    delta_margin, delta_quantity, mean_growth,
    paper_reward, top1_top3, reward_retention,
)
from src.evaluation.profit_model import PROFIT_DEFINITION, label_outcome
from src.evaluation.baselines import HEURISTIC_BASELINES, set_dop_median


CHECKPOINT = ROOT / "models/ppo/ppo_batched_economic_50_updates.pth"
PRODUCTS = ROOT / "data/processed/products_clean.parquet"
OUTPUT_DIR = ROOT / "reports/base_paper_reproduction_v2"

REWARD_SETTINGS = {
    "sales": (0.2, 0.8), "balanced": (0.5, 0.5), "profit": (0.8, 0.2),
}
NOISE_FRACTIONS = [0.0, 0.05, 0.10, 0.15]
NOISE_SEEDS = list(range(10))


def load_actor() -> PPOActor:
    device = torch.device("cpu")
    ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    if ckpt.get("state_dim") != 111 or ckpt.get("action_dim") != 9:
        raise ValueError("PPO checkpoint dims not 111/9.")
    actor = PPOActor(state_dim=111, action_dim=9).to(device)
    actor.load_state_dict(ckpt["actor_state_dict"])
    actor.eval()
    return actor


def load_test_split():
    dataset = load_ppo_dataset(
        state_path=ROOT / "data/processed/state/state.npy",
        index_path=ROOT / "data/processed/state/state_row_indices.npy",
        products_path=PRODUCTS,
    )
    df = pd.read_parquet(PRODUCTS)
    aligned = df.iloc[dataset.source_row_indices].reset_index(drop=True)
    test_mask = aligned["MonthNum"].isin([9, 10]).to_numpy()
    states = dataset.states[test_mask]
    contexts = [c for c, k in zip(dataset.products, test_mask) if k]
    months = aligned.loc[test_mask, "MonthNum"].to_numpy()
    source_rows = dataset.source_row_indices[test_mask]
    return states, contexts, months, source_rows


def outcome_row(state, ctx, month, candidate_discount, simulator):
    growth = simulator.predict(state, candidate_discount)
    raw = calculate_pricing_outcome(
        observed_price=ctx.observed_price,
        observed_discount=ctx.observed_discount,
        last_monthly_sales=ctx.last_monthly_sales,
        predicted_growth_rate=growth,
        candidate_discount=candidate_discount,
        gross_margin_rate=ctx.gross_margin_rate,
    )
    lab = label_outcome(raw)
    return {
        "month_num": int(month),
        "candidate_discount_percent": int(candidate_discount),
        "predicted_growth_percent": float(growth),
        "estimated_sales": float(raw.estimated_sales),
        "candidate_price": float(raw.candidate_price),
        "reference_price": float(raw.reference_price),
        "gross_profit_proxy": (
            float(lab.gross_profit_proxy)
            if lab.gross_profit_proxy is not None else np.nan
        ),
        "profit_definition": PROFIT_DEFINITION,
    }

def main() -> None:
    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        raise FileExistsError(
            f"{OUTPUT_DIR} exists; remove it or pick another dir."
        )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    actor = load_actor()
    simulator = CounterfactualSimulator(
        checkpoint_dir=str(ROOT / "models/sales_predictor"),
        device=torch.device("cpu"),
    )
    states, contexts, months, source_rows = load_test_split()
    n = len(states)
    if n == 0:
        raise RuntimeError("No test rows for MonthNum in {9,10}.")

    set_dop_median(float(np.median([c.last_monthly_sales for c in contexts])))

    with torch.no_grad():
        logits = actor(torch.as_tensor(states, dtype=torch.float32))
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
    ppo_action_idx = probs.argmax(axis=1)

    cf_rows = []
    chosen_rows = {"PPO": []}
    for name in HEURISTIC_BASELINES:
        chosen_rows[name] = []

    rng = np.random.default_rng(42)
    for i in range(n):
        state, ctx, month = states[i], contexts[i], months[i]
        action_table = {}
        for d in DISCOUNT_BINS:
            row = outcome_row(state, ctx, month, int(d), simulator)
            action_table[int(d)] = row
            cf_rows.append({**row,
                "test_position": i,
                "source_row_index": int(source_rows[i]),
                "model": "PPO"})

        d_ppo = int(DISCOUNT_BINS[int(ppo_action_idx[i])])
        chosen_rows["PPO"].append({**action_table[d_ppo],
            "test_position": i,
            "source_row_index": int(source_rows[i]),
            "chosen_discount_percent": d_ppo})

        for name, fn in HEURISTIC_BASELINES.items():
            idx = fn(ctx, int(month), rng)
            d_base = int(DISCOUNT_BINS[int(idx)])
            chosen_rows[name].append({**action_table[d_base],
                "test_position": i,
                "source_row_index": int(source_rows[i]),
                "chosen_discount_percent": d_base})

    cf_df = pd.DataFrame(cf_rows)
    cf_df.to_csv(OUTPUT_DIR / "all_action_counterfactuals_v2.csv", index=False)
    for name, rows in chosen_rows.items():
        pd.DataFrame(rows).to_csv(
            OUTPUT_DIR / f"chosen_{name}_v2.csv", index=False)

    q_prev = np.array([c.last_monthly_sales for c in contexts], dtype=np.float64)
    m_prev = np.array([
        calculate_pricing_outcome(
            observed_price=contexts[i].observed_price,
            observed_discount=contexts[i].observed_discount,
            last_monthly_sales=contexts[i].last_monthly_sales,
            predicted_growth_rate=0.0,
            candidate_discount=contexts[i].observed_discount,
            gross_margin_rate=contexts[i].gross_margin_rate,
        ).estimated_margin
        for i in range(n)
    ], dtype=np.float64)

    macro_rows = []
    for model_name, rows in chosen_rows.items():
        df_m = (pd.DataFrame(rows)
                .sort_values("test_position").reset_index(drop=True))
        q_t = df_m["estimated_sales"].to_numpy(dtype=np.float64)
        m_t = df_m["gross_profit_proxy"].to_numpy(dtype=np.float64)
        dM = delta_margin(m_t, m_prev)
        dQ = delta_quantity(q_t, q_prev)
        for setting, (alpha, beta) in REWARD_SETTINGS.items():
            r = paper_reward(dM, dQ, alpha, beta)
            macro_rows.append({
                "model": model_name, "reward_setting": setting,
                "alpha": alpha, "beta": beta,
                "mean_delta_margin": mean_growth(dM),
                "mean_delta_quantity": mean_growth(dQ),
                "paper_reward": r,
                "nonfinite_delta_margin": int((~np.isfinite(dM)).sum()),
                "nonfinite_delta_quantity": int((~np.isfinite(dQ)).sum()),
                "n_products": int(len(df_m))})
    pd.DataFrame(macro_rows).to_csv(
        OUTPUT_DIR / "macro_reward_v2.csv", index=False)

    micro_rows = []
    for setting, (alpha, beta) in REWARD_SETTINGS.items():
        per_dM, per_dQ = {}, {}
        for model_name, rows in chosen_rows.items():
            df_m = (pd.DataFrame(rows)
                    .sort_values("test_position").reset_index(drop=True))
            per_dM[model_name] = delta_margin(
                df_m["gross_profit_proxy"].to_numpy(dtype=np.float64), m_prev)
            per_dQ[model_name] = delta_quantity(
                df_m["estimated_sales"].to_numpy(dtype=np.float64), q_prev)
        for mname, per_model in [("delta_margin", per_dM),
                                 ("delta_quantity", per_dQ)]:
            ranking = top1_top3(per_model, higher_is_better=True)
            for model_name, stats in ranking.items():
                micro_rows.append({
                    "reward_setting": setting,
                    "metric": mname,
                    "model": model_name,
                    **stats})
    pd.DataFrame(micro_rows).to_csv(
        OUTPUT_DIR / "micro_topk_v2.csv", index=False)

    std_g = float(np.std(
        cf_df["predicted_growth_percent"].to_numpy(), ddof=1))
    alpha, beta = REWARD_SETTINGS["balanced"]
    df_ppo = (pd.DataFrame(chosen_rows["PPO"])
              .sort_values("test_position").reset_index(drop=True))
    q_t = df_ppo["estimated_sales"].to_numpy(dtype=np.float64)
    m_t = df_ppo["gross_profit_proxy"].to_numpy(dtype=np.float64)
    R_orig = paper_reward(
        delta_margin(m_t, m_prev), delta_quantity(q_t, q_prev), alpha, beta)

    price = df_ppo["candidate_price"].to_numpy(dtype=np.float64)
    gmr = np.array([
        c.gross_margin_rate if c.gross_margin_rate is not None else np.nan
        for c in contexts], dtype=np.float64)
    g_hat = df_ppo["predicted_growth_percent"].to_numpy(dtype=np.float64)

    robust_rows = []
    for frac in NOISE_FRACTIONS:
        sigma = frac * std_g
        retained = []
        for seed in NOISE_SEEDS:
            rng2 = np.random.default_rng(seed)
            eps = rng2.normal(0.0, sigma, size=n) if sigma > 0 else np.zeros(n)
            pq = np.maximum(0.0, q_prev * (1.0 + (g_hat + eps) / 100.0))
            pm = price * pq * gmr
            R_n = paper_reward(
                delta_margin(pm, m_prev),
                delta_quantity(pq, q_prev), alpha, beta)
            retained.append(reward_retention(R_n, R_orig))
        robust_rows.append({
            "noise_fraction": frac, "sigma": sigma, "std_g_hat": std_g,
            "R_original": R_orig,
            "retention_mean_percent": float(np.mean(retained)),
            "retention_std_percent": (
                float(np.std(retained, ddof=1)) if len(retained) > 1 else 0.0),
            "retention_min_percent": float(np.min(retained)),
            "retention_max_percent": float(np.max(retained)),
            "n_seeds": len(NOISE_SEEDS)})
    pd.DataFrame(robust_rows).to_csv(
        OUTPUT_DIR / "robustness_v2.csv", index=False)

    summary = {
        "evaluation_type": "one-step counterfactual, paper-aligned metrics",
        "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
        "state_dimension": 111,
        "paper_state_dimension": 121,
        "action_bins_percent": DISCOUNT_BINS,
        "test_months": [9, 10],
        "test_products": int(n),
        "profit_definition": PROFIT_DEFINITION,
        "reward_settings": REWARD_SETTINGS,
        "models_evaluated": list(chosen_rows.keys()),
        "models_not_implemented_in_v2": [
            "UCB", "TS", "DQNF", "DecisionTransformer", "TransformerDQN"],
        "comparable_to_paper": {
            "mean_delta_margin": "partial (Eq.9; Eq.8 reduced S=V=0,Q^r=0)",
            "mean_delta_quantity": "yes in definition (Eq.13); simulator-dependent",
            "paper_reward": "partial (Eq.14 shape; one-step only)",
            "top1_top3": "partial (heuristic baselines only)",
            "robustness_retention": "yes in protocol (eps ~ N(0,(frac*Std(g))^2))",
            "cumulative_return_G": "no (sequential env not implemented)"},
        "limitations": [
            "111-D state omits paper 10-D sentiment.",
            "Title uses MiniLM-32D, not ada-002.",
            "Profit uses Eq.8 reduced with S=V=0,Q^r=0,C=1-GMR.",
            "One-step; not cumulative return G.",
            "Top-1/Top-3 over heuristics only.",
            "Not the paper's 11-baseline population."]}

    (OUTPUT_DIR / "v2_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")

    comparison = [
        {"metric": "mean_profit_margin_growth",
         "paper": ">16.7% (Eq.8 full cost)",
         "v1": "31.39% (rev*GMR proxy)",
         "v2": f"{mean_growth(delta_margin(m_t, m_prev)):.6f}",
         "comparable_to_paper": "partial"},
        {"metric": "mean_sales_growth",
         "paper": "-0.3% to -2.0%",
         "v1": "+22.55% (aggregate simulated)",
         "v2": f"{mean_growth(delta_quantity(q_t, q_prev)):.6f}",
         "comparable_to_paper": "yes in definition"},
        {"metric": "paper_reward_balanced",
         "paper": "cumulative G Jan-Oct",
         "v1": "0.3848 (one-step)",
         "v2": f"{R_orig:.6f} (one-step)",
         "comparable_to_paper": "partial"},
        {"metric": "top1_profit_growth",
         "paper": "52.32% (3563/6810)",
         "v1": "25.69% (margin proxy)",
         "v2": "see micro_topk_v2.csv (heuristics only)",
         "comparable_to_paper": "partial"},
        {"metric": "top3_sales_growth",
         "paper": "74.33% (5062/6810)",
         "v1": "85.92% (sales result)",
         "v2": "see micro_topk_v2.csv (heuristics only)",
         "comparable_to_paper": "partial"},
        {"metric": "robustness_retention_10pct",
         "paper": "92.8% (Table 12)",
         "v1": "99.864%",
         "v2": "see robustness_v2.csv",
         "comparable_to_paper": "yes in protocol"}]
    pd.DataFrame(comparison).to_csv(
        OUTPUT_DIR / "v2_vs_v1_vs_paper.csv", index=False)

    print(f"v2 done. Output: {OUTPUT_DIR}")
    print(f"  test_products = {n}")
    print(f"  R_original    = {R_orig:.6f}")
    print(f"  Std(g_hat)    = {std_g:.4f}")
    for path in sorted(OUTPUT_DIR.glob('*')):
        print(f"  {path.name}")


if __name__ == "__main__":
    main()