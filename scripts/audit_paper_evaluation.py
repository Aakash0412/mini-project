from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "evaluation_audit"
EVAL = ROOT / "models" / "ppo_transfer" / "transfer_supported_8action_policy_evaluation.csv"
ABL = ROOT / "reports" / "ablation" / "ablation_results.csv"
ROB = ROOT / "reports" / "robustness" / "robustness_results.csv"
ROB_SUM = ROOT / "reports" / "robustness" / "robustness_summary.json"
CKPT = ROOT / "models" / "ppo_transfer" / "ppo_transfer_supported_8action_1000_updates.pth"

def summary(series):
    x = pd.to_numeric(series, errors="coerce").dropna()
    if x.empty:
        return {"count": 0}
    return {"count": int(x.size), "mean": float(x.mean()), "median": float(x.median()),
            "std": float(x.std()), "min": float(x.min()), "max": float(x.max()),
            "q05": float(x.quantile(.05)), "q25": float(x.quantile(.25)),
            "q75": float(x.quantile(.75)), "q95": float(x.quantile(.95))}

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    r = {"read_only": True, "model_training_performed": False, "checkpoint_modified": False,
         "artifacts": {}, "evaluation": {}, "paper_alignment": {}, "limitations": []}
    for name, path in [("ppo_evaluation", EVAL), ("predictor_ablation", ABL),
                       ("robustness_results", ROB), ("robustness_summary", ROB_SUM),
                       ("ppo_checkpoint", CKPT)]:
        r["artifacts"][name] = {"path": str(path.relative_to(ROOT)), "exists": path.is_file()}
    if EVAL.is_file():
        df = pd.read_csv(EVAL)
        r["evaluation"]["ppo_evaluation_rows"] = len(df)
        r["evaluation"]["columns"] = list(df.columns)
        if "split" in df:
            r["evaluation"]["split_counts"] = {str(k): int(v) for k,v in df["split"].value_counts(dropna=False).items()}
        if {"split","ppo_reward"}.issubset(df.columns):
            r["evaluation"]["reward_by_split"] = {str(k): summary(g["ppo_reward"]) for k,g in df.groupby("split", dropna=False)}
        if {"split","ppo_reward","oracle_reward"}.issubset(df.columns):
            test = df[df["split"].astype(str).str.lower().eq("test")]
            if len(test):
                pm, om = float(test["ppo_reward"].mean()), float(test["oracle_reward"].mean())
                r["evaluation"]["test_ppo_vs_oracle"] = {
                    "rows": len(test), "ppo_reward": summary(test["ppo_reward"]),
                    "oracle_reward": summary(test["oracle_reward"]),
                    "mean_reward_ratio_percent": 100*pm/om if abs(om)>1e-12 else None,
                    "mean_reward_gap": om-pm,
                    "caveat": "Per-row oracle is an offline comparator, not a deployable policy and not necessarily the paper's comparator."}
        r["evaluation"]["product_ranking_metrics"] = {
            "status": "not computed from this file alone",
            "reason": "Top-1/Top-3 paper metrics need aligned per-product outcomes for each competing policy, not only PPO and oracle."
        }
    else:
        r["limitations"].append("PPO evaluation CSV missing at expected path.")
    if ABL.is_file():
        a = pd.read_csv(ABL)
        r["evaluation"]["predictor_ablation"] = a.to_dict(orient="records")
    else:
        r["limitations"].append("Predictor ablation CSV missing.")
    if ROB.is_file():
        b = pd.read_csv(ROB)
        r["evaluation"]["robustness"] = b.to_dict(orient="records")
    else:
        r["limitations"].append("Robustness CSV missing.")
    if ROB_SUM.is_file():
        try: r["evaluation"]["robustness_metadata"] = json.loads(ROB_SUM.read_text(encoding="utf-8"))
        except Exception as e: r["limitations"].append("Robustness summary JSON parse error: "+str(e))
    r["paper_alignment"] = {
        "predictor_metrics": "Available if ablation artifact exists; these are not policy metrics.",
        "macro_reward_distribution": "Can summarize per-product PPO reward rows; this audit calculates descriptive statistics.",
        "three_reward_settings": "Must verify/run sales-oriented (0.2,0.8), balanced (0.5,0.5), and profit-oriented (0.8,0.2) separately.",
        "micro_product_rankings": "Requires per-product outcomes from all compared policies; cannot be inferred from PPO-vs-oracle alone.",
        "algorithm_baselines": "Incomplete unless comparable heuristic, bandit, DRL, and Transformer evaluation outputs exist.",
        "robustness": "Separate simulated Gaussian-noise experiment; does not establish real-world robustness.",
        "reproduction_caveat": "Current PPO uses 8 supported discounts (5%-40%); paper describes 9 actions including 0%."
    }
    (OUT/"paper_evaluation_audit.json").write_text(json.dumps(r, indent=2, default=str), encoding="utf-8")
    lines = ["# Base-Paper Evaluation Audit", "",
             "Read-only audit. No model training and no checkpoint changes.", "",
             "## Artifact presence"]
    for k,v in r["artifacts"].items():
        lines.append(f"- **{k}:** {'FOUND' if v['exists'] else 'MISSING'} — `{v['path']}`")
    lines += ["", "## PPO reward summaries"]
    for k,v in r["evaluation"].get("reward_by_split", {}).items():
        if v.get("count"):
            lines.append(f"- **{k}:** n={v['count']}, mean={v['mean']:.6f}, median={v['median']:.6f}, SD={v['std']:.6f}, 5th–95th percentile={v['q05']:.6f}–{v['q95']:.6f}")
    z = r["evaluation"].get("test_ppo_vs_oracle")
    if z:
        lines += ["", "## Test PPO vs oracle",
                  f"- Test rows: {z['rows']}",
                  f"- PPO mean reward: {z['ppo_reward']['mean']:.6f}",
                  f"- Oracle mean reward: {z['oracle_reward']['mean']:.6f}",
                  f"- Mean reward ratio: {z['mean_reward_ratio_percent']:.2f}%",
                  f"- Mean reward gap: {z['mean_reward_gap']:.6f}",
                  f"- Caveat: {z['caveat']}"]
    if r["evaluation"].get("predictor_ablation"):
        lines += ["", "## Predictor ablation test metrics"]
        for x in r["evaluation"]["predictor_ablation"]:
            lines.append(f"- **{x.get('description',x.get('configuration'))}:** MAE={x.get('test_mae')}, RMSE={x.get('test_rmse')}, R²={x.get('test_r2')}, Spearman={x.get('test_spearman')}")
    if r["evaluation"].get("robustness"):
        lines += ["", "## Robustness results"]
        for x in r["evaluation"]["robustness"]:
            lines.append(f"- Noise={x.get('noise_fraction')}: mean reward={x.get('mean_reward')}, retention={x.get('reward_retention_percent')}%, n={x.get('evaluated_products')}")
    lines += ["", "## Paper-alignment assessment"]
    for k,v in r["paper_alignment"].items():
        lines.append(f"- **{k.replace('_',' ').title()}:** {v}")
    lines += ["", "## Next steps",
              "1. Locate or run the three reward settings on identical held-out rows.",
              "2. Evaluate supported simple pricing baselines with the same economics and test products.",
              "3. Compute product-level Top-1/Top-3 sales and margin rankings only when per-policy outcomes are aligned.",
              "4. Then run the transfer-target ablation and multi-seed reliability experiments.",
              "", "All results are offline simulated; do not claim verified real-world profit improvement."]
    (OUT/"paper_evaluation_audit.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print("Audit complete:", OUT)
    print("Created paper_evaluation_audit.md and paper_evaluation_audit.json")
    print("No training performed; checkpoints unchanged.")

if __name__ == "__main__":
    main()
