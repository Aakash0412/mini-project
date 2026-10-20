from __future__ import annotations

import json
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "models" / "ppo_transfer" / "transfer_supported_8action_policy_evaluation.csv"
OUT_DIR = ROOT / "reports" / "evaluation_audit" / "reward_weight_rescoring"
OUT_DIR.mkdir(parents=True, exist_ok=True)

if not INPUT.exists():
    raise FileNotFoundError(f"Required existing evaluation CSV not found: {INPUT}")

df = pd.read_csv(INPUT)
required = {
    "split", "source_row", "ppo_action", "ppo_discount",
    "ppo_relative_delta_quantity", "ppo_relative_delta_margin",
}
missing = sorted(required - set(df.columns))
if missing:
    raise ValueError(f"Missing required columns in existing evaluation CSV: {missing}")

# These match src/environment/reward.py's documented reward definition:
# reward = alpha * relative_delta_margin + beta * relative_delta_quantity.
configs = {
    "sales": {"alpha": 0.2, "beta": 0.8},
    "balanced": {"alpha": 0.5, "beta": 0.5},
    "profit": {"alpha": 0.8, "beta": 0.2},
}
results = []
summary = {
    "purpose": "Post-hoc rescoring of the already-selected PPO actions under three reward weight configurations.",
    "read_only_inputs": True,
    "model_inference_performed": False,
    "training_performed": False,
    "checkpoint_modified": False,
    "input_csv": str(INPUT.relative_to(ROOT)),
    "output_directory": str(OUT_DIR.relative_to(ROOT)),
    "important_interpretation": (
        "PPO actions are held fixed. This is reward-weight sensitivity analysis, "
        "not evaluation of policies trained under each reward objective."
    ),
    "formula": "rescored_reward = alpha * ppo_relative_delta_margin + beta * ppo_relative_delta_quantity",
    "configs": configs,
    "rows": int(len(df)),
    "split_counts": {str(k): int(v) for k, v in df["split"].value_counts(dropna=False).items()},
    "metrics": [],
}

for name, cfg in configs.items():
    scored = df.copy()
    scored["reward_config"] = name
    scored["reward_alpha"] = cfg["alpha"]
    scored["reward_beta"] = cfg["beta"]
    scored["rescored_ppo_reward"] = (
        cfg["alpha"] * pd.to_numeric(scored["ppo_relative_delta_margin"], errors="coerce")
        + cfg["beta"] * pd.to_numeric(scored["ppo_relative_delta_quantity"], errors="coerce")
    )
    scored["rescored_ppo_reward"] = scored["rescored_ppo_reward"].replace([np.inf, -np.inf], np.nan)
    results.append(scored)

    for split, part in scored.groupby("split", dropna=False):
        values = part["rescored_ppo_reward"].dropna()
        summary["metrics"].append({
            "reward_config": name,
            "alpha": cfg["alpha"],
            "beta": cfg["beta"],
            "split": str(split),
            "n": int(values.size),
            "mean_reward": float(values.mean()) if values.size else None,
            "median_reward": float(values.median()) if values.size else None,
            "std_reward_sample": float(values.std(ddof=1)) if values.size > 1 else None,
            "q05": float(values.quantile(0.05)) if values.size else None,
            "q95": float(values.quantile(0.95)) if values.size else None,
            "positive_reward_fraction": float((values > 0).mean()) if values.size else None,
        })

combined = pd.concat(results, ignore_index=True)
metrics_df = pd.DataFrame(summary["metrics"])
combined.to_csv(OUT_DIR / "ppo_reward_weight_rescoring_rows.csv", index=False)
metrics_df.to_csv(OUT_DIR / "ppo_reward_weight_rescoring_summary.csv", index=False)

# Policy action distribution is objective-independent because the selected actions are frozen.
action_counts = (
    df.groupby(["split", "ppo_action", "ppo_discount"], dropna=False)
      .size().rename("count").reset_index()
)
action_counts["fraction_within_split"] = action_counts["count"] / action_counts.groupby("split")["count"].transform("sum")
action_counts.to_csv(OUT_DIR / "frozen_ppo_action_distribution.csv", index=False)

summary["action_distribution_file"] = "frozen_ppo_action_distribution.csv"
summary["row_results_file"] = "ppo_reward_weight_rescoring_rows.csv"
summary["summary_file"] = "ppo_reward_weight_rescoring_summary.csv"
(OUT_DIR / "ppo_reward_weight_rescoring.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

md = [
    "# PPO Reward-Weight Rescoring",
    "",
    "## Scope",
    "- Existing evaluation CSV only; no inference, training, or checkpoint changes.",
    "- PPO's already-selected action for each row is held fixed.",
    "- Reward is recomputed from the stored relative margin and quantity changes.",
    "- This is not a claim that the policy was trained under balanced/profit objectives.",
    "",
    "## Results by split",
    "",
    metrics_df.to_markdown(index=False),
    "",
    "## Important limitation",
    "These results measure how the same frozen action recommendations score when the weights change. "
    "They do not replace separately trained policies for each reward objective. Baseline comparisons "
    "should be run in a separate experiment using the same held-out products and exact reward definition.",
]
(OUT_DIR / "ppo_reward_weight_rescoring.md").write_text("\n".join(md) + "\n", encoding="utf-8")

print(f"Rescoring complete: {OUT_DIR}")
print(metrics_df.to_string(index=False))
print("No model inference/training performed; checkpoints unchanged.")
