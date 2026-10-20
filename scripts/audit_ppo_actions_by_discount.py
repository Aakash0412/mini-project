"""
Read-only PPO action/outcome audit for Dynamic Pricing for E-commerce.

Reads the existing evaluation CSV only. It does NOT load models, run inference,
train anything, or modify checkpoints / source datasets / baseline evaluation files.
All generated artifacts go into:
  reports/evaluation_audit/action_outcome_audit/

Run from the project root:
  .\.venv\Scripts\python.exe scripts\audit_ppo_actions_by_discount.py
"""
from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "models" / "ppo_transfer" / "transfer_supported_8action_policy_evaluation.csv"
OUT = ROOT / "reports" / "evaluation_audit" / "action_outcome_audit"

# These are the relative-change columns used by the prior reward rescoring audit.
REWARD_CONFIGS = {
    "sales": {"alpha": 0.2, "beta": 0.8},
    "balanced": {"alpha": 0.5, "beta": 0.5},
    "profit": {"alpha": 0.8, "beta": 0.2},
}

def first_present(columns, candidates):
    for col in candidates:
        if col in columns:
            return col
    return None

def safe_numeric(df, col):
    if col is not None:
        df[col] = pd.to_numeric(df[col], errors="coerce")

def main():
    if not INPUT.exists():
        raise FileNotFoundError(
            f"Evaluation CSV not found: {INPUT}\n"
            "Run this script from the project root and confirm the existing evaluation CSV is present."
        )

    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(INPUT, low_memory=False)
    original_rows = len(df)
    cols = list(df.columns)

    split_col = first_present(cols, ["split", "dataset_split", "data_split"])
    discount_col = first_present(cols, ["ppo_discount", "discount", "recommended_discount"])
    action_col = first_present(cols, ["ppo_action", "action"])
    source_row_col = first_present(cols, ["source_row", "row_id", "source_index"])

    rel_margin_col = first_present(cols, ["ppo_relative_delta_margin", "relative_delta_margin", "relative_margin_change"])
    rel_quantity_col = first_present(cols, ["ppo_relative_delta_quantity", "relative_delta_quantity", "relative_quantity_change"])
    reward_col = first_present(cols, ["ppo_reward", "reward"])

    if split_col is None or discount_col is None:
        raise ValueError(
            f"Could not identify split and PPO discount columns. Available columns: {cols}"
        )

    numeric_candidates = [
        discount_col, action_col, rel_margin_col, rel_quantity_col, reward_col,
        first_present(cols, ["ppo_delta_margin", "delta_margin"]),
        first_present(cols, ["ppo_delta_quantity", "delta_quantity"]),
        first_present(cols, ["ppo_sales", "sales"]),
        first_present(cols, ["ppo_revenue", "revenue"]),
        first_present(cols, ["ppo_margin", "margin"]),
        first_present(cols, ["baseline_sales"]),
        first_present(cols, ["baseline_margin"]),
        first_present(cols, ["oracle_reward"]),
        first_present(cols, ["oracle_sales"]),
        first_present(cols, ["oracle_revenue"]),
        first_present(cols, ["oracle_margin"]),
    ]
    for c in set(x for x in numeric_candidates if x is not None):
        safe_numeric(df, c)

    # Per-split, per-discount outcome summaries.
    group_cols = [split_col, discount_col]
    aggregations = {"row_count": (discount_col, "size")}
    if action_col:
        aggregations["action_id"] = (action_col, "first")
        aggregations["distinct_action_ids"] = (action_col, "nunique")
    if reward_col:
        aggregations["stored_mean_ppo_reward"] = (reward_col, "mean")
        aggregations["stored_median_ppo_reward"] = (reward_col, "median")
        aggregations["stored_positive_reward_fraction"] = (reward_col, lambda s: (s > 0).mean())
    if rel_quantity_col:
        aggregations["mean_relative_quantity_change"] = (rel_quantity_col, "mean")
        aggregations["median_relative_quantity_change"] = (rel_quantity_col, "median")
    if rel_margin_col:
        aggregations["mean_relative_margin_change"] = (rel_margin_col, "mean")
        aggregations["median_relative_margin_change"] = (rel_margin_col, "median")

    # Include direct PPO / baseline / oracle outcome fields if available.
    for output_name, candidates in {
        "mean_ppo_sales": ["ppo_sales", "sales"],
        "mean_ppo_revenue": ["ppo_revenue", "revenue"],
        "mean_ppo_margin": ["ppo_margin", "margin"],
        "mean_baseline_sales": ["baseline_sales"],
        "mean_baseline_margin": ["baseline_margin"],
        "mean_oracle_sales": ["oracle_sales"],
        "mean_oracle_revenue": ["oracle_revenue"],
        "mean_oracle_margin": ["oracle_margin"],
        "mean_oracle_reward": ["oracle_reward"],
    }.items():
        c = first_present(cols, candidates)
        if c:
            aggregations[output_name] = (c, "mean")

    by_discount = df.groupby(group_cols, dropna=False).agg(**aggregations).reset_index()
    by_discount.to_csv(OUT / "ppo_outcomes_by_split_and_discount.csv", index=False)

    # Recompute reward under all three weights from stored relative changes only.
    rescoring_path = None
    if rel_margin_col and rel_quantity_col:
        parts = []
        for config, weights in REWARD_CONFIGS.items():
            tmp = df[[split_col, discount_col]].copy()
            tmp["reward_config"] = config
            tmp["alpha"] = weights["alpha"]
            tmp["beta"] = weights["beta"]
            tmp["rescored_reward"] = (
                weights["alpha"] * df[rel_margin_col] +
                weights["beta"] * df[rel_quantity_col]
            )
            parts.append(tmp)
        rescored = pd.concat(parts, ignore_index=True)
        reward_summary = (
            rescored.groupby(["reward_config", "alpha", "beta", split_col, discount_col], dropna=False)
            .agg(
                row_count=("rescored_reward", "size"),
                mean_rescored_reward=("rescored_reward", "mean"),
                median_rescored_reward=("rescored_reward", "median"),
                positive_reward_fraction=("rescored_reward", lambda s: (s > 0).mean()),
            )
            .reset_index()
        )
        rescoring_path = OUT / "rescored_rewards_by_split_and_discount.csv"
        reward_summary.to_csv(rescoring_path, index=False)

    # Split sizes and source-row diagnostics. source_row is not assumed to be a product ID.
    split_summary = df.groupby(split_col, dropna=False).size().rename("rows").reset_index()
    split_summary.to_csv(OUT / "split_row_counts.csv", index=False)

    # Search for likely product identifiers; only test actual columns, never infer that source_row is a product.
    id_candidates = [
        "asin", "ASIN", "product_id", "productId", "productID", "sku", "SKU",
        "product_identifier", "item_id", "itemId", "parent_asin", "product_code"
    ]
    id_col = first_present(cols, id_candidates)
    overlap_report = {
        "identifier_column_used": id_col,
        "source_row_column_present": source_row_col,
        "overlap_check_status": "not_possible_no_product_identifier_column",
        "split_unique_identifiers": {},
        "pairwise_overlap_counts": {},
        "note": (
            "source_row is treated as a row locator, not a product identifier. "
            "No product-level overlap conclusion is made unless an explicit product ID column exists."
        ),
    }
    if id_col:
        work = df[[split_col, id_col]].dropna().copy()
        work[id_col] = work[id_col].astype(str)
        sets = {
            str(split): set(group[id_col].unique())
            for split, group in work.groupby(split_col, dropna=False)
        }
        overlap_report["overlap_check_status"] = "checked_using_explicit_product_identifier"
        overlap_report["split_unique_identifiers"] = {k: len(v) for k, v in sets.items()}
        keys = list(sets)
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                a, b = keys[i], keys[j]
                overlap_report["pairwise_overlap_counts"][f"{a}__{b}"] = len(sets[a] & sets[b])

    # Basic validation checks and exact columns used.
    missing_by_col = df.isna().sum().sort_values(ascending=False)
    missing_by_col = missing_by_col[missing_by_col > 0]
    report = {
        "purpose": "Read-only grouping of stored PPO outcomes by split and selected discount",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_csv": str(INPUT.relative_to(ROOT)),
        "output_directory": str(OUT.relative_to(ROOT)),
        "rows_read": original_rows,
        "columns": cols,
        "identified_columns": {
            "split": split_col,
            "discount": discount_col,
            "action": action_col,
            "source_row": source_row_col,
            "stored_reward": reward_col,
            "relative_margin_change": rel_margin_col,
            "relative_quantity_change": rel_quantity_col,
            "explicit_product_identifier": id_col,
        },
        "split_counts": {str(k): int(v) for k, v in df[split_col].value_counts(dropna=False).items()},
        "missing_value_counts_nonzero": {str(k): int(v) for k, v in missing_by_col.items()},
        "product_overlap": overlap_report,
        "outputs": {
            "outcomes_by_split_and_discount": "ppo_outcomes_by_split_and_discount.csv",
            "rescored_rewards_by_split_and_discount": (
                "rescored_rewards_by_split_and_discount.csv" if rescoring_path else None
            ),
            "split_row_counts": "split_row_counts.csv",
            "summary_json": "action_outcome_audit.json",
            "summary_markdown": "action_outcome_audit.md",
        },
        "safety": {
            "model_loaded": False,
            "inference_performed": False,
            "training_performed": False,
            "checkpoint_modified": False,
            "input_csv_modified": False,
        },
    }
    (OUT / "action_outcome_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "# PPO Action/Outcome Audit",
        "",
        "## Scope",
        "- Read-only analysis of the existing evaluation CSV.",
        "- No model loading, inference, training, checkpoint changes, or input CSV changes.",
        "- `source_row` is not treated as a product identifier.",
        "",
        "## Split row counts",
        split_summary.to_markdown(index=False),
        "",
        "## Selected discount outcome summary",
        by_discount.to_markdown(index=False),
        "",
        "## Product overlap check",
        f"- Status: `{overlap_report['overlap_check_status']}`",
        f"- Explicit product identifier column: `{id_col}`",
    ]
    if id_col:
        lines += [
            f"- Unique identifiers by split: `{json.dumps(overlap_report['split_unique_identifiers'], sort_keys=True)}`",
            f"- Pairwise overlaps: `{json.dumps(overlap_report['pairwise_overlap_counts'], sort_keys=True)}`",
        ]
    else:
        lines += [
            "- The evaluation CSV has no recognized explicit product identifier column.",
            "- Product overlap cannot be established from this file alone. A `source_row` value is only a row locator unless its mapping to product IDs is verified.",
        ]
    if rescoring_path:
        lines += [
            "",
            "## Reward-weight sensitivity by split and discount",
            "See `rescored_rewards_by_split_and_discount.csv` for sales, balanced, and profit reward weights.",
            "These are post-hoc scores for the same fixed PPO actions, not separately trained policies.",
        ]
    lines += [
        "",
        "## Interpretation cautions",
        "- Differences by discount are descriptive and do not establish causality.",
        "- A discount group can contain different products/states across splits.",
        "- The stored oracle is an offline comparator, not automatically a deployable policy.",
        "- Inspect the CSV columns and project definitions before interpreting sales/margin units as real-world business outcomes.",
        "",
        "## Safety",
        "- Model inference: no",
        "- Training: no",
        "- Checkpoint modified: no",
        "- Input evaluation CSV modified: no",
    ]
    (OUT / "action_outcome_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("Read-only PPO action/outcome audit completed.")
    print(f"Rows read: {original_rows}")
    print(f"Split counts: {report['split_counts']}")
    print(f"Discount outcome summary: {OUT / 'ppo_outcomes_by_split_and_discount.csv'}")
    if rescoring_path:
        print(f"Reward-weight summary: {rescoring_path}")
    print(f"Product overlap status: {overlap_report['overlap_check_status']}")
    print(f"Report: {OUT / 'action_outcome_audit.md'}")
    print("Safety: no inference, no training, no checkpoint changes, no input CSV changes.")

if __name__ == "__main__":
    main()
