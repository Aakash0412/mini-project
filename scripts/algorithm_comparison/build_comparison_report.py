from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
BASELINE_DIR = ROOT / "reports" / "evaluation_audit" / "final_baseline_comparison"
COMPARISON_DIR = ROOT / "reports" / "algorithm_comparison"

PPO_DETAILS = BASELINE_DIR / "by_product.csv"
BASELINE_SUMMARY = BASELINE_DIR / "summary.csv"

EVALUATION_FILES = {
    "DDPG": COMPARISON_DIR / "ddpg_test_evaluation.csv",
    "SAC": COMPARISON_DIR / "sac_test_evaluation.csv",
    "DDQN": COMPARISON_DIR / "ddqn_test_evaluation.csv",
}

OUTPUT_SUMMARY = COMPARISON_DIR / "algorithm_comparison_summary.csv"
OUTPUT_DETAILS = COMPARISON_DIR / "algorithm_comparison_details.csv"
OUTPUT_REPORT = COMPARISON_DIR / "algorithm_comparison_report.md"

DISCOUNT_BINS = [5, 10, 15, 20, 25, 30, 35, 40]


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required input file not found: {path}")


def main() -> None:
    inputs = [PPO_DETAILS, BASELINE_SUMMARY, *EVALUATION_FILES.values()]
    for path in inputs:
        require_file(path)

    outputs = [OUTPUT_SUMMARY, OUTPUT_DETAILS, OUTPUT_REPORT]
    existing = [str(path) for path in outputs if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing report files:\n"
            + "\n".join(existing)
            + "\nPreserve them and choose new output filenames."
        )

    baseline_summary = pd.read_csv(BASELINE_SUMMARY)
    ppo_rows = baseline_summary.loc[
        baseline_summary["policy"].eq("PPO")
    ]
    if len(ppo_rows) != 1:
        raise ValueError("Expected exactly one PPO row in baseline summary.")

    ppo_summary = ppo_rows.iloc[0]
    details_frames = []
    summary_rows = []

    # Preserve the original PPO per-product file, adapting its column names
    # only in memory for a common cross-algorithm schema.
    ppo_df = pd.read_csv(PPO_DETAILS)
    if len(ppo_df) != int(ppo_summary["evaluated_products"]):
        raise ValueError(
            "PPO per-product row count does not match the baseline summary."
        )

    ppo_columns = set(ppo_df.columns)
    required_ppo_candidates = {
        "source_row": ["source_row", "source_row_index", "row_index"],
        "reward": ["ppo_reward", "reward", "mean_reward"],
        "discount": ["ppo_discount", "discount", "discount_pct"],
    }

    def find_column(frame: pd.DataFrame, candidates: list[str], label: str) -> str:
        for candidate in candidates:
            if candidate in frame.columns:
                return candidate
        raise ValueError(
            f"Cannot find {label} in columns: {list(frame.columns)}"
        )

    ppo_source_col = find_column(
        ppo_df, required_ppo_candidates["source_row"], "PPO source row"
    )
    ppo_reward_col = find_column(
        ppo_df, required_ppo_candidates["reward"], "PPO reward"
    )
    ppo_discount_col = find_column(
        ppo_df, required_ppo_candidates["discount"], "PPO discount"
    )

    ppo_details = pd.DataFrame({
        "algorithm": "PPO",
        "split": "test",
        "source_row": ppo_df[ppo_source_col].astype("int64"),
        "discount": pd.to_numeric(ppo_df[ppo_discount_col], errors="raise"),
        "reward": pd.to_numeric(ppo_df[ppo_reward_col], errors="raise"),
    })

    if ppo_details["source_row"].duplicated().any():
        raise ValueError("Duplicate source_row values found in PPO results.")

    details_frames.append(ppo_details)

    summary_rows.append({
        "algorithm": "PPO",
        "evaluated_products": int(ppo_summary["evaluated_products"]),
        "mean_reward": float(ppo_summary["mean_reward"]),
        "median_reward": float(ppo_summary["median_reward"]),
        "mean_oracle_reward": float(
            ppo_summary["mean_reward"] + ppo_summary["mean_oracle_reward_gap"]
        ),
        "mean_oracle_gap": float(ppo_summary["mean_oracle_reward_gap"]),
        "oracle_action_match_pct": np.nan,
        "mean_discount_pct": float(ppo_summary["mean_discount_pct"]),
        "reward_ratio_to_oracle_mean_pct": float(
            ppo_summary["reward_ratio_to_oracle_mean_pct"]
        ),
        "source": str(PPO_DETAILS.relative_to(ROOT)),
    })

    for algorithm, path in EVALUATION_FILES.items():
        frame = pd.read_csv(path)

        reward_col = f"{algorithm.lower()}_reward"
        discount_col = f"{algorithm.lower()}_discount"

        required = {
            "source_row",
            reward_col,
            discount_col,
            "oracle_reward",
            "oracle_gap",
            "oracle_action_match",
        }
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")

        if frame["source_row"].duplicated().any():
            raise ValueError(f"Duplicate source_row values in {path.name}")

        if len(frame) != 7139:
            raise ValueError(
                f"Expected 7,139 rows in {path.name}; found {len(frame)}."
            )

        numeric_cols = [
            reward_col, discount_col, "oracle_reward",
            "oracle_gap", "oracle_action_match",
        ]
        if not np.isfinite(frame[numeric_cols].to_numpy(dtype=float)).all():
            raise ValueError(f"Non-finite evaluation values in {path.name}")

        details_frames.append(pd.DataFrame({
            "algorithm": algorithm,
            "split": frame["split"].astype(str),
            "source_row": frame["source_row"].astype("int64"),
            "discount": pd.to_numeric(frame[discount_col], errors="raise"),
            "reward": pd.to_numeric(frame[reward_col], errors="raise"),
            "oracle_reward": pd.to_numeric(
                frame["oracle_reward"], errors="raise"
            ),
            "oracle_gap": pd.to_numeric(frame["oracle_gap"], errors="raise"),
            "oracle_action_match": pd.to_numeric(
                frame["oracle_action_match"], errors="raise"
            ),
        }))

        summary_rows.append({
            "algorithm": algorithm,
            "evaluated_products": len(frame),
            "mean_reward": float(frame[reward_col].mean()),
            "median_reward": float(frame[reward_col].median()),
            "mean_oracle_reward": float(frame["oracle_reward"].mean()),
            "mean_oracle_gap": float(frame["oracle_gap"].mean()),
            "oracle_action_match_pct": float(
                100 * frame["oracle_action_match"].mean()
            ),
            "mean_discount_pct": float(frame[discount_col].mean()),
            "reward_ratio_to_oracle_mean_pct": float(
                100 * frame[reward_col].mean()
                / frame["oracle_reward"].mean()
            ),
            "source": str(path.relative_to(ROOT)),
        })

    details = pd.concat(details_frames, ignore_index=True)
    summary = pd.DataFrame(summary_rows)

    # Validate same held-out source rows across the three newly evaluated
    # algorithms. PPO's row alignment is checked separately against its file.
    reference_rows = set(
        pd.read_csv(EVALUATION_FILES["DDPG"])["source_row"].astype(int)
    )
    for algorithm, path in EVALUATION_FILES.items():
        rows = set(pd.read_csv(path)["source_row"].astype(int))
        if rows != reference_rows:
            raise ValueError(
                f"{algorithm} source rows differ from DDPG's test rows."
            )

    if summary["algorithm"].duplicated().any():
        raise ValueError("Duplicate algorithm summary rows detected.")

    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_SUMMARY, index=False)
    details.to_csv(OUTPUT_DETAILS, index=False)

    ranked = summary.sort_values("mean_reward", ascending=False)
    report_lines = [
        "# Dynamic Pricing Algorithm Comparison",
        "",
        "## Experimental setup",
        "",
        "- Held-out test split: MonthNum 9–10.",
        "- Test size: 7,139 products per algorithm.",
        "- Reward: sales-focused configuration (alpha=0.2, beta=0.8).",
        "- DDPG and SAC continuous outputs are mapped to the nearest supported",
        "  discount in 5–40%.",
        "- DDQN selects among the same eight supported discounts at evaluation.",
        "- The oracle is a per-product retrospective benchmark computed using",
        "  the same counterfactual simulator; it is not market ground truth.",
        "",
        "## Summary",
        "",
        "| Algorithm | Products | Mean reward | Median reward | Mean oracle gap | Oracle match | Mean discount | Reward / oracle |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for _, row in ranked.iterrows():
        match = (
            "N/A"
            if pd.isna(row["oracle_action_match_pct"])
            else f"{row['oracle_action_match_pct']:.2f}%"
        )
        report_lines.append(
            f"| {row['algorithm']} | {int(row['evaluated_products'])} "
            f"| {row['mean_reward']:.6f} | {row['median_reward']:.6f} "
            f"| {row['mean_oracle_gap']:.6f} | {match} "
            f"| {row['mean_discount_pct']:.2f}% "
            f"| {row['reward_ratio_to_oracle_mean_pct']:.2f}% |"
        )

    report_lines.extend([
        "",
        "## Interpretation and limitations",
        "",
        "- These are counterfactual simulator results, not live-market outcomes.",
        "- The environment makes one-step pricing decisions rather than modeling",
        "  a sequential market with feedback over time.",
        "- The economics use a dataset-derived gross-margin proxy.",
        "- DDPG, SAC, and DDQN each use one training seed; uncertainty across",
        "  independent training seeds has not been estimated.",
        "- The DDPG–SAC mean-reward difference should not be treated as a",
        "  meaningful advantage without repeated-seed evaluation.",
        "- PPO summary metrics are taken from the existing baseline summary;",
        "  its per-product file is preserved and adapted in memory.",
        "",
        "## Input artifacts",
        "",
        f"- PPO details: `{PPO_DETAILS.relative_to(ROOT)}`",
        f"- Baseline summary: `{BASELINE_SUMMARY.relative_to(ROOT)}`",
        *[
            f"- {algorithm}: `{path.relative_to(ROOT)}`"
            for algorithm, path in EVALUATION_FILES.items()
        ],
        "",
    ])

    OUTPUT_REPORT.write_text("\n".join(report_lines), encoding="utf-8")

    print("Comparison summary saved:", OUTPUT_SUMMARY)
    print("Per-product details saved:", OUTPUT_DETAILS)
    print("Markdown report saved:", OUTPUT_REPORT)
    print("\nSummary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
