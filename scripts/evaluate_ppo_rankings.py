
import numpy as np
import pandas as pd

EVAL_PATH = (
    "models/ppo_transfer/"
    "transfer_supported_8action_policy_evaluation.csv"
)
PRODUCTS_PATH = "data/processed/products_clean.parquet"
STATE_INDICES_PATH = "data/processed/state/state_row_indices.npy"


def main():
    evaluation = pd.read_csv(EVAL_PATH)
    products = pd.read_parquet(PRODUCTS_PATH)
    state_indices = np.load(STATE_INDICES_PATH)

    test = (
        evaluation.loc[evaluation["split"].eq("test")]
        .copy()
        .reset_index(drop=True)
    )

    print("=" * 78)
    print("PPO RANKING EVALUATION")
    print("=" * 78)
    print(f"Held-out test rows: {len(test)}")
    print(f"Cleaned products rows: {len(products)}")
    print(f"State rows: {len(state_indices)}")

    # source_row represents the row position in the cleaned product
    # dataframe. Validate before attaching identifiers.
    source_rows = test["source_row"].to_numpy(dtype=int)

    if np.any(source_rows < 0) or np.any(source_rows >= len(products)):
        raise ValueError("source_row contains invalid product row positions.")

    if test["source_row"].duplicated().any():
        raise ValueError("Duplicate source_row values in test evaluation.")

    # Attach product identifiers using the original cleaned-data row
    # positions. Do not use test dataframe row numbers for this mapping.
    product_info = products.iloc[source_rows].reset_index(drop=True)

    test["ASIN"] = product_info["ASIN"].to_numpy()
    test["PASIN"] = product_info["PASIN"].to_numpy()
    test["Title"] = product_info["Title"].to_numpy()
    test["Deal Month"] = product_info["Deal Month"].to_numpy()
    test["MonthNum"] = product_info["MonthNum"].to_numpy()

    # Check the source-row mapping against the state-row mapping.
    state_row_lookup = {
        int(original_row): state_position
        for state_position, original_row in enumerate(state_indices)
    }
    mapped = test["source_row"].isin(state_row_lookup)

    print(f"Test rows mapped to state rows: {mapped.sum()} / {len(test)}")
    print(f"Unique ASINs in test set: {test['ASIN'].nunique(dropna=True)}")

    # Local simulated uplift definitions.
    test["sales_uplift"] = test["ppo_sales"] - test["baseline_sales"]
    test["margin_uplift"] = test["ppo_margin"] - test["baseline_margin"]

    test["sales_relative_uplift_pct"] = (
        test["sales_uplift"]
        / test["baseline_sales"].replace(0, np.nan)
        * 100
    )
    test["margin_relative_uplift_pct"] = (
        test["margin_uplift"]
        / test["baseline_margin"].replace(0, np.nan)
        * 100
    )

    def report_top_k(label, score_col, k, outcome_col, baseline_col):
        # This ranks test observations, not necessarily distinct ASINs.
        ranked = test.sort_values(score_col, ascending=False).head(k)
        baseline_total = ranked[baseline_col].sum()
        outcome_total = ranked[outcome_col].sum()
        total_uplift = outcome_total - baseline_total

        ratio_change = (
            (outcome_total / baseline_total - 1) * 100
            if baseline_total != 0
            else np.nan
        )

        print(f"\n{label}: Top {k} test observations")
        print(f"  Baseline total: {baseline_total:,.4f}")
        print(f"  PPO simulated total: {outcome_total:,.4f}")
        print(f"  Total simulated uplift: {total_uplift:,.4f}")
        print(f"  Ratio-of-totals change: {ratio_change:.4f}%")
        print(f"  Distinct ASINs: {ranked['ASIN'].nunique(dropna=True)}")
        print("  Ranked products:")

        display_cols = [
            "ASIN", "Title", "Deal Month", "MonthNum",
            "observed_discount", "ppo_discount",
            score_col, outcome_col, baseline_col,
        ]
        print(ranked[display_cols].to_string(index=False))

    print("\n" + "=" * 78)
    print("1. TOP PRODUCTS BY SIMULATED MARGIN UPLIFT")
    print("=" * 78)

    report_top_k(
        "Margin ranking",
        "margin_uplift",
        1,
        "ppo_margin",
        "baseline_margin",
    )
    report_top_k(
        "Margin ranking",
        "margin_uplift",
        3,
        "ppo_margin",
        "baseline_margin",
    )

    print("\n" + "=" * 78)
    print("2. TOP PRODUCTS BY SIMULATED SALES UPLIFT")
    print("=" * 78)

    report_top_k(
        "Sales ranking",
        "sales_uplift",
        1,
        "ppo_sales",
        "baseline_sales",
    )
    report_top_k(
        "Sales ranking",
        "sales_uplift",
        3,
        "ppo_sales",
        "baseline_sales",
    )

    print("\n" + "=" * 78)
    print("3. TOP 1% AND TOP 3% OF TEST OBSERVATIONS")
    print("=" * 78)
    print(
        "These percentile-based summaries are provided as a separate "
        "local ranking analysis. They are not assumed to match the "
        "base paper's Top-1/Top-3 definitions."
    )

    for metric, score_col, ppo_col, base_col in [
        ("Margin", "margin_uplift", "ppo_margin", "baseline_margin"),
        ("Sales", "sales_uplift", "ppo_sales", "baseline_sales"),
    ]:
        ranked = test.sort_values(score_col, ascending=False)

        print(f"\n{metric} ranking:")
        for pct in [1, 3]:
            k = max(1, int(np.ceil(len(ranked) * pct / 100)))
            top = ranked.head(k)
            base_total = top[base_col].sum()
            ppo_total = top[ppo_col].sum()
            change = (
                (ppo_total / base_total - 1) * 100
                if base_total != 0
                else np.nan
            )

            print(
                f"  Top {pct}% (n={k}): "
                f"baseline={base_total:,.4f}, "
                f"PPO={ppo_total:,.4f}, "
                f"ratio-of-totals change={change:.4f}%, "
                f"distinct ASINs={top['ASIN'].nunique(dropna=True)}"
            )

    print("\n" + "=" * 78)
    print("INTERPRETATION NOTES")
    print("=" * 78)
    print(
        "- Rankings use simulated uplift against the stored historical "
        "baseline, not observed causal treatment effects."
    )
    print(
        "- Top-k observations can contain repeated ASINs across months."
    )
    print(
        "- These metrics are not directly comparable with the paper "
        "until its exact Top-1/Top-3 definitions are confirmed."
    )
    print(
        "- Existing evaluation files and model checkpoints were not modified."
    )


if __name__ == "__main__":
    main()
