from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "processed" / "products_clean.parquet"

KEYS = ["ASIN", "MonthNum"]
TARGET = "Estimated Monthly Sales Growth Rate"
DISCOUNT = "Discount"

CHECK_COLUMNS = [
    "Deal Month",
    "Buybox Price (USD)",
    "Discount",
    "Last Monthly Sales",
    "Estimated Monthly Sales Growth Rate",
    "Gross Margin Rate",
    "Total Rating Count",
    "New Rating Count",
    "Total Review Count",
    "New Review Count",
    "Rating Score",
    "Listing Duration",
]


def section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def main():
    if not DATA_FILE.exists():
        print(f"ERROR: Dataset not found:\n{DATA_FILE}")
        print("Run this script from your project after checking the dataset path.")
        sys.exit(1)

    print(f"Dataset: {DATA_FILE}")
    df = pd.read_parquet(DATA_FILE)

    section("1. DATASET OVERVIEW")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    print(f"Unique ASINs: {df['ASIN'].nunique(dropna=True):,}")
    print(f"Unique ASIN-month pairs: {df[KEYS].drop_duplicates().shape[0]:,}")
    print(f"Rows with missing ASIN: {df['ASIN'].isna().sum():,}")
    print(f"Rows with missing MonthNum: {df['MonthNum'].isna().sum():,}")

    section("2. MONTHLY ROW COUNTS")
    print(df.groupby("MonthNum", dropna=False).size().to_string())

    section("3. DUPLICATE ASIN-MONTH GROUPS")
    group_sizes = df.groupby(KEYS, dropna=False).size()
    duplicate_groups = group_sizes[group_sizes > 1]

    print(f"ASIN-month groups with multiple rows: {len(duplicate_groups):,}")
    print(
        "Rows beyond the first row in each ASIN-month group: "
        f"{int((duplicate_groups - 1).sum()):,}"
    )

    if duplicate_groups.empty:
        print("No duplicate ASIN-month groups found.")
    else:
        print("\nLargest duplicate group sizes:")
        print(duplicate_groups.sort_values(ascending=False).head(15).to_string())

        print("\nSample duplicate groups:")
        sample_keys = duplicate_groups.head(10).index
        sample_index = pd.MultiIndex.from_frame(df[KEYS])
        sample = df.loc[sample_index.isin(sample_keys)]

        cols = [c for c in KEYS + CHECK_COLUMNS if c in df.columns]
        print(sample[cols].sort_values(KEYS).to_string(index=False, max_rows=100))

    section("4. WITHIN-GROUP VARIATION")
    available = [c for c in CHECK_COLUMNS if c in df.columns]

    if duplicate_groups.empty:
        print("Not applicable: no duplicate groups.")
    else:
        duplicated_rows = df.set_index(KEYS).loc[
            df.set_index(KEYS).index.isin(duplicate_groups.index)
        ].reset_index()

        variation = duplicated_rows.groupby(KEYS, dropna=False)[available].nunique(
            dropna=False
        )

        for col in available:
            changed = int((variation[col] > 1).sum())
            print(
                f"{col}: varies in {changed:,} of "
                f"{len(variation):,} duplicate groups"
            )

        print("\nGroups with differing discounts:")
        if DISCOUNT in variation.columns:
            differing_discount = variation[variation[DISCOUNT] > 1]
            print(f"Count: {len(differing_discount):,}")
            if not differing_discount.empty:
                print(differing_discount.head(15).to_string())

        print("\nGroups with differing sales-growth targets:")
        if TARGET in variation.columns:
            differing_target = variation[variation[TARGET] > 1]
            print(f"Count: {len(differing_target):,}")
            if not differing_target.empty:
                print(differing_target.head(15).to_string())

    section("5. DISCOUNT SUPPORT")
    if DISCOUNT in df.columns:
        print("Discount values:")
        print(df[DISCOUNT].value_counts(dropna=False).sort_index().to_string())

        print(
            "\nRows with zero discount: "
            f"{int((df[DISCOUNT] == 0).sum()):,}"
        )
        print(
            "Rows with missing discount: "
            f"{int(df[DISCOUNT].isna().sum()):,}"
        )

    section("6. SALES-GROWTH TARGET")
    if TARGET in df.columns:
        target = pd.to_numeric(df[TARGET], errors="coerce")
        print(target.describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.95, 0.99]))
        print(f"Missing or nonnumeric values: {target.isna().sum():,}")
        print(f"Negative values: {(target < 0).sum():,}")
        print(f"Values below -100: {(target < -100).sum():,}")
        print(f"Values above 1000: {(target > 1000).sum():,}")

        if DISCOUNT in df.columns:
            valid = df[TARGET].notna() & df[DISCOUNT].notna()
            paired = df.loc[valid, [TARGET, DISCOUNT]].drop_duplicates()
            target_discount_counts = paired.groupby(TARGET)[DISCOUNT].nunique()
            print(
                "Distinct targets associated with multiple discounts: "
                f"{int((target_discount_counts > 1).sum()):,} "
                "distinct target values"
            )

    section("7. TEMPORAL CONTINUITY")
    valid = df.dropna(subset=KEYS).copy()
    valid["MonthNum"] = pd.to_numeric(valid["MonthNum"], errors="coerce")
    valid = valid.dropna(subset=["MonthNum"])

    month_sets = valid.groupby("ASIN")["MonthNum"].agg(
        lambda values: sorted(set(values.astype(int)))
    )

    gap_count = 0
    gap_examples = []

    for asin, months in month_sets.items():
        gaps = [
            (months[i], months[i + 1])
            for i in range(len(months) - 1)
            if months[i + 1] - months[i] > 1
        ]
        if gaps:
            gap_count += len(gaps)
            if len(gap_examples) < 10:
                gap_examples.append((asin, months, gaps))

    print(f"ASINs with observations in multiple months: "
          f"{sum(len(months) > 1 for months in month_sets):,}")
    print(f"ASINs with observations in only one month: "
          f"{sum(len(months) == 1 for months in month_sets):,}")
    print(f"Month gaps across ASIN timelines: {gap_count:,}")

    if gap_examples:
        print("\nSample timelines with month gaps:")
        for asin, months, gaps in gap_examples:
            print(f"ASIN={asin} | months={months} | gaps={gaps}")

    section("8. REQUIRED COLUMNS")
    required = ["ASIN", "MonthNum", TARGET, DISCOUNT]
    for col in required:
        print(f"{col}: {'PRESENT' if col in df.columns else 'MISSING'}")

    section("AUDIT COMPLETE")
    print("No dataset or model files were modified.")
    print("Use these findings to determine how duplicate rows should be handled.")


if __name__ == "__main__":
    main()