from pathlib import Path
import json
import re
import pandas as pd

DATA_PATH = Path("data/processed/products_clean.parquet")

GROUPS = {
    "identity_and_time": [
        r"\basin\b", r"product.?id", r"\bpasin\b", r"deal month", r"monthnum", r"\bdate\b", r"\bmonth\b"
    ],
    "price_and_discount": [
        r"buybox price", r"price", r"discount", r"promotion", r"reference price", r"original price"
    ],
    "sales_and_growth": [
        r"monthly sales", r"sales growth", r"growth rate", r"sales volume", r"quantity", r"\bsales\b"
    ],
    "profit_and_costs": [
        r"gross margin", r"profit", r"product cost", r"shipping", r"fixed cost", r"variable cost",
        r"cost ratio", r"cost"
    ],
    "ratings_and_reviews": [
        r"rating", r"review", r"sentiment", r"positive", r"negative"
    ],
    "listing_and_market_dynamics": [
        r"listing", r"duration", r"age", r"bsr", r"ranking", r"competitor", r"historical action"
    ],
}

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {DATA_PATH}. Run this script from the project root."
        )

    df = pd.read_parquet(DATA_PATH)
    columns = list(df.columns)

    report = {
        "file": str(DATA_PATH),
        "rows": int(len(df)),
        "column_count": int(len(columns)),
        "columns": columns,
        "groups": {},
    }

    for group, patterns in GROUPS.items():
        matches = []
        for col in columns:
            if any(re.search(pattern, str(col), flags=re.IGNORECASE) for pattern in patterns):
                series = df[col]
                item = {
                    "column": str(col),
                    "dtype": str(series.dtype),
                    "non_null": int(series.notna().sum()),
                    "missing_percent": round(float(series.isna().mean() * 100), 3),
                    "unique_non_null": int(series.nunique(dropna=True)),
                }
                if pd.api.types.is_numeric_dtype(series) and series.notna().any():
                    vals = pd.to_numeric(series, errors="coerce").dropna()
                    item["min"] = float(vals.min())
                    item["median"] = float(vals.median())
                    item["max"] = float(vals.max())
                matches.append(item)
        report["groups"][group] = matches

    # Basic temporal/identity diagnostics where matching columns exist.
    asin_col = next((c for c in columns if str(c).strip().lower() == "asin"), None)
    month_col = next((c for c in columns if str(c).strip().lower() == "monthnum"), None)
    if asin_col:
        report["asin_diagnostics"] = {
            "unique_asin": int(df[asin_col].nunique(dropna=True)),
            "missing_asin": int(df[asin_col].isna().sum()),
        }
        if month_col:
            report["asin_diagnostics"]["unique_asin_month_pairs"] = int(
                df[[asin_col, month_col]].dropna().drop_duplicates().shape[0]
            )
            report["asin_diagnostics"]["duplicate_rows_beyond_first_per_asin_month"] = int(
                df.duplicated([asin_col, month_col], keep="first").sum()
            )
    if month_col:
        report["month_counts"] = {
            str(k): int(v) for k, v in
            df[month_col].value_counts(dropna=False).sort_index().items()
        }

    print("READ-ONLY PAPER DATA FEASIBILITY AUDIT")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print("\nNo files were written or modified.")

if __name__ == "__main__":
    main()
