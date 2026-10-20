import pandas as pd
from pathlib import Path

path = Path("data/processed/products_clean.parquet")
df = pd.read_parquet(path)

print("=" * 75)
print("PRODUCT-MONTH DUPLICATE AUDIT")
print("=" * 75)

print("\n1. EXACT DUPLICATE ROWS")
print("Exact duplicate rows:", int(df.duplicated().sum()))

pair = df.groupby(["ASIN", "MonthNum"], dropna=False)
pair_sizes = pair.size()
duplicate_pairs = pair_sizes[pair_sizes > 1]

print("\n2. ASIN + MONTH")
print("Unique ASIN-month pairs:", len(pair_sizes))
print("Pairs with multiple rows:", len(duplicate_pairs))
print("Rows belonging to repeated pairs:", int(duplicate_pairs.sum()))
print("Largest rows in one ASIN-month:", int(pair_sizes.max()))
print("\nRepeated-pair size distribution:")
print(duplicate_pairs.value_counts().sort_index().to_string())

print("\n3. COUNTRY-SITE CHECK")
country_counts = pair["Country Site"].nunique(dropna=False)
print("ASIN-month pairs spanning multiple country sites:",
      int((country_counts > 1).sum()))

site_groups = df.groupby(
    ["ASIN", "MonthNum", "Country Site"], dropna=False
).size()
duplicate_site_groups = site_groups[site_groups > 1]

print("ASIN-month-country groups with multiple rows:",
      len(duplicate_site_groups))
print("Rows in those groups:", int(duplicate_site_groups.sum()))

print("\n4. VARIATION WITHIN REPEATED ASIN-MONTH GROUPS")
check_columns = [
    "Country Site",
    "PASIN",
    "Buybox Price (USD)",
    "Discount",
    "Last Monthly Sales",
    "Estimated Monthly Sales Growth Rate",
    "Gross Margin Rate",
    "Main Image URL",
]

for column in check_columns:
    if column in df.columns:
        variation = pair[column].nunique(dropna=False)
        print(f"{column}: {int((variation > 1).sum())} groups have multiple values")

print("\n5. SAMPLE REPEATED ASIN-MONTH GROUPS")
if len(duplicate_pairs):
    sample_keys = duplicate_pairs.sort_values(ascending=False).head(5).index
    sample_keys = set(sample_keys)
    mask = [
        (row.ASIN, row.MonthNum) in sample_keys
        for row in df[["ASIN", "MonthNum"]].itertuples(index=False)
    ]

    sample_columns = [
        "ASIN", "MonthNum", "Country Site", "PASIN",
        "Buybox Price (USD)", "Discount", "Last Monthly Sales",
        "Estimated Monthly Sales Growth Rate", "Gross Margin Rate"
    ]
    sample_columns = [c for c in sample_columns if c in df.columns]
    print(df.loc[mask, sample_columns].sort_values(
        ["ASIN", "MonthNum", "Country Site"]
    ).to_string(index=False, max_rows=100))
else:
    print("No repeated ASIN-month groups.")

print("\n6. PRODUCT HISTORY COVERAGE")
months_per_asin = df.groupby("ASIN")["MonthNum"].nunique()
print("ASINs:", len(months_per_asin))
print("ASINs observed in 1 month:", int((months_per_asin == 1).sum()))
print("ASINs observed in 2+ months:", int((months_per_asin >= 2).sum()))
print("ASINs observed in 6+ months:", int((months_per_asin >= 6).sum()))
print("ASINs observed in all 10 months:", int((months_per_asin == 10).sum()))

print("\nAudit complete. No files were modified.")
