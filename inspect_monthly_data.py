import pandas as pd
from pathlib import Path

path = Path("data/processed/products_clean.parquet")
df = pd.read_parquet(path)

print("DATASET SHAPE:", df.shape)
print("\nCOLUMNS:")
for column in df.columns:
    print(f"  {column!r}  dtype={df[column].dtype}")

print("\nPOSSIBLE PRODUCT IDENTIFIERS:")
for column in df.columns:
    name = str(column).lower()
    if any(key in name for key in ["asin", "product_id", "product id", "item_id", "sku"]):
        print(
            f"{column!r}: unique={df[column].nunique(dropna=True)}, "
            f"missing={df[column].isna().sum()}"
        )

print("\nMONTH DISTRIBUTION:")
if "MonthNum" in df.columns:
    print(df["MonthNum"].value_counts(dropna=False).sort_index().to_string())
else:
    print("No exact 'MonthNum' column found.")

print("\nPOSSIBLE PRODUCT-MONTH DUPLICATES:")
id_candidates = [
    c for c in df.columns
    if any(key in str(c).lower() for key in ["asin", "product_id", "product id", "item_id", "sku"])
]
if id_candidates and "MonthNum" in df.columns:
    for column in id_candidates:
        pairs = df.groupby([column, "MonthNum"], dropna=False).size()
        print(
            f"{column!r}: duplicate product-month groups="
            f"{int((pairs > 1).sum())}"
        )
else:
    print("Cannot check product-month duplicates until the identifier is identified.")
