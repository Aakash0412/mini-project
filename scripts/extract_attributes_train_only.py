from pathlib import Path
import json

import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(".")
DATA_PATH = ROOT / "data/processed/products_clean.parquet"
CONFIG_PATH = ROOT / "configs/data_split.yaml"
OUTPUT_DIR = ROOT / "data/processed/features_train_only"
OUTPUT_PATH = OUTPUT_DIR / "attribute_features_train_only.npy"
INDEX_PATH = OUTPUT_DIR / "feature_index.parquet"
REPORT_PATH = OUTPUT_DIR / "attribute_preprocessing_report.json"

NUMERIC_COLS = [
    "MonthNum", "Total Rating Count", "New Rating Count",
    "Total Review Count", "New Review Count", "Rating Score",
    "Positive Rating Rate", "Negative Rating Rate",
    "Buybox Price (USD)", "Weight (pounds)",
    "Volume (cubic inches)", "Number of Variants",
    "Number of Sellers", "FBA Shipping Fee (USD)",
    "Gross Margin Rate", "Main Category BSR Ranking",
    "Main Category BSR Change Number", "Main Category BSR Change Rate",
]
CATEGORICAL_COLS = ["Main Category BSR Category", "Buybox Seller Type"]
EXPECTED_DIM = 47


def main():
    df = pd.read_parquet(DATA_PATH).reset_index(drop=True)
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        split = yaml.safe_load(f)["split"]

    train_mask = df["MonthNum"].isin(split["train_months"]).to_numpy()
    if not train_mask.any():
        raise ValueError("No training rows found from configured months.")

    missing = [
        c for c in NUMERIC_COLS + CATEGORICAL_COLS
        if c not in df.columns
    ]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    work = df[NUMERIC_COLS + CATEGORICAL_COLS].copy()
    for col in NUMERIC_COLS:
        work[col] = pd.to_numeric(
            work[col].astype(str).str.replace(",", "", regex=False)
            .str.replace("%", "", regex=False).str.strip(),
            errors="coerce",
        )
        work.loc[df[col].isna(), col] = np.nan

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
        )),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, NUMERIC_COLS),
        ("cat", categorical_pipeline, CATEGORICAL_COLS),
    ])

    # Fit only on training months; transform every row with that fitted pipeline.
    preprocessor.fit(work.loc[train_mask])
    features = preprocessor.transform(work).astype(np.float32)

    if features.shape[1] != EXPECTED_DIM:
        raise ValueError(
            f"Training-fitted preprocessing produced {features.shape[1]} "
            f"dimensions, but the existing state expects {EXPECTED_DIM}. "
            "No projection or existing-file overwrite was performed."
        )
    if not np.isfinite(features).all():
        raise ValueError("Generated attributes contain NaN or Inf.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT_PATH, features)
    df[["ASIN"]].to_parquet(INDEX_PATH, index=False)

    report = {
        "rows": int(len(df)),
        "training_rows": int(train_mask.sum()),
        "training_months": split["train_months"],
        "validation_months": split["val_months"],
        "test_months": split["test_months"],
        "feature_dimension": int(features.shape[1]),
        "expected_dimension": EXPECTED_DIM,
        "fit_scope": "training months only",
        "transform_scope": "all rows using training-fitted transformations",
        "projection_used": False,
        "finite_values": True,
        "feature_path": str(OUTPUT_PATH),
        "index_path": str(INDEX_PATH),
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
