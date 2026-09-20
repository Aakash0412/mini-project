import os
import pandas as pd
import pytest

def test_raw_csv_exists():
    assert os.path.exists("data/raw/products.csv")

def test_processed_files_exist():
    assert os.path.exists("data/processed/products_clean.parquet")
    assert os.path.exists("data/processed/products_train.parquet")
    assert os.path.exists("data/processed/products_val.parquet")
    assert os.path.exists("data/processed/products_test.parquet")

def test_numeric_conversion():
    df = pd.read_parquet("data/processed/products_clean.parquet")
    assert pd.api.types.is_numeric_dtype(df['Buybox Price (USD)'])
    assert pd.api.types.is_numeric_dtype(df['Total Review Count'])
    assert pd.api.types.is_numeric_dtype(df['Gross Margin Rate'])

def test_discount_validation():
    df = pd.read_parquet("data/processed/products_clean.parquet")
    assert df['Discount'].min() >= 0
    assert df['Discount'].max() <= 100

def test_price_validation():
    df = pd.read_parquet("data/processed/products_clean.parquet")
    assert df['Buybox Price (USD)'].min() > 0

def test_leakage_and_chronology():
    train_df = pd.read_parquet("data/processed/products_train.parquet")
    val_df = pd.read_parquet("data/processed/products_val.parquet")
    test_df = pd.read_parquet("data/processed/products_test.parquet")
    
    assert train_df['MonthNum'].max() < val_df['MonthNum'].min()
    assert val_df['MonthNum'].max() < test_df['MonthNum'].min()
