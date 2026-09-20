import pandas as pd
import yaml
import os
import re

def clean_percentage(val):
    if pd.isna(val):
        return None
    val = str(val).replace('%', '').replace(',', '').strip()
    try:
        return float(val)
    except:
        return None

def clean_numeric(val):
    if pd.isna(val):
        return None
    val = str(val).replace(',', '').strip()
    try:
        return float(val)
    except:
        return None

def preprocess_dataset():
    raw_path = "data/raw/products.csv"
    if not os.path.exists(raw_path):
        return
        
    df = pd.read_csv(raw_path, encoding='utf-8', encoding_errors='replace', low_memory=False)
    
    # Clean numeric columns
    numeric_cols = [
        'Main Category BSR Ranking', 'Total Rating Count', 'New Rating Count', 
        'Total Review Count', 'New Review Count', 'Buybox Price (USD)'
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].apply(clean_numeric)
            
    # Clean percentage columns
    pct_cols = [
        'Gross Margin Rate', 'Estimated Monthly Sales Growth Rate',
        'Positive Rating Rate', 'Negative Rating Rate'
    ]
    for col in pct_cols:
        if col in df.columns:
            df[col] = df[col].apply(clean_percentage)
            
    # Normalize Month
    month_map = {
        'January': 1, 'February': 2, 'March': 3, 'April': 4,
        'May': 5, 'June': 6, 'July': 7, 'August': 8,
        'September': 9, 'October': 10, 'November': 11, 'December': 12
    }
    if 'Deal Month' in df.columns:
        df['MonthNum'] = df['Deal Month'].map(month_map)
        
    # Validation filters
    df = df[df['Buybox Price (USD)'] > 0]
    df = df[df['Discount'] >= 0]
    df = df[df['Discount'] <= 100]
    
    # Save full cleaned dataset
    os.makedirs('data/processed', exist_ok=True)
    df.to_parquet('data/processed/products_clean.parquet')
    
    # Split
    with open('configs/data_split.yaml', 'r') as f:
        split_config = yaml.safe_load(f)['split']
        
    train_df = df[df['MonthNum'].isin(split_config['train_months'])]
    val_df = df[df['MonthNum'].isin(split_config['val_months'])]
    test_df = df[df['MonthNum'].isin(split_config['test_months'])]
    
    train_df.to_parquet('data/processed/products_train.parquet')
    val_df.to_parquet('data/processed/products_val.parquet')
    test_df.to_parquet('data/processed/products_test.parquet')
    
    print(f"Train size: {len(train_df)}, Val size: {len(val_df)}, Test size: {len(test_df)}")

if __name__ == "__main__":
    preprocess_dataset()
