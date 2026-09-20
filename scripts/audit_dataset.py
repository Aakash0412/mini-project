import pandas as pd
import json
import os

def audit_dataset():
    file_path = "data/raw/products.csv"
    if not os.path.exists(file_path):
        print(f"Error: {file_path} does not exist.")
        return
        
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding='utf-8', encoding_errors='replace', low_memory=False)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return
    
    audit_results = {
        "num_rows": len(df),
        "num_columns": len(df.columns),
        "columns": list(df.columns),
        "data_types": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_values": df.isnull().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_product_identifiers": None,
        "unique_products": None,
        "unique_categories": None,
        "date_time_range": None,
        "price_range": None,
        "discount_range": None,
        "columns_summary": {}
    }

    # Try to find ID column (like ASIN or ID)
    id_col = None
    for col in df.columns:
        if 'id' in col.lower() or 'asin' in col.lower():
            id_col = col
            break
            
    if id_col:
        audit_results["duplicate_product_identifiers"] = int(df.duplicated(subset=[id_col]).sum())
        audit_results["unique_products"] = df[id_col].nunique()
    
    # Try to find category column
    cat_col = None
    for col in df.columns:
        if 'category' in col.lower():
            cat_col = col
            break
    if cat_col:
        audit_results["unique_categories"] = df[cat_col].nunique()
        
    # Try to find date/time column
    date_col = None
    for col in df.columns:
        if 'date' in col.lower() or 'time' in col.lower() or 'month' in col.lower():
            date_col = col
            break
    if date_col:
        try:
            dates = pd.to_datetime(df[date_col])
            audit_results["date_time_range"] = [str(dates.min()), str(dates.max())]
        except:
            audit_results["date_time_range"] = [str(df[date_col].min()), str(df[date_col].max())]
            
    # Try to find price column
    price_col = None
    for col in df.columns:
        if 'price' in col.lower() and 'discount' not in col.lower() and 'historical' not in col.lower():
            price_col = col
            break
    if price_col and pd.api.types.is_numeric_dtype(df[price_col]):
        audit_results["price_range"] = [float(df[price_col].min()), float(df[price_col].max())]

    # Try to find discount column
    discount_col = None
    for col in df.columns:
        if 'discount' in col.lower() or 'promotion' in col.lower():
            discount_col = col
            break
    if discount_col and pd.api.types.is_numeric_dtype(df[discount_col]):
        audit_results["discount_range"] = [float(df[discount_col].min()), float(df[discount_col].max())]

    for col in df.columns:
        audit_results["columns_summary"][col] = {
            "type": str(df[col].dtype),
            "missing_pct": round(df[col].isnull().sum() / len(df) * 100, 2),
            "unique_values": df[col].nunique(),
            "example_values": [str(x) for x in df[col].dropna().head(3).tolist()]
        }
        
    with open("reports/phase_2_dataset_schema.json", "w") as f:
        json.dump(audit_results, f, indent=4)
        
    # Generate Markdown Report
    with open("reports/phase_2_dataset_audit.md", "w") as f:
        f.write("# Dataset Audit Report\n\n")
        f.write(f"- **Number of rows:** {audit_results['num_rows']}\n")
        f.write(f"- **Number of columns:** {audit_results['num_columns']}\n")
        f.write(f"- **Duplicate rows:** {audit_results['duplicate_rows']}\n")
        if id_col:
            f.write(f"- **Unique products ({id_col}):** {audit_results['unique_products']}\n")
            f.write(f"- **Duplicate product identifiers:** {audit_results['duplicate_product_identifiers']}\n")
        if cat_col:
            f.write(f"- **Unique categories ({cat_col}):** {audit_results['unique_categories']}\n")
        if date_col:
            f.write(f"- **Date/time range ({date_col}):** {audit_results['date_time_range']}\n")
        if price_col:
            f.write(f"- **Price range ({price_col}):** {audit_results['price_range']}\n")
        if discount_col:
            f.write(f"- **Discount range ({discount_col}):** {audit_results['discount_range']}\n")
            
        f.write("\n## Columns Summary\n")
        f.write("| Column | Type | Missing % | Unique values | Example values |\n")
        f.write("|---|---|---|---|---|\n")
        for col, info in audit_results["columns_summary"].items():
            examples = ", ".join(info["example_values"])
            f.write(f"| {col} | {info['type']} | {info['missing_pct']}% | {info['unique_values']} | {examples} |\n")
            
if __name__ == "__main__":
    audit_dataset()
