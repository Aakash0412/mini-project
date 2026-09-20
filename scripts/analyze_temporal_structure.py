import pandas as pd
import os

def analyze_temporal_structure():
    file_path = "data/raw/products.csv"
    if not os.path.exists(file_path):
        return
        
    df = pd.read_csv(file_path, encoding='utf-8', encoding_errors='replace', low_memory=False)
    
    # We have Deal Month (January-October) and Listing Time.
    # The paper explicitly states dataset is from January-October 2023.
    # The Deal Month is the chronological feature we should use.
    
    # Map months to numbers
    month_map = {
        'January': 1, 'February': 2, 'March': 3, 'April': 4,
        'May': 5, 'June': 6, 'July': 7, 'August': 8,
        'September': 9, 'October': 10, 'November': 11, 'December': 12
    }
    
    if 'Deal Month' in df.columns:
        df['MonthNum'] = df['Deal Month'].map(month_map)
        
        counts_per_month = df['MonthNum'].value_counts().sort_index()
        
        # Unique products per month
        products_per_month = df.groupby('MonthNum')['ASIN'].nunique()
        
        # Check how many months a product appears
        months_per_product = df.groupby('ASIN')['MonthNum'].nunique()
        
        with open("reports/temporal_analysis.md", "w") as f:
            f.write("# Temporal Structure Analysis\n\n")
            f.write("## Observations per Month\n")
            f.write(counts_per_month.to_string() + "\n\n")
            f.write("## Unique Products per Month\n")
            f.write(products_per_month.to_string() + "\n\n")
            f.write("## Month Count Distribution per Product\n")
            f.write(months_per_product.value_counts().sort_index().to_string() + "\n")
            
        temporal_df = pd.DataFrame({
            'Total_Observations': counts_per_month,
            'Unique_Products': products_per_month
        })
        temporal_df.to_csv("reports/temporal_analysis.csv")

if __name__ == "__main__":
    analyze_temporal_structure()
