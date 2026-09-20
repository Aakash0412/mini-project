import os
import torch
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import yaml

def clean_numeric_str(val):
    if pd.isna(val):
        return np.nan
    val = str(val).replace(',', '').replace('%', '').strip()
    try:
        return float(val)
    except:
        return np.nan

def extract_attributes(df):
    numeric_cols = [
        'MonthNum', 'Total Rating Count', 'New Rating Count', 'Total Review Count',
        'New Review Count', 'Rating Score', 'Positive Rating Rate', 'Negative Rating Rate',
        'Buybox Price (USD)', 'Weight (pounds)', 'Volume (cubic inches)', 'Number of Variants',
        'Number of Sellers', 'FBA Shipping Fee (USD)', 'Gross Margin Rate',
        'Main Category BSR Ranking', 'Main Category BSR Change Number', 'Main Category BSR Change Rate'
    ]
    
    categorical_cols = [
        'Main Category BSR Category', 'Buybox Seller Type'
    ]
    
    # Pre-clean the numeric string columns that might not be parsed fully
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].apply(clean_numeric_str), errors='coerce')
            
    # Imputers and Scalers
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
    
    # Fit and transform
    attributes_array = preprocessor.fit_transform(df)
    
    # Check shape
    current_dim = attributes_array.shape[1]
    expected_dim = 47
    
    if current_dim != expected_dim:
        print(f"Warning: Extracted {current_dim} dims, expected {expected_dim}. Projecting.")
        # Simple linear projection using a fixed random matrix to match 47 if mismatch
        np.random.seed(42)
        proj_matrix = np.random.randn(current_dim, expected_dim) / np.sqrt(current_dim)
        attributes_array = np.dot(attributes_array, proj_matrix)
        
    return attributes_array

def extract_text(df, device):
    print("Loading sentence transformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    
    titles = df['Title'].fillna("").astype(str).tolist()
    
    print("Extracting title embeddings...")
    # Using batch size of 16 as per memory-safe rules
    embeddings = model.encode(titles, batch_size=16, show_progress_bar=True, convert_to_tensor=True)
    
    # Native dim is 384, we project to 32
    print("Projecting text embeddings to 32-D...")
    native_dim = embeddings.shape[1]
    target_dim = 32
    
    # Fixed random projection to preserve deterministic behavior and remove API/training need
    torch.manual_seed(42)
    proj_matrix = torch.randn(native_dim, target_dim, device=device) / (native_dim ** 0.5)
    
    projected_embeddings = torch.matmul(embeddings.float(), proj_matrix)
    
    # Move to CPU and numpy
    return projected_embeddings.cpu().numpy()

def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    
    df = pd.read_parquet("data/processed/products_clean.parquet")
    
    os.makedirs("data/processed/features", exist_ok=True)
    
    print("Extracting attributes...")
    attrs = extract_attributes(df)
    np.save("data/processed/features/attribute_features.npy", attrs)
    print(f"Saved attribute features of shape {attrs.shape}")
    
    print("Extracting text features...")
    text_feats = extract_text(df, device)
    np.save("data/processed/features/text_features.npy", text_feats)
    print(f"Saved text features of shape {text_feats.shape}")
    
    # Create an index
    index_df = df[['ASIN']].copy()
    index_df = index_df.reset_index(drop=True)
    index_df.to_parquet("data/processed/features/feature_index.parquet")
    print("Saved feature index")

if __name__ == "__main__":
    main()
