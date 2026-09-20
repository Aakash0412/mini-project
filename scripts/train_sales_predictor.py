import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
import json
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class SalesPredictorMLP(nn.Module):
    def __init__(self, input_dim):
        super(SalesPredictorMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        return self.net(x).squeeze(-1)

def train_sales_predictor():
    print("Training Sales Predictor...")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # Load state and raw data
    state = np.load("data/processed/state/state.npy")
    index = pd.read_parquet("data/processed/features/feature_index.parquet")
    df = pd.read_parquet("data/processed/products_clean.parquet").reset_index(drop=True)
    
    # Ensure alignment
    assert len(state) == len(df)
    
    # Target
    target = df['Estimated Monthly Sales Growth Rate'].fillna(0).values
    
    # Action (Discount) - included as feature
    discount = df['Discount'].fillna(0).values.reshape(-1, 1)
    
    # Full Input: [State, Action]
    X = np.concatenate([state, discount], axis=1)
    
    # Splits
    train_mask = df['MonthNum'].isin([1, 2, 3, 4, 5, 6]).values
    val_mask = df['MonthNum'].isin([7, 8]).values
    test_mask = df['MonthNum'].isin([9, 10]).values
    
    X_train, y_train = X[train_mask], target[train_mask]
    X_val, y_val = X[val_mask], target[val_mask]
    X_test, y_test = X[test_mask], target[test_mask]
    
    # Convert to tensors
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)
    
    model = SalesPredictorMLP(input_dim=X.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    epochs = 20
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            pred = model(bx)
            loss = criterion(pred, by)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * bx.size(0)
            
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                pred = model(bx)
                val_loss += criterion(pred, by).item() * bx.size(0)
                
        val_loss /= len(val_dataset)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs("models/sales_predictor", exist_ok=True)
            torch.save(model.state_dict(), "models/sales_predictor/best_model.pth")
            
    # Load best model for evaluation
    model.load_state_dict(torch.load("models/sales_predictor/best_model.pth"))
    model.eval()
    
    test_preds = []
    test_actuals = []
    with torch.no_grad():
        for bx, by in test_loader:
            bx = bx.to(device)
            pred = model(bx).cpu().numpy()
            test_preds.extend(pred)
            test_actuals.extend(by.numpy())
            
    mae = mean_absolute_error(test_actuals, test_preds)
    rmse = np.sqrt(mean_squared_error(test_actuals, test_preds))
    r2 = r2_score(test_actuals, test_preds)
    
    metrics = {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "Test_Samples": len(test_actuals)
    }
    
    print(f"Test Metrics: {metrics}")
    
    os.makedirs("reports", exist_ok=True)
    with open("reports/sales_predictor_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    print("Sales Predictor trained, evaluated, and frozen.")

if __name__ == "__main__":
    train_sales_predictor()
