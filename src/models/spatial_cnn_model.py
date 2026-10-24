# #!/usr/bin/env python3
# """
# spatial_cnn_model.py
# ====================
# A pure PyTorch implementation of the "Deep multivariable spatial attention CNN" 
# methodology from the paper.

# Architecture:
# 1. Input Attention: Dynamically learns which weather variables (channels) matter.
# 2. Spatial Convolution: Compresses the 3x3 bounding box into a 1x1 vector.
# 3. Temporal LSTM: Learns the sequence of weather over 24 hours.
# 4. Multi-Quantile Head: Outputs P25, P50, and P90 simultaneously.
# """

# import os
# from pathlib import Path
# import numpy as np
# import pandas as pd
# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# import matplotlib.pyplot as plt

# PROJECT_ROOT = Path(__file__).resolve().parents[2]
# SPATIAL_FILE = PROJECT_ROOT / "data" / "processed" / "spatial_model_ready.csv"
# OUTPUT_DIR   = PROJECT_ROOT / "models" / "spatial_model"

# SEQ_LEN = 96      # Look back 24 hours (96 x 15-min intervals)
# BATCH_SIZE = 64
# EPOCHS = 10
# LR = 0.001
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# class SpatialGridDataset(Dataset):
#     """
#     Transforms tabular DataFrame into 3D Spatial Tensors and 1D Physical Features.
#     Output Shape: Spatial (Seq_Len, 4, 3, 3), Physical (Seq_Len, 4)
#     """
#     def __init__(self, df, seq_len=96):
#         self.seq_len = seq_len
#         self.systems = df["system_id"].unique()
        
#         self.X_spatial = []
#         self.X_flat = []
#         self.Y_data = []
        
#         # Spatial Variables (Channels)
#         self.spatial_vars = ["ghi", "cloud", "temp", "wind"]
#         # Flat Physical Variables (Geometry & Clear Sky)
#         self.flat_vars = ["clearsky_ghi", "sun_elevation", "sun_azimuth", "hour_cos"]
        
#         for sid in self.systems:
#             sys_df = df[df["system_id"] == sid].sort_values("timestamp")
            
#             # Construct (Time, Channel, H, W) Tensor array
#             # We map grid_0 to grid_8 back into a 3x3 matrix
#             t_len = len(sys_df)
#             sys_tensor = np.zeros((t_len, len(self.spatial_vars), 3, 3), dtype=np.float32)
#             flat_tensor = sys_df[self.flat_vars].values.astype(np.float32)
            
#             for c_idx, v in enumerate(self.spatial_vars):
#                 for i in range(3):
#                     for j in range(3):
#                         grid_idx = i * 3 + j
#                         col_name = f"{v}_grid_{grid_idx}"
#                         sys_tensor[:, c_idx, i, j] = sys_df[col_name].values
                        
#             targets = sys_df["target_norm"].values.astype(np.float32)
            
#             # Create Sliding Windows
#             for start in range(t_len - seq_len):
#                 self.X_spatial.append(sys_tensor[start : start + seq_len])
#                 self.X_flat.append(flat_tensor[start : start + seq_len])
#                 # Target is the value at the END of the sequence
#                 self.Y_data.append(targets[start + seq_len])
                
#     def __len__(self):
#         return len(self.X_spatial)
        
#     def __getitem__(self, idx):
#         return (torch.tensor(self.X_spatial[idx]), 
#                 torch.tensor(self.X_flat[idx]), 
#                 torch.tensor(self.Y_data[idx]))

# class SpatialAttentionCNN(nn.Module):
#     def __init__(self, in_channels=4, num_flat_features=4, seq_len=96):
#         super().__init__()
        
#         # 1. INPUT ATTENTION (From the paper): Learnable weights per variable
#         # A 1x1 Convolution applied channel-wise acts as an attention mask
#         self.input_attention = nn.Conv2d(in_channels, in_channels, kernel_size=1, groups=in_channels)
        
#         # 2. SPATIAL FEATURE EXTRACTION: Compress 3x3 Grid into spatial vector
#         # kernel_size=3 with no padding on a 3x3 grid reduces it exactly to 1x1
#         self.spatial_conv = nn.Conv2d(in_channels, 16, kernel_size=3)
#         self.relu = nn.ReLU()
        
#         # 3. TEMPORAL ENCODER: Process the 16-dimensional spatial vectors + 4 flat physics features
#         self.lstm = nn.LSTM(input_size=16 + num_flat_features, hidden_size=64, num_layers=2, batch_first=True, dropout=0.1)
        
#         # 4. MULTI-QUANTILE DECODER (P25, P50, P90)
#         self.fc_p25 = nn.Linear(64, 1)
#         self.fc_p50 = nn.Linear(64, 1)
#         self.fc_p90 = nn.Linear(64, 1)

#     def forward(self, x_spatial, x_flat):
#         # Input shape: (Batch, Seq_Len, Channels, Height, Width)
#         B, S, C, H, W = x_spatial.shape
        
#         # Merge Batch and Sequence dimensions to process spatial grids through CNN
#         x = x_spatial.view(B * S, C, H, W)
        
#         # --- Paper Phase 4: Input Attention ---
#         # Apply 1x1 attention and multiply by original input (element-wise)
#         attention_weights = torch.sigmoid(self.input_attention(x))
#         x = x * attention_weights 
        
#         # --- Paper Phase 3: Spatial Encoder ---
#         x = self.relu(self.spatial_conv(x)) # Output shape: (B*S, 16, 1, 1)
#         x_spatial_encoded = x.view(B, S, 16) # Reshape back to sequence: (Batch, Seq_Len, 16)
        
#         # --- Physics Injection ---
#         # Concatenate the 16 spatial features with the 4 physical features
#         x_combined = torch.cat([x_spatial_encoded, x_flat], dim=2) # Shape: (Batch, Seq_Len, 20)
        
#         # --- Temporal Decoder ---
#         lstm_out, (hn, cn) = self.lstm(x_combined)
#         last_hidden = lstm_out[:, -1, :]    # Take the last time step: (Batch, 64)
        
#         # --- Outputs ---
#         p25 = self.fc_p25(last_hidden).squeeze()
#         p50 = self.fc_p50(last_hidden).squeeze()
#         p90 = self.fc_p90(last_hidden).squeeze()
        
#         # Return stacked quantiles: (Batch, 3)
#         return torch.stack([p25, p50, p90], dim=1)

# def quantile_loss(preds, target, quantiles=[0.25, 0.50, 0.90]):
#     """Calculates Pinball Loss for multiple quantiles simultaneously."""
#     assert preds.shape[1] == len(quantiles), "Prediction dimension must match quantiles."
#     loss = 0.0
#     for i, q in enumerate(quantiles):
#         err = target - preds[:, i]
#         # Pinball formula
#         loss += torch.mean(torch.max((q - 1) * err, q * err))
#     return loss

# def main():
#     if not SPATIAL_FILE.exists():
#         print(f"Error: Run spatial_make_model_ready.py first.")
#         return
        
#     OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
#     print("Loading data...")
#     df = pd.read_csv(SPATIAL_FILE)
    
#     # Train/Val Split
#     train_df = df[df["split"] == "train"]
#     val_df = df[df["split"] == "val"]
    
#     print(f"Generating 3D spatial sequences (This takes a moment)...")
#     train_dataset = SpatialGridDataset(train_df, seq_len=SEQ_LEN)
#     val_dataset = SpatialGridDataset(val_df, seq_len=SEQ_LEN)
    
#     train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
#     print(f"Initializing Spatial Attention CNN (Hybrid Physics) -> {DEVICE}")
#     model = SpatialAttentionCNN(in_channels=4, num_flat_features=4).to(DEVICE)
#     optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
#     # NEW CHECKPOINT NAME to force retraining with the new architecture
#     best_model_path = OUTPUT_DIR / "spatial_hybrid_best.pth"
    
#     if best_model_path.exists():
#         print(f"\n[!] Existing model found at {best_model_path.name}. Skipping training!")
#     else:
#         print("\nStarting Training...")
#         best_val_loss = float('inf')
        
#         for epoch in range(EPOCHS):
#             model.train()
#             train_loss = 0.0
            
#             for batch_x_spatial, batch_x_flat, batch_y in train_loader:
#                 batch_x_spatial = batch_x_spatial.to(DEVICE)
#                 batch_x_flat = batch_x_flat.to(DEVICE)
#                 batch_y = batch_y.to(DEVICE)
                
#                 optimizer.zero_grad()
#                 preds = model(batch_x_spatial, batch_x_flat)
#                 loss = quantile_loss(preds, batch_y)
                
#                 loss.backward()
#                 optimizer.step()
#                 train_loss += loss.item()
            
#             model.eval()
#             val_loss = 0.0
#             with torch.no_grad():
#                 for batch_x_spatial, batch_x_flat, batch_y in val_loader:
#                     batch_x_spatial = batch_x_spatial.to(DEVICE)
#                     batch_x_flat = batch_x_flat.to(DEVICE)
#                     batch_y = batch_y.to(DEVICE)
                    
#                     preds = model(batch_x_spatial, batch_x_flat)
#                     val_loss += quantile_loss(preds, batch_y).item()
                
#             avg_train = train_loss / len(train_loader)
#             avg_val = val_loss / len(val_loader)
#             print(f"Epoch {epoch+1:02d}/{EPOCHS} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")
            
#             if avg_val < best_val_loss:
#                 best_val_loss = avg_val
#                 torch.save(model.state_dict(), best_model_path)
                
#         print("\n==================================================")
#         print("SPATIAL HYBRID NWP ERROR CORRECTION COMPLETE")
#         print(f"Best Validation Loss: {best_val_loss:.4f}")
#         print("==================================================")
    
#     print("\nEvaluating on Test Set (Unseen Time Period)...")
#     # Load the best weights we just saved (or found on disk)
#     model.load_state_dict(torch.load(best_model_path, weights_only=True))
#     model.eval()
    
#     test_df = df[df["split"] == "test"]
#     test_dataset = SpatialGridDataset(test_df, seq_len=SEQ_LEN)
#     test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
#     all_p25, all_p50, all_actuals = [], [], []
    
#     with torch.no_grad():
#         for batch_x_spatial, batch_x_flat, batch_y in test_loader:
#             batch_x_spatial = batch_x_spatial.to(DEVICE)
#             batch_x_flat = batch_x_flat.to(DEVICE)
            
#             preds = model(batch_x_spatial, batch_x_flat)
            
#             all_p25.extend(preds[:, 0].cpu().numpy())
#             all_p50.extend(preds[:, 1].cpu().numpy())
#             all_actuals.extend(batch_y.numpy())
            
#     all_p25 = np.array(all_p25)
#     all_p50 = np.array(all_p50)
#     all_actuals = np.array(all_actuals)
    
#     # Daylight-only evaluation (Apples-to-Apples with previous models)
#     day_mask = all_actuals > 0.01
#     a_day = all_actuals[day_mask]
#     p50_day = all_p50[day_mask]
#     p25_day = all_p25[day_mask]
    
#     rmse_p50 = np.sqrt(np.mean((p50_day - a_day)**2))
#     r2_p50 = 1 - (np.sum((p50_day - a_day)**2) / np.sum((a_day - np.mean(a_day))**2))
    
#     rmse_p25 = np.sqrt(np.mean((p25_day - a_day)**2))
#     mbe_p25 = np.mean(p25_day - a_day)

#     print("\n" + "="*50)
#     print("SPATIAL CNN FORECAST RESULTS (Daylight Only)")
#     print("="*50)
#     print("1. Business/Manager Metrics (P50 Forecast):")
#     print(f"   RMSE : {rmse_p50:.4f}")
#     print(f"   R^2  : {r2_p50:.4f}")
#     print("\n2. Battery EMS Metrics (P25 Forecast):")
#     print(f"   RMSE : {rmse_p25:.4f}")
#     print(f"   MBE  : {mbe_p25:.4f}")
#     print("="*50)

# if __name__ == "__main__":
#     main()
#!/usr/bin/env python3
"""
spatial_cnn_model.py
====================
A PyTorch implementation of the "Deep multivariable spatial attention CNN" 
methodology from the paper, upgraded into a Hybrid model that injects physical 
astronomical boundaries (clear-sky proxy) directly into the neural network.

Architecture:
1. Input Attention: Dynamically learns which weather variables matter.
2. Spatial Convolution: Compresses the 3x3 bounding box into a spatial vector.
3. Physical Fusion: Concatenates the spatial vector with exact solar geometry.
4. Temporal LSTM: Learns the sequence of weather over 24 hours.
5. Multi-Quantile Head: Outputs P25, P50, and P90 safely.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPATIAL_FILE = PROJECT_ROOT / "data" / "processed" / "spatial_model_ready.csv"
OUTPUT_DIR   = PROJECT_ROOT / "models" / "spatial_model"

SEQ_LEN = 96      # Look back 24 hours (96 x 15-min intervals)
BATCH_SIZE = 64
EPOCHS = 10
LR = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SpatialGridDataset(Dataset):
    """
    Transforms tabular DataFrame into 3D Spatial Tensors and 1D Physical Features.
    Output Shape: Spatial (Seq_Len, 4, 3, 3), Physical (Seq_Len, 4)
    """
    def __init__(self, df, seq_len=96):
        self.seq_len = seq_len
        self.systems = df["system_id"].unique()
        
        self.X_spatial = []
        self.X_flat = []
        self.Y_data = []
        
        # Spatial Variables (Channels) - 4 weather layers
        self.spatial_vars = ["ghi", "cloud", "temp", "wind"]
        # Flat Physical Variables (Geometry & Clear Sky) - 4 physics constraints
        self.flat_vars = ["clearsky_ghi", "sun_elevation", "sun_azimuth", "hour_cos"]
        
        for sid in self.systems:
            sys_df = df[df["system_id"] == sid].sort_values("timestamp")
            
            # Construct (Time, Channel, H, W) Tensor array
            # Map grid_0 to grid_8 back into a 3x3 matrix
            t_len = len(sys_df)
            sys_tensor = np.zeros((t_len, len(self.spatial_vars), 3, 3), dtype=np.float32)
            flat_tensor = sys_df[self.flat_vars].values.astype(np.float32)
            
            for c_idx, v in enumerate(self.spatial_vars):
                for i in range(3):
                    for j in range(3):
                        grid_idx = i * 3 + j
                        col_name = f"{v}_grid_{grid_idx}"
                        sys_tensor[:, c_idx, i, j] = sys_df[col_name].values
                        
            targets = sys_df["target_norm"].values.astype(np.float32)
            
            # Create Sliding Windows
            for start in range(t_len - seq_len):
                self.X_spatial.append(sys_tensor[start : start + seq_len])
                self.X_flat.append(flat_tensor[start : start + seq_len])
                # Target is the value at the END of the sequence
                self.Y_data.append(targets[start + seq_len])
                
    def __len__(self):
        return len(self.X_spatial)
        
    def __getitem__(self, idx):
        return (torch.tensor(self.X_spatial[idx]), 
                torch.tensor(self.X_flat[idx]), 
                torch.tensor(self.Y_data[idx]))

class SpatialAttentionCNN(nn.Module):
    def __init__(self, in_channels=4, num_flat_features=4, seq_len=96):
        super().__init__()
        
        # 1. INPUT ATTENTION (From the paper): Learnable weights per variable
        self.input_attention = nn.Conv2d(in_channels, in_channels, kernel_size=1, groups=in_channels)
        
        # 2. SPATIAL FEATURE EXTRACTION: Compress 3x3 Grid into spatial vector
        # kernel_size=3 with no padding on a 3x3 grid reduces it exactly to 1x1
        self.spatial_conv = nn.Conv2d(in_channels, 16, kernel_size=3)
        self.relu = nn.ReLU()
        
        # 3. TEMPORAL ENCODER: Process the 16-dimensional spatial vectors + 4 flat physics features
        # The fusion happens here: 16 (spatial) + 4 (physical) = 20 input size
        self.lstm = nn.LSTM(input_size=16 + num_flat_features, hidden_size=64, num_layers=2, batch_first=True, dropout=0.1)
        
        # 4. MULTI-QUANTILE DECODER (P25, P50, P90)
        self.fc_p25 = nn.Linear(64, 1)
        self.fc_p50 = nn.Linear(64, 1)
        self.fc_p90 = nn.Linear(64, 1)

    def forward(self, x_spatial, x_flat):
        # Input shape: (Batch, Seq_Len, Channels, Height, Width)
        B, S, C, H, W = x_spatial.shape
        
        # Merge Batch and Sequence dimensions to process spatial grids through CNN
        x = x_spatial.view(B * S, C, H, W)
        
        # --- Paper Phase 4: Input Attention ---
        attention_weights = torch.sigmoid(self.input_attention(x))
        x = x * attention_weights 
        
        # --- Paper Phase 3: Spatial Encoder ---
        x = self.relu(self.spatial_conv(x)) # Output shape: (B*S, 16, 1, 1)
        x_spatial_encoded = x.view(B, S, 16) # Reshape back to sequence: (Batch, Seq_Len, 16)
        
        # --- Physics Injection (The Hybrid Upgrade) ---
        # Concatenate the 16 spatial features with the 4 physical features
        x_combined = torch.cat([x_spatial_encoded, x_flat], dim=2) # Shape: (Batch, Seq_Len, 20)
        
        # --- Temporal Decoder ---
        lstm_out, (hn, cn) = self.lstm(x_combined)
        last_hidden = lstm_out[:, -1, :]    # Take the last time step: (Batch, 64)
        
        # --- Outputs ---
        p25 = self.fc_p25(last_hidden).squeeze()
        p50 = self.fc_p50(last_hidden).squeeze()
        p90 = self.fc_p90(last_hidden).squeeze()
        
        # Return stacked quantiles: (Batch, 3)
        return torch.stack([p25, p50, p90], dim=1)

def quantile_loss(preds, target, quantiles=[0.25, 0.50, 0.90]):
    """Calculates Pinball Loss for multiple quantiles simultaneously."""
    assert preds.shape[1] == len(quantiles), "Prediction dimension must match quantiles."
    loss = 0.0
    for i, q in enumerate(quantiles):
        err = target - preds[:, i]
        loss += torch.mean(torch.max((q - 1) * err, q * err))
    return loss

def main():
    if not SPATIAL_FILE.exists():
        print(f"Error: Run spatial_make_model_ready.py first.")
        return
        
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("Loading data...")
    df = pd.read_csv(SPATIAL_FILE)
    
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    
    print(f"Generating 3D spatial sequences (This takes a moment)...")
    train_dataset = SpatialGridDataset(train_df, seq_len=SEQ_LEN)
    val_dataset = SpatialGridDataset(val_df, seq_len=SEQ_LEN)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Initializing Spatial-Physical Hybrid CNN -> {DEVICE}")
    model = SpatialAttentionCNN(in_channels=4, num_flat_features=4).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    # We use a distinct filename for the hybrid model
    best_model_path = OUTPUT_DIR / "spatial_hybrid_best.pth"
    
    if best_model_path.exists():
        print(f"\n[!] Existing model found at {best_model_path.name}. Skipping training!")
    else:
        print("\nStarting Training...")
        best_val_loss = float('inf')
        
        for epoch in range(EPOCHS):
            model.train()
            train_loss = 0.0
            
            for batch_x_spatial, batch_x_flat, batch_y in train_loader:
                batch_x_spatial = batch_x_spatial.to(DEVICE)
                batch_x_flat = batch_x_flat.to(DEVICE)
                batch_y = batch_y.to(DEVICE)
                
                optimizer.zero_grad()
                preds = model(batch_x_spatial, batch_x_flat)
                loss = quantile_loss(preds, batch_y)
                
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_x_spatial, batch_x_flat, batch_y in val_loader:
                    batch_x_spatial = batch_x_spatial.to(DEVICE)
                    batch_x_flat = batch_x_flat.to(DEVICE)
                    batch_y = batch_y.to(DEVICE)
                    
                    preds = model(batch_x_spatial, batch_x_flat)
                    val_loss += quantile_loss(preds, batch_y).item()
                
            avg_train = train_loss / len(train_loader)
            avg_val = val_loss / len(val_loader)
            print(f"Epoch {epoch+1:02d}/{EPOCHS} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")
            
            if avg_val < best_val_loss:
                best_val_loss = avg_val
                torch.save(model.state_dict(), best_model_path)
                
        print("\n==================================================")
        print("HYBRID SPATIAL NWP ERROR CORRECTION COMPLETE")
        print(f"Best Validation Loss: {best_val_loss:.4f}")
        print("==================================================")
    
    print("\nEvaluating on Test Set (Unseen Time Period)...")
    
    # Safe load with weights_only=True to prevent FutureWarnings
    model.load_state_dict(torch.load(best_model_path, map_location=DEVICE, weights_only=True))
    model.eval()
    
    test_df = df[df["split"] == "test"]
    test_dataset = SpatialGridDataset(test_df, seq_len=SEQ_LEN)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    all_p25, all_p50, all_actuals = [], [], []
    
    with torch.no_grad():
        for batch_x_spatial, batch_x_flat, batch_y in test_loader:
            batch_x_spatial = batch_x_spatial.to(DEVICE)
            batch_x_flat = batch_x_flat.to(DEVICE)
            
            preds = model(batch_x_spatial, batch_x_flat)
            
            all_p25.extend(preds[:, 0].cpu().numpy())
            all_p50.extend(preds[:, 1].cpu().numpy())
            all_actuals.extend(batch_y.numpy())
            
    all_p25 = np.array(all_p25)
    all_p50 = np.array(all_p50)
    all_actuals = np.array(all_actuals)
    
    # Daylight-only evaluation (Apples-to-Apples with previous models)
    day_mask = all_actuals > 0.01
    a_day = all_actuals[day_mask]
    p50_day = all_p50[day_mask]
    p25_day = all_p25[day_mask]
    
    rmse_p50 = np.sqrt(np.mean((p50_day - a_day)**2))
    r2_p50 = 1 - (np.sum((p50_day - a_day)**2) / np.sum((a_day - np.mean(a_day))**2))
    
    rmse_p25 = np.sqrt(np.mean((p25_day - a_day)**2))
    mbe_p25 = np.mean(p25_day - a_day)

    print("\n" + "="*50)
    print("HYBRID SPATIAL CNN FORECAST RESULTS (Daylight Only)")
    print("="*50)
    print("1. Business/Manager Metrics (P50 Forecast):")
    print(f"   RMSE : {rmse_p50:.4f}")
    print(f"   R^2  : {r2_p50:.4f}")
    print("\n2. Battery EMS Metrics (P25 Forecast):")
    print(f"   RMSE : {rmse_p25:.4f}")
    print(f"   MBE  : {mbe_p25:.4f}  <-- Should be safely negative")
    print("="*50)

if __name__ == "__main__":
    main()