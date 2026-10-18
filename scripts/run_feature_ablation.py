"""
Grouped Feature-Ablation Experiment for Initial SoC Estimation (50s Nominal Window)

Evaluates 4 cumulative feature representations:
  - Group A: [V0, I0, T0] (3 features)
  - Group B: Group A + [Vend, Iend, Tend, dV, dI, dT] (9 features)
  - Group C: Group B + [meanV, meanI, stdV] (12 features)
  - Group D: Group C + [Q_window] (13 features)

Evaluation protocol:
  - 50 s nominal observation windows extracted with stride 500
  - Scaler fitted on TRAINING DATA ONLY for each group
  - Primary model: Summary MLP across 5 fixed random seeds [42, 100, 200, 300, 400]
  - Linear reference: Ridge regression (alpha=1.0)
  - Decision split: Validation (-10°C) strictly
  - Test set (-20°C): LOCKED / NOT ACCESSED
"""

import os
import sys
import time
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# 1. Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Selected compute device: {device}")
print(f"PyTorch Version:         {torch.__version__}")
if torch.cuda.is_available():
    print(f"CUDA Runtime Version:    {torch.version.cuda}")
    print(f"GPU Hardware:            {torch.cuda.get_device_name(0)}")
    props = torch.cuda.get_device_properties(0)
    print(f"Compute Capability:      {props.major}.{props.minor}")
    print(f"Total Dedicated VRAM:    {props.total_memory / (1024**2):.1f} MiB")

# 2. Paths
PROJECT_ROOT = Path(r"c:\Users\sheok\OneDrive\Desktop\New folder (2)\Major Project")
DATA_DIR     = PROJECT_ROOT / "data"
RESULTS_DIR  = PROJECT_ROOT / "results"
MODELS_DIR   = RESULTS_DIR / "models"

MODELS_DIR.mkdir(parents=True, exist_ok=True)

# 3. Load 50s Nominal Dataset
DATASET_PATH = DATA_DIR / "initial_soc_datasets.pkl"
SCALER_PATH  = DATA_DIR / "initial_soc_scaler.npz"

with open(DATASET_PATH, "rb") as f:
    datasets = pickle.load(f)

dur = 50
X_train_raw = datasets[dur]["X_train"]  # (5182, 500, 3)
t_train_raw = datasets[dur]["t_train"]  # (5182, 500)
y_train     = datasets[dur]["y_train"]  # (5182,)

X_val_raw   = datasets[dur]["X_val"]    # (1045, 500, 3)
t_val_raw   = datasets[dur]["t_val"]    # (1045, 500)
y_val       = datasets[dur]["y_val"]    # (1045,)

# Strictly do NOT touch test set
print(f"\nLoaded 50s Nominal Dataset:")
print(f"  Train windows:      {len(X_train_raw)} (25°C, 10°C, 0°C)")
print(f"  Validation windows: {len(X_val_raw)} (-10°C)")
print(f"  Test set (-20°C):   LOCKED (not queried)")

# 4. Feature Extraction (13 Raw Features)
scaler_data = np.load(SCALER_PATH)
feat_min    = scaler_data["feat_min"]
feat_range  = scaler_data["feat_range"]

def extract_all_13_features(X_seq, t_seq):
    V_0 = X_seq[:, 0, 0]
    I_0 = X_seq[:, 0, 1]
    T_0 = X_seq[:, 0, 2]
    
    V_end = X_seq[:, -1, 0]
    I_end = X_seq[:, -1, 1]
    T_end = X_seq[:, -1, 2]
    
    dV = V_end - V_0
    dI = I_end - I_0
    dT = T_end - T_0
    
    meanV = np.mean(X_seq[:, :, 0], axis=1)
    meanI = np.mean(X_seq[:, :, 1], axis=1)
    stdV  = np.std(X_seq[:, :, 0], axis=1)
    
    I_phys = X_seq[:, :, 1] * feat_range[1] + feat_min[1]
    Q_window_As = np.trapezoid(I_phys, x=t_seq, axis=1)
    Q_window_Ah = Q_window_As / 3600.0
    
    feats = np.column_stack([
        V_0, I_0, T_0,
        V_end, I_end, T_end,
        dV, dI, dT,
        meanV, meanI, stdV,
        Q_window_Ah
    ])
    return feats

feats_train_all = extract_all_13_features(X_train_raw, t_train_raw)
feats_val_all   = extract_all_13_features(X_val_raw, t_val_raw)

FEATURE_NAMES = [
    "V0", "I0", "T0",
    "Vend", "Iend", "Tend",
    "dV", "dI", "dT",
    "meanV", "meanI", "stdV",
    "Q_window"
]

GROUPS = {
    "Group A": {
        "desc": "Instantaneous operating point [V0, I0, T0]",
        "indices": [0, 1, 2],
        "dim": 3,
        "names": FEATURE_NAMES[0:3]
    },
    "Group B": {
        "desc": "Group A + Endpoint dynamics [Vend, Iend, Tend, dV, dI, dT]",
        "indices": list(range(0, 9)),
        "dim": 9,
        "names": FEATURE_NAMES[0:9]
    },
    "Group C": {
        "desc": "Group B + Dynamic trajectory statistics [meanV, meanI, stdV]",
        "indices": list(range(0, 12)),
        "dim": 12,
        "names": FEATURE_NAMES[0:12]
    },
    "Group D": {
        "desc": "Group C + Integrated charge [Q_window]",
        "indices": list(range(0, 13)),
        "dim": 13,
        "names": FEATURE_NAMES[0:13]
    }
}

# 5. Model Architecture
class SummaryMLP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.net(x)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# 6. Evaluation Protocols
SEEDS = [42, 100, 200, 300, 400]
BATCH_SIZE = 64
EPOCHS = 150
PATIENCE = 15
LR = 1e-3
WEIGHT_DECAY = 1e-5

seed_records = []
summary_records = []

print("\n" + "="*70)
print("STARTING GROUPED FEATURE ABLATION EXPERIMENT (Validation Set @ -10°C)")
print("="*70)

for grp_key, grp_info in GROUPS.items():
    idx = grp_info["indices"]
    dim = grp_info["dim"]
    desc = grp_info["desc"]
    
    print(f"\n>>> {grp_key} (dim={dim}): {desc}")
    print(f"    Features: {grp_info['names']}")
    
    # Preprocessing: Fit scaler on TRAINING DATA ONLY
    scaler = StandardScaler()
    X_tr_grp = scaler.fit_transform(feats_train_all[:, idx])
    X_va_grp = scaler.transform(feats_val_all[:, idx])
    
    # Save group scaler
    scaler_save_path = DATA_DIR / f"ablation_scaler_{grp_key.replace(' ', '_').lower()}.pkl"
    with open(scaler_save_path, "wb") as f:
        pickle.dump(scaler, f)
        
    # --- Linear Reference: Ridge Regression ---
    ridge = Ridge(alpha=1.0, random_state=42)
    ridge.fit(X_tr_grp, y_train)
    pred_ridge_val = ridge.predict(X_va_grp) * 100.0
    y_val_phys = y_val * 100.0
    
    ridge_rmse = np.sqrt(mean_squared_error(y_val_phys, pred_ridge_val))
    ridge_mae  = mean_absolute_error(y_val_phys, pred_ridge_val)
    print(f"    [Linear Reference - Ridge] Val RMSE: {ridge_rmse:.2f}% | Val MAE: {ridge_mae:.2f}%")
    
    # --- Primary Model: Summary MLP across 5 seeds ---
    grp_val_rmses = []
    grp_val_maes  = []
    grp_runtimes  = []
    
    for seed in SEEDS:
        # Strict seed initialization
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        
        g = torch.Generator()
        g.manual_seed(seed)
        
        train_ds = TensorDataset(torch.tensor(X_tr_grp, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
        val_ds   = TensorDataset(torch.tensor(X_va_grp, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
        
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, generator=g)
        val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
        
        model = SummaryMLP(in_dim=dim).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        
        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None
        best_epoch = 0
        
        t0 = time.perf_counter()
        for epoch in range(EPOCHS):
            model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                preds = model(xb).squeeze()
                loss = criterion(preds, yb)
                loss.backward()
                optimizer.step()
                
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    preds = model(xb).squeeze()
                    loss = criterion(preds, yb)
                    val_loss += loss.item() * xb.size(0)
            val_loss /= len(val_ds)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    break
                    
        elapsed_s = time.perf_counter() - t0
        
        # Load best weights & evaluate
        model.load_state_dict(best_state)
        model.to(device)
        model.eval()
        with torch.no_grad():
            preds_all = []
            for xb, _ in val_loader:
                xb = xb.to(device)
                preds_all.append(model(xb).squeeze().cpu().numpy())
            val_preds_norm = np.concatenate(preds_all)
            
        val_preds_phys = val_preds_norm * 100.0
        val_rmse = np.sqrt(mean_squared_error(y_val_phys, val_preds_phys))
        val_mae  = mean_absolute_error(y_val_phys, val_preds_phys)
        
        # Save model checkpoint
        ckpt_name = f"ablation_{grp_key.replace(' ', '_').lower()}_seed_{seed}.pt"
        torch.save(best_state, MODELS_DIR / ckpt_name)
        
        grp_val_rmses.append(val_rmse)
        grp_val_maes.append(val_mae)
        grp_runtimes.append(elapsed_s)
        
        param_count = count_parameters(model)
        seed_records.append({
            "Group": grp_key,
            "Description": desc,
            "Feature_Dim": dim,
            "Seed": seed,
            "Val_RMSE_pct": val_rmse,
            "Val_MAE_pct": val_mae,
            "Best_Epoch": best_epoch,
            "Train_Time_s": elapsed_s,
            "Param_Count": param_count
        })
        print(f"      Seed {seed:3d}: Val RMSE={val_rmse:5.2f}% | Val MAE={val_mae:5.2f}% | Best Epoch={best_epoch:3d} ({elapsed_s:4.1f}s)")
        
    mean_rmse = np.mean(grp_val_rmses)
    std_rmse  = np.std(grp_val_rmses)
    mean_mae  = np.mean(grp_val_maes)
    std_mae   = np.std(grp_val_maes)
    mean_time = np.mean(grp_runtimes)
    
    print(f"    --> MLP Summary Mean ± Std: RMSE = {mean_rmse:.2f} ± {std_rmse:.2f}% | MAE = {mean_mae:.2f} ± {std_mae:.2f}%")
    
    summary_records.append({
        "Group": grp_key,
        "Description": desc,
        "Feature_Dim": dim,
        "MLP_Val_RMSE_Mean": mean_rmse,
        "MLP_Val_RMSE_Std": std_rmse,
        "MLP_Val_MAE_Mean": mean_mae,
        "MLP_Val_MAE_Std": std_mae,
        "Ridge_Val_RMSE": ridge_rmse,
        "Ridge_Val_MAE": ridge_mae,
        "Param_Count": param_count,
        "Avg_Train_Time_s": mean_time
    })

# 7. Save Results
df_seeds = pd.DataFrame(seed_records)
df_seeds.to_csv(RESULTS_DIR / "feature_ablation_seeds.csv", index=False)

df_summary = pd.DataFrame(summary_records)
df_summary.to_csv(RESULTS_DIR / "feature_ablation_summary.csv", index=False)

print("\n" + "="*70)
print("FINAL FEATURE ABLATION SUMMARY TABLE (Validation @ -10°C)")
print("="*70)
print(df_summary[["Group", "Feature_Dim", "MLP_Val_RMSE_Mean", "MLP_Val_RMSE_Std", "MLP_Val_MAE_Mean", "MLP_Val_MAE_Std", "Ridge_Val_RMSE", "Ridge_Val_MAE"]].to_string(index=False))

# 8. Incremental Transitions Analysis
print("\n" + "="*70)
print("INCREMENTAL TRANSITION ANALYSIS (MLP Mean Errors)")
print("="*70)

for i in range(len(summary_records) - 1):
    curr = summary_records[i]
    nxt  = summary_records[i+1]
    
    d_rmse = nxt["MLP_Val_RMSE_Mean"] - curr["MLP_Val_RMSE_Mean"]
    d_mae  = nxt["MLP_Val_MAE_Mean"] - curr["MLP_Val_MAE_Mean"]
    pooled_std = np.sqrt(0.5 * (curr["MLP_Val_RMSE_Std"]**2 + nxt["MLP_Val_RMSE_Std"]**2))
    
    print(f"Transition {curr['Group']} -> {nxt['Group']}:")
    print(f"  Added Features:  {GROUPS[nxt['Group']]['names'][len(GROUPS[curr['Group']]['names']):]}")
    print(f"  Δ Val RMSE:      {d_rmse:+.2f}% SoC")
    print(f"  Δ Val MAE:       {d_mae:+.2f}% SoC")
    print(f"  Pooled Seed Std: ±{pooled_std:.2f}% SoC")
    if abs(d_rmse) > pooled_std:
        print(f"  Signal Status:   Meaningful improvement exceeding seed variability (|Δ| > std).")
    else:
        print(f"  Signal Status:   Within stochastic seed variability (|Δ| <= std); marginal gain.")
    print("-" * 50)

print("\nAblation experiment complete. All models and tables successfully saved.")

