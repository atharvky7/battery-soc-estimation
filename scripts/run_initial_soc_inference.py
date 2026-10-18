#!/usr/bin/env python3
"""
Single-Window Initial SoC Inference Script
==========================================

Performs cold-start initial State of Charge (SoC) estimation from a single
50-second dynamic observation window of terminal voltage, load current,
and battery temperature:

    [V(t), I(t), T(t)] over 50 s (500 samples) -> Initial SoC [%] at t=0

Model Architecture:
    SummaryMLP: Linear(12, 64) -> ReLU -> Linear(64, 32) -> ReLU -> Linear(32, 1)

Designated Frozen Artifacts:
    - Checkpoint:  results/models/ablation_group_c_seed_42.pt
    - Scaler:      data/ablation_scaler_group_c.pkl
    - Normalizer:  data/initial_soc_scaler.npz

Important Methodological Principles:
    1. Zero retraining.
    2. Zero target leakage: No Ah measurements or SoC references are ever used.
    3. Input units must be strictly: Voltage in Volts (V), Current in Amperes (A),
       and Temperature in Celsius (°C).
    4. Sign convention: Negative current = Discharge, Positive current = Charge/Regen.
"""

import sys
import argparse
import pickle
from pathlib import Path
from typing import Dict, Tuple, Union, Optional

import numpy as np
import torch
import torch.nn as nn


# -------------------------------------------------------------------------
# 1. Model Definition (Frozen Architecture)
# -------------------------------------------------------------------------
class SummaryMLP(nn.Module):
    """3-layer Multilayer Perceptron for Group C 12-feature representation."""
    def __init__(self, in_dim: int = 12):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# -------------------------------------------------------------------------
# 2. Artifact Loader
# -------------------------------------------------------------------------
def load_frozen_artifacts(
    project_root: Optional[Union[str, Path]] = None,
    device: str = "cpu"
) -> Dict[str, object]:
    """
    Loads designated frozen model and scalers.
    
    Returns:
        dict with keys: 'model', 'scaler_c', 'feat_min', 'feat_range', 'device'
    """
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent
    else:
        project_root = Path(project_root)

    model_path = project_root / "results" / "models" / "ablation_group_c_seed_42.pt"
    scaler_c_path = project_root / "data" / "ablation_scaler_group_c.pkl"
    scaler_npz_path = project_root / "data" / "initial_soc_scaler.npz"

    if not model_path.exists():
        raise FileNotFoundError(f"Missing frozen model checkpoint: {model_path}")
    if not scaler_c_path.exists():
        raise FileNotFoundError(f"Missing frozen feature scaler: {scaler_c_path}")
    if not scaler_npz_path.exists():
        raise FileNotFoundError(f"Missing frozen channel normalizer: {scaler_npz_path}")

    # Load channel normalizer (MinMax)
    scaler_data = np.load(scaler_npz_path)
    feat_min = scaler_data["feat_min"]
    feat_range = scaler_data["feat_range"]

    # Load feature scaler (StandardScaler)
    with open(scaler_c_path, "rb") as f:
        scaler_c = pickle.load(f)

    # Load neural network model
    dev = torch.device(device)
    model = SummaryMLP(in_dim=12)
    state_dict = torch.load(model_path, map_location=dev)
    model.load_state_dict(state_dict)
    model.to(dev)
    model.eval()

    return {
        "model": model,
        "scaler_c": scaler_c,
        "feat_min": feat_min,
        "feat_range": feat_range,
        "device": dev
    }


# -------------------------------------------------------------------------
# 3. Exact Feature Extraction
# -------------------------------------------------------------------------
FEATURE_NAMES = [
    "V0", "I0", "T0",
    "Vend", "Iend", "Tend",
    "dV", "dI", "dT",
    "meanV", "meanI", "stdV"
]


def extract_group_c_features(
    v_raw: np.ndarray,
    i_raw: np.ndarray,
    t_raw: np.ndarray,
    feat_min: np.ndarray,
    feat_range: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts the exact 12 Group C features from raw physical signals.
    
    Args:
        v_raw: 1D array of terminal voltages [V] (length N, nominally 500)
        i_raw: 1D array of load currents [A] (length N, nominally 500)
        t_raw: 1D array of battery temperatures [°C] (length N, nominally 500)
        feat_min: (3,) array with minimum channel values from training
        feat_range: (3,) array with channel ranges from training
        
    Returns:
        feats_12: (1, 12) array of extracted summary features
        x_norm: (N, 3) channel-normalized sequence
    """
    assert len(v_raw) == len(i_raw) == len(t_raw), "All channel arrays must have identical length"
    n_samples = len(v_raw)
    if n_samples < 10:
        raise ValueError(f"Observation window too short ({n_samples} samples). Nominal is 500 samples.")

    # Stack raw physical channels
    x_raw = np.column_stack([v_raw, i_raw, t_raw])  # (N, 3)

    # Stage 1: Channel MinMax Normalization
    x_norm = (x_raw - feat_min) / feat_range

    # Stage 2: Extract 12 summary features from normalized sequence
    v0, i0, t0 = x_norm[0, 0], x_norm[0, 1], x_norm[0, 2]
    vend, iend, tend = x_norm[-1, 0], x_norm[-1, 1], x_norm[-1, 2]

    dv = vend - v0
    di = iend - i0
    dt = tend - t0

    mean_v = np.mean(x_norm[:, 0])
    mean_i = np.mean(x_norm[:, 1])
    std_v  = np.std(x_norm[:, 0])

    feats_12 = np.array([
        v0, i0, t0,
        vend, iend, tend,
        dv, di, dt,
        mean_v, mean_i, std_v
    ], dtype=np.float32).reshape(1, -1)

    return feats_12, x_norm


# -------------------------------------------------------------------------
# 4. Primary Inference API
# -------------------------------------------------------------------------
def predict_initial_soc(
    window_data: Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    artifacts: Optional[Dict[str, object]] = None,
    clip_physical: bool = True,
    apply_current_correction: bool = False
) -> Dict[str, Union[float, np.ndarray, Dict[str, float]]]:
    """
    Estimates initial State of Charge (SoC) from a single observation window.
    
    Args:
        window_data: Either:
            - (500, 3) numpy array with columns [Voltage_V, Current_A, Temp_degC]
            - tuple (v_arr, i_arr, t_arr)
        artifacts: Pre-loaded dict from load_frozen_artifacts(). If None, loads defaults.
        clip_physical: Whether to enforce physical [0.0, 100.0]% bounds.
        apply_current_correction: Whether to apply the empirical trajectory-grouped
            current-dependent offset: e_hat(I) = 1.0231 + 2.3575 * mean(I_phys).
            
    Returns:
        dict containing:
            - 'predicted_soc_pct': Final predicted initial SoC [%]
            - 'raw_model_soc_pct': Raw unconstrained model prediction [%]
            - 'corrected_soc_pct': Prediction after empirical current correction [%] (if enabled)
            - 'operating_stats': dict of mean_voltage_V, mean_current_A, mean_temp_C
            - 'features_12': dict mapping feature name to normalized value
    """
    if artifacts is None:
        artifacts = load_frozen_artifacts()

    model = artifacts["model"]
    scaler_c = artifacts["scaler_c"]
    feat_min = artifacts["feat_min"]
    feat_range = artifacts["feat_range"]
    device = artifacts["device"]

    # Parse input channels
    if isinstance(window_data, tuple) and len(window_data) == 3:
        v_raw, i_raw, t_raw = window_data
    elif isinstance(window_data, np.ndarray) and window_data.ndim == 2 and window_data.shape[1] == 3:
        v_raw, i_raw, t_raw = window_data[:, 0], window_data[:, 1], window_data[:, 2]
    else:
        raise ValueError("Input window_data must be (N, 3) array or tuple (v, i, t).")

    # Extract features
    feats_12, _ = extract_group_c_features(v_raw, i_raw, t_raw, feat_min, feat_range)

    # Standardize features using training-only StandardScaler
    feats_scaled = scaler_c.transform(feats_12)

    # Forward pass
    with torch.no_grad():
        x_tensor = torch.tensor(feats_scaled, dtype=torch.float32, device=device)
        pred_norm = model(x_tensor).squeeze().item()

    # Denormalize to percentage
    pred_raw_pct = pred_norm * 100.0

    # Optional empirical current correction (derived from unseen validation trajectories)
    mean_i_phys = float(np.mean(i_raw))
    c0 = 1.023101
    c1 = 2.357521
    e_hat = c0 + c1 * mean_i_phys
    pred_corrected_pct = pred_raw_pct - e_hat

    # Select active prediction
    if apply_current_correction:
        active_soc = pred_corrected_pct
    else:
        active_soc = pred_raw_pct

    if clip_physical:
        final_soc = float(np.clip(active_soc, 0.0, 100.0))
    else:
        final_soc = float(active_soc)

    return {
        "predicted_soc_pct": round(final_soc, 4),
        "raw_model_soc_pct": round(float(pred_raw_pct), 4),
        "corrected_soc_pct": round(float(np.clip(pred_corrected_pct, 0.0, 100.0) if clip_physical else pred_corrected_pct), 4),
        "correction_delta_pct": round(float(e_hat), 4),
        "operating_stats": {
            "mean_voltage_V": round(float(np.mean(v_raw)), 4),
            "mean_current_A": round(mean_i_phys, 4),
            "mean_temp_C": round(float(np.mean(t_raw)), 2),
            "sample_count": len(v_raw)
        },
        "features_12": {k: round(float(v), 5) for k, v in zip(FEATURE_NAMES, feats_12[0])}
    }


# -------------------------------------------------------------------------
# 5. CLI Interface & Self-Test Mode
# -------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Run frozen Group C initial-SoC estimation on a 50s observation window."
    )
    parser.add_argument(
        "--csv", type=str, default=None,
        help="Path to CSV file with 500 rows containing [Voltage, Current, Temperature]."
    )
    parser.add_argument(
        "--test-sample", type=int, default=None,
        help="Run self-test on validation sample index (e.g. --test-sample 0)."
    )
    parser.add_argument(
        "--apply-correction", action="store_true",
        help="Apply empirical grouped current-dependent residual correction."
    )
    parser.add_argument(
        "--no-clip", action="store_true",
        help="Do not clip predictions to physical [0.0, 100.0]%% bounds."
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Compute device ('cpu' or 'cuda')."
    )

    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    artifacts = load_frozen_artifacts(project_root=project_root, device=args.device)

    # Mode 1: Self-test against frozen dataset sample
    if args.test_sample is not None:
        dataset_path = project_root / "data" / "initial_soc_datasets.pkl"
        if not dataset_path.exists():
            print(f"Error: Dataset file not found at {dataset_path}")
            sys.exit(1)

        with open(dataset_path, "rb") as f:
            datasets = pickle.load(f)

        d50 = datasets[50]
        n_val = len(d50["X_val"])
        idx = args.test_sample
        if idx < 0 or idx >= n_val:
            print(f"Error: Index {idx} out of range (validation set has {n_val} windows).")
            sys.exit(1)

        # Retrieve normalized validation sequence and reconstruct physical channels
        x_norm = d50["X_val"][idx]
        y_true_pct = float(d50["y_val"][idx] * 100.0)
        feat_min = artifacts["feat_min"]
        feat_range = artifacts["feat_range"]

        x_raw = x_norm * feat_range + feat_min

        result = predict_initial_soc(
            x_raw,
            artifacts=artifacts,
            clip_physical=not args.no_clip,
            apply_current_correction=args.apply_correction
        )

        err_raw = result["raw_model_soc_pct"] - y_true_pct
        err_corr = result["corrected_soc_pct"] - y_true_pct

        print("\n" + "=" * 65)
        print(f"INITIAL SOC INFERENCE REPORT: Validation Window #{idx} (-10°C)")
        print("="*65)
        print(f"True Initial SoC (Ah Reference): {y_true_pct:8.2f} %")
        print(f"Raw Model Predicted SoC:         {result['raw_model_soc_pct']:8.2f} %  (Error: {err_raw:+6.2f} %)")
        print(f"Grouped-Corrected Predicted SoC: {result['corrected_soc_pct']:8.2f} %  (Error: {err_corr:+6.2f} %)")
        print(f"Operating Conditions:            V={result['operating_stats']['mean_voltage_V']:.3f}V, "
              f"I={result['operating_stats']['mean_current_A']:+.2f}A, "
              f"T={result['operating_stats']['mean_temp_C']:.1f}°C")
        print(f"Extracted 12 Features (Group C): {result['features_12']}")
        print("=" * 65)
        print()
        return

    # Mode 2: Predict on user-supplied CSV
    if args.csv is not None:
        csv_path = Path(args.csv)
        if not csv_path.exists():
            print(f"Error: CSV file not found: {csv_path}")
            sys.exit(1)

        data = np.genfromtxt(csv_path, delimiter=",", skip_header=1)
        if data.shape[1] > 3:
            # If time column exists at index 0, take indices 1, 2, 3
            data = data[:, 1:4]
        if data.shape[1] != 3:
            print(f"Error: CSV must have 3 columns [Voltage, Current, Temp]. Found shape {data.shape}.")
            sys.exit(1)

        result = predict_initial_soc(
            data,
            artifacts=artifacts,
            clip_physical=not args.no_clip,
            apply_current_correction=args.apply_correction
        )

        print("\n" + "=" * 60)
        print(f"INITIAL SOC INFERENCE REPORT: {csv_path.name}")
        print("="*60)
        print(f"Estimated Initial SoC: {result['predicted_soc_pct']:.2f} %")
        print(f"Raw Model Output:      {result['raw_model_soc_pct']:.2f} %")
        if args.apply_correction:
            print(f"Applied Correction:    {result['correction_delta_pct']:+.2f} %")
        print(f"Mean Operating Point:  V={result['operating_stats']['mean_voltage_V']:.3f}V, "
              f"I={result['operating_stats']['mean_current_A']:+.2f}A, "
              f"T={result['operating_stats']['mean_temp_C']:.1f}°C")
        print("=" * 60)
        print()
        return

    # If neither provided, print usage
    parser.print_help()
    print()
    print("Quick self-test example: python scripts/run_initial_soc_inference.py --test-sample 0")


if __name__ == "__main__":
    main()
