"""
data_loader.py — Panasonic 18650PF .mat file loader
Part of: AI-Based SoC Estimation of Li-ion Batteries (Group 1)
"""

from pathlib import Path
import numpy as np
import pandas as pd
import scipy.io as sio


def load_panasonic_mat(file_path, nominal_capacity: float = 2.9) -> pd.DataFrame:
    """
    Load a McMaster Panasonic 18650PF MATLAB .mat file into a standardised DataFrame.

    The dataset stores measurements inside a MATLAB structured array ``meas``.
    This function extracts all time-series signals, computes the ground-truth
    State-of-Charge (SoC) from the cumulative Ah counter, and returns a clean
    DataFrame ready for model training.

    Parameters
    ----------
    file_path : str or Path
        Absolute or relative path to the ``.mat`` file.
    nominal_capacity : float, optional
        Nominal cell capacity in Ah (default 2.9 Ah for Panasonic 18650PF).
        Note: at cold temperatures the effective usable capacity is lower
        (e.g. ~1.74 Ah at -20 °C), but the nominal value is used as the
        denominator so that SoC is reported relative to the rated capacity.

    Returns
    -------
    pd.DataFrame
        7-column DataFrame with columns:
        ``[Time_s, Voltage_V, Current_A, Ah, Battery_Temp_C, Chamber_Temp_C, SoC_pct]``

    Notes
    -----
    * Current sign convention: **negative = discharge**, positive = charge
      (follows the McMaster dataset convention).
    * SoC formula: ``SoC(t) = clip((Q_max + Ah(t)) / Q_max * 100, 0, 100)``
      where ``Ah(t)`` is the cumulative Ah counter (starts at 0, goes negative
      as the battery discharges).
    * Duplicate timestamps (Δt = 0) and large gaps (Δt > 1 s) exist in some
      files.  Callers should apply interpolation or masking before feeding
      sequences to LSTM / CNN-LSTM models.
    """
    mat = sio.loadmat(str(file_path))
    meas = mat["meas"]

    time     = meas["Time"][0, 0].flatten()
    voltage  = meas["Voltage"][0, 0].flatten()
    current  = meas["Current"][0, 0].flatten()
    ah       = meas["Ah"][0, 0].flatten()
    temp_batt = meas["Battery_Temp_degC"][0, 0].flatten()
    temp_amb  = meas["Chamber_Temp_degC"][0, 0].flatten()

    # Ground-truth SoC clamped to [0, 100] %
    soc = np.clip((nominal_capacity + ah) / nominal_capacity * 100.0, 0.0, 100.0)

    df = pd.DataFrame({
        "Time_s":         time,
        "Voltage_V":      voltage,
        "Current_A":      current,
        "Ah":             ah,
        "Battery_Temp_C": temp_batt,
        "Chamber_Temp_C": temp_amb,
        "SoC_pct":        soc,
    })

    return df


def load_all_panasonic(dataset_root, nominal_capacity: float = 2.9,
                       temperatures=None, drive_cycles=None):
    """
    Batch-load multiple Panasonic drive-cycle ``.mat`` files into a single
    concatenated DataFrame with additional ``Temp_label`` and ``Source_file``
    columns.

    Parameters
    ----------
    dataset_root : str or Path
        Path to the folder that directly contains the temperature sub-folders
        (e.g. ``25degC``, ``10degC``, …).

        **Dataset layout note:** The Panasonic dataset has two levels:
        - Outer level (``Panasonic 18650PF Data/``): contains ``-10degC``,
          ``-20degC``, ``0degC``, ``10degC`` and the inner folder.
        - Inner level (``Panasonic 18650PF Data/Panasonic 18650PF Data/``):
          contains only ``25degC``.

        Pass whichever root is appropriate. To load all temperatures, call
        this function twice (once per root) and concatenate the results.

    nominal_capacity : float
        Nominal capacity in Ah (default 2.9).
    temperatures : list of str, optional
        Sub-folder names to include, e.g. ``["25degC", "10degC", "0degC"]``.
        Defaults to all sub-folders whose names contain ``"degC"``.
    drive_cycles : list of str, optional
        Glob patterns for drive-cycle file names, e.g. ``["*UDDS*", "*NN*"]``.
        Defaults to every ``.mat`` file found in the ``Drive cycles`` folder.

    Returns
    -------
    pd.DataFrame
        Concatenated DataFrame with columns:
        ``[Time_s, Voltage_V, Current_A, Ah, Battery_Temp_C, Chamber_Temp_C,
        SoC_pct, Temp_label, Source_file]``

    Examples
    --------
    >>> from src.data_loader import load_all_panasonic
    >>> INNER = PROJECT_ROOT / "DATASET/Panasonic/Panasonic 18650PF Data/Panasonic 18650PF Data"
    >>> OUTER = PROJECT_ROOT / "DATASET/Panasonic/Panasonic 18650PF Data"
    >>> df_25  = load_all_panasonic(INNER, temperatures=["25degC"])
    >>> df_cold = load_all_panasonic(OUTER, temperatures=["10degC", "0degC", "-10degC", "-20degC"])
    >>> import pandas as pd
    >>> df_all = pd.concat([df_25, df_cold], ignore_index=True)
    """
    root = Path(dataset_root)
    if temperatures is None:
        temperatures = [p.name for p in root.iterdir()
                        if p.is_dir() and "degC" in p.name]

    dfs = []
    for temp_label in temperatures:
        # Try both capitalisation variants used in the dataset
        for dc_name in ("Drive cycles", "Drive Cycles"):
            drive_dir = root / temp_label / dc_name
            if drive_dir.exists():
                break
        else:
            print(f"  [SKIP] No Drive cycles folder found for {temp_label}")
            continue

        patterns = drive_cycles if drive_cycles else ["*.mat"]
        for pattern in patterns:
            for mat_file in sorted(drive_dir.glob(pattern)):
                df = load_panasonic_mat(mat_file, nominal_capacity)
                df["Temp_label"]  = temp_label
                df["Source_file"] = mat_file.name
                dfs.append(df)
                print(f"  Loaded {temp_label}/{mat_file.name}  "
                      f"({len(df):,} rows)")

    if not dfs:
        raise FileNotFoundError(
            f"No .mat files found under {dataset_root} for the given filters."
        )
    return pd.concat(dfs, ignore_index=True)
