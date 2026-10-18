# Inference Environment

This guide describes the frozen Group C seed-42 initial-SoC inference path. It does not declare a complete environment for historical training notebooks.

## Verified setup

The existing environment was inspected on Windows (`win32`) with Python 3.12.3:

| Package or runtime | Verified version / status | Role |
| --- | --- | --- |
| Python | 3.12.3 | Tested interpreter |
| NumPy | 2.5.2 | Input arrays and feature calculations |
| PyTorch | 2.6.0+cu124 | Frozen MLP and checkpoint loading |
| scikit-learn | 1.9.0 | Loads the serialized `StandardScaler` |
| SciPy | 1.18.1 | Installed transitively through scikit-learn; not directly imported by inference |
| joblib | 1.5.3 | Installed transitively through scikit-learn |

The verified PyTorch package is built for CUDA 12.4, and this machine reports CUDA available. The inference CLI defaults to `--device cpu`; CUDA can be selected with `--device cuda` where a compatible CUDA runtime and device are available. The CPU-default path does not require CUDA. Other operating systems, Python versions, and CPU-only package combinations have not been tested here.

## Create an environment

From the repository root, create a Python 3.12 virtual environment:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the versions in `requirements-inference.txt`. That file pins the verified CUDA 12.4 PyTorch wheel, so include the PyTorch CUDA 12.4 wheel index in addition to the normal package index:

```powershell
python -m pip install -r requirements-inference.txt --extra-index-url https://download.pytorch.org/whl/cu124
```

This documents an installation command; a newly created environment has **not** been built or tested as part of T006A. PyTorch wheels are platform-specific. For CPU-only or another platform, select the corresponding official PyTorch wheel and adjust the PyTorch pin to that verified build; the CUDA-specific pin in the manifest does not guarantee a universal installation. Do not assume a different PyTorch build reproduces the exact existing environment without testing it.

The direct runtime requirements are NumPy, PyTorch, and scikit-learn. SciPy, joblib, and threadpoolctl are scikit-learn dependencies and are not separately pinned here. Python's `pickle`, `csv`, `argparse`, `pathlib`, and `typing` modules are from the standard library.

## Required artifacts and inference

The repository must contain these frozen artifacts at their repository-relative paths:

- `results/models/ablation_group_c_seed_42.pt` — model state dictionary.
- `data/ablation_scaler_group_c.pkl` — fitted Group C `StandardScaler`.
- `data/initial_soc_scaler.npz` — channel normalization parameters.

Provide one 500-row CSV with voltage in volts, current in amperes, and temperature in degrees Celsius. Supported layouts are headerless V/I/T rows, exact header `Voltage,Current,Temperature`, or timestamped columns `Time,Voltage,Current,Temperature` with time in seconds. Run from the repository root:

```powershell
python scripts/run_initial_soc_inference.py --csv path/to/window.csv
```

The CLI uses CPU by default. Add `--device cuda` only when CUDA is available and a compatible PyTorch build is installed. `--test-sample INDEX` is a separate optional smoke-test mode that requires the local processed dataset `data/initial_soc_datasets.pkl`; this large artifact is intentionally excluded from Git. It is not required for ordinary CSV inference.

Raw battery dataset payloads and large processed window datasets are intentionally excluded from Git. Their local absence does not mean they have been deleted. The three frozen inference artifacts above are required to reproduce this model's prediction, but their availability depends on the reviewed repository contents and local storage.

## Existing inference checks

From the repository root, run both focused suites with the existing environment:

```powershell
.\.venv\Scripts\python.exe -B tests/test_inference_contract.py
.\.venv\Scripts\python.exe -B tests/test_inference_parity.py
```

These checks exercise the frozen model contract, artifacts, supported inputs, and numerical parity within their documented scope. They are not a new model-accuracy evaluation. T006A reran them in the existing environment; the outcomes are recorded in the ticket completion report. A newly created environment has not been validated by these tests.

## Portability work still pending

Several notebooks and `scripts/run_feature_ablation.py` contain machine-specific absolute project roots; update only under an approved follow-up ticket. `notebooks/04_ML_Baselines.ipynb` also uses `np.trapz`, which is unavailable in the verified NumPy 2.5.2 environment; it needs a separate targeted compatibility repair. Neither issue affects the frozen inference path tested here, and neither was changed in T006A.
