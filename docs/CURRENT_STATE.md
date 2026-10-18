# Project Current State

**Project:** Machine Learning-Based Initial State-of-Charge Estimation
**Last reviewed:** T003 inference repair and verification
**Project scope:** Approved initial-SoC estimation only  
**Evidence note:** Performance figures below are historical saved results, not freshly reproduced. T003 performed one inference smoke test only and did not recompute aggregate metrics.

## Objective and approved scope

Estimate the Ah-based reference SoC at the **start of an observation window** from a short dynamic measurement window of cell voltage, current, and battery surface temperature. The approved current scope is **AI-based initial SoC estimation for lithium-ion batteries**. Continuous SoC tracking, charging control, and hardware integration are not current deliverables. The historical Group 1 / Group 2 documents describe possible downstream uses, which remain outside scope unless separately approved.

## Current pipeline and selected model

The designated estimator uses a nominal 50-second window of 500 samples at approximately 10 Hz. It predicts reference SoC at the first sample, so its estimate is available after the observation has ended and refers to an earlier time.

Current documented data flow:

1. Select individual Panasonic drive-cycle recordings and assign files by temperature.
2. Extract finite, approximately 10 Hz windows from voltage, current, and battery temperature.
3. Use the logged Ah counter only to construct the target, not as an ML input.
4. Normalize the three signal channels using training-derived min/range values.
5. Extract 12 Group C features: initial, final, and change values for V/I/T; mean V and mean I; and voltage standard deviation.
6. Apply a training-fitted StandardScaler and a three-layer MLP, `12 → 64 → 32 → 1`, with ReLU hidden layers.
7. Convert model output to percent and optionally clip to `[0, 100]`.

The designated frozen artifact is `results/models/ablation_group_c_seed_42.pt`; its associated artifacts are `data/ablation_scaler_group_c.pkl` and `data/initial_soc_scaler.npz`. The intended architecture has 2,945 trainable parameters. T003 confirmed that the checkpoint and scalers load successfully and that the inference API produces a prediction for one saved 500-sample validation window.

## Data and preprocessing

- **Primary dataset:** McMaster Panasonic 18650PF measurements, described by the dataset README as one cell tested across 25, 10, 0, −10, and −20 °C conditions. The dataset includes voltage, current, Ah, battery temperature, and time in MATLAB files.
- **Main split:** train at 25/10/0 °C; validation at −10 °C; test at −20 °C.
- **Saved 50-second windows:** train `N=5,182`, validation `N=1,045`, test `N=871` (7,098 total), as recorded in `results/initial_soc_dataset_validation.txt`.
- **Reference label:** `clip((2.9 Ah + Ah) / 2.9 Ah × 100, 0, 100)`. This is a nominal-capacity Ah-based reference, not independently established absolute SoC. The guide's choice of reference capacity remains unresolved.
- **Coverage limitation:** saved −20 °C targets cover approximately 40–100% SoC. Low-SoC performance at that temperature is unmeasured.
- **Other supplied data:** LG HG2 MAT/CSV data and a prepared MATLAB FNN example, plus an EV lithium-ion SoC workbook. No external-cell results for the selected estimator were found.

## Completed experiments and saved results

The repository contains EDA, Coulomb-counting/EKF baselines, six observation-duration datasets, linear/MLP/GRU comparisons, a four-group/five-seed feature ablation, synthetic sensor perturbations, residual diagnostics, and two styles of residual correction analysis. Saved plots, CSV tables, notebooks, and reports describe these experiments.

**Keep these statistics distinct:**

| Result | Evaluation context | Saved RMSE |
|---|---|---:|
| Frozen Group C seed-42 model | −10 °C validation, 1,045 windows | **4.3005 percentage points** |
| Frozen Group C seed-42 model | −20 °C test, 871 windows | **10.1544 percentage points** |
| Group C five-seed mean | −10 °C validation; mean across seeds 42, 100, 200, 300, 400 | **4.1622 ± 0.1980 percentage points** |
| Grouped correction, clipped | Three −10 °C evaluation files, 315 windows; existing grouped result table | **2.8535 percentage points** |
| Grouped correction, clipped, post-hoc | −20 °C test; existing grouped result table | **7.8633 percentage points** |

These are different experiments and model states. In particular, 4.1622 is the five-seed mean; it is not the saved seed-42 model's validation score. None of these figures was freshly reproduced in T000–T003. The grouped-correction result tables exist, but the code that generated those grouped tables was not located during T000. Notebook 07 contains a separate random-window correction experiment.

## Known bugs and blocked functionality

- T003 repaired malformed strings and the duplicate `--no-clip` help text in `scripts/run_initial_soc_inference.py`; the module now parses and imports, and CLI `--help` works.
- `notebooks/02_Baselines.ipynb` predicts SoC using subtraction of current. Given the declared negative-discharge convention, negative discharge current increases predicted SoC in the process step. The EKF needs a correctness repair and focused verification. The saved EKF metric does not establish correctness.
- `notebooks/04_ML_Baselines.ipynb` calls `np.trapz`; the inspected project environment's NumPy 2.5.2 does not expose `np.trapz` and does expose `np.trapezoid`.
- Several notebooks and `scripts/run_feature_ablation.py` contain a machine-specific absolute root pointing to an earlier project copy. New development should use repository-relative paths.
- The inference script accepts windows of ten or more samples although its documented model contract specifies 500 samples. It does not enforce the specification's timestamp-gap and finite-value requirements.
- Processed windows do not retain explicit source filename/start-index identifiers; this limits durable provenance and makes trajectory grouping fragile.
- The timing contract predicts SoC at the window start but returns it after about 50 seconds. No current-scope implementation verifies that the returned state is advanced to the time of use.

## Testing and validation gaps

- No automated test suite or standard test command was found.
- Notebooks 03 and 05 have no saved code-cell execution history/output; Notebook 04 has only one executed code cell. Notebooks 06 and 07 have saved execution counts and outputs. Other notebook outputs are mixed or partial.
- A saved dataset-validation text report records window-count reconciliation, split checks, and five example target checks; its producer was not found, so it is historical evidence rather than a repeatable test command.
- T003 smoke-tested the loaded frozen model on one saved validation window (`N=500`) and checked the API's correction/clipping switch combinations. This is a single-window execution check, not an accuracy evaluation or numerical-consistency study.
- Full input-contract validation and numerical consistency between training and inference preprocessing remain unverified. The inference API still accepts windows shorter than the specified 500 samples and does not enforce timestamp-gap or finite-value requirements.
- The EKF's current-sign behavior is defective. Existing saved EKF results must not be presented as a verified correct baseline.
- Sensor-robustness results use synthetic perturbations, not measurements from deployment sensors.
- Current data cover one Panasonic cell and do not verify cross-cell or cross-chemistry generalization.
- Reports contain conflicting values for the frozen model versus ablation means, architecture parameter counts, training settings, capacity estimates, and experiment counts. Reconcile against code and saved artifacts before reuse.

## Environment and Git status

- Existing `.venv/pyvenv.cfg` identifies Python 3.12.3. During T000, the interpreter identified NumPy 2.5.2. Other listed package versions are recorded in the environment metadata, not a pinned dependency manifest.
- PyTorch, scikit-learn, NumPy, pandas, SciPy, Jupyter, and plotting libraries are present in the inspected environment. No dependency lockfile or project dependency manifest was found.
- Git is initialized on `main`; the reviewed initial checkpoint has been pushed to the approved private GitHub repository and local `main` is synchronized with `origin/main`.
- `.gitignore` excludes raw dataset payloads, the large processed window dataset, local environments, and generated report exports. Excluded source data remain local; their absence from Git is not deletion.

## Open technical and research questions

1. Is the nominal 2.9 Ah reference denominator approved, given measured capacity variation and ageing?
2. What accuracy/range acceptance criteria apply, especially at −20 °C and near charging decision thresholds?
3. Is the grouped correction producer available elsewhere, or are its aggregate tables legacy outputs without recoverable code?
4. Which report is authoritative, and which discrepancies should be corrected under a future approved ticket?
5. Is a new independent evaluation set required after prior post-hoc inspection of the current test set?
6. Which external dataset is required, if external validation is part of the academic rubric?
7. What deadlines, report format, viva requirements, and mandatory demonstrations apply?

Future tickets must preserve the approved initial-SoC-only scope unless the user explicitly approves a scope change.
