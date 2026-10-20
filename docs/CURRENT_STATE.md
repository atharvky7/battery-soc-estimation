# Project Current State

**Project:** Machine Learning-Based Initial State-of-Charge Estimation for Lithium-Ion Batteries
**Last reviewed:** T007B authoritative results register
**Project scope:** Approved initial-SoC estimation only  
**Evidence note:** Performance figures below are historical saved results, not freshly reproduced. T003 performed one inference smoke test, T004A performed numerical parity checks, and T004B checked the input contract; none recomputed aggregate metrics.

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

These are different experiments and model states. In particular, 4.1622 is the five-seed mean; it is not the saved seed-42 model's validation score. These remain historical saved results and were not freshly reproduced by the provenance audit. The grouped-correction result tables exist, but their original producing script (`scripts/run_grouped_correction.py`) is unavailable in the current repository. Notebook 07 contains a separate random-window correction experiment.

## Experiment provenance audit (T007A)

`docs/RESULTS_PROVENANCE_AUDIT.md` is the detailed, evidence-labeled audit of the saved tables, source code, selected artifact metadata, notebook outputs, and report claims. It records source-traceable results separately from verified artifact properties, provenance-incomplete tables, and inferences or unknowns. The audit inspected saved values and performed arithmetic; it did not run inference, training, notebooks, or dataset evaluation.

- The frozen Group C seed-42 checkpoint's historical validation RMSE is **4.3005 pp**. **4.1622 ± 0.1980 pp** is a separate five-seed validation mean and population standard deviation; neither statistic was freshly reproduced in T007A.
- The historical **4.0331 pp** validation result is from a separate 13-feature MLP experiment, not the frozen 12-feature Group C model.
- The saved GRU checkpoint contains **14,305** parameter elements, conflicting with historical reports of 60,869 and “about 20×” model size. This is a checked property of the present checkpoint; a different historical run cannot be ruled out.
- The grouped correction tables have no recoverable producer in this repository. Saved-table arithmetic gives an approximate pooled clipped validation RMSE of **2.9054 pp**, which conflicts with the historical overall corrected RMSE claim of **2.60**. This derived aggregation is not a newly evaluated model metric.
- The −20 °C test data were inspected post hoc in historical residual and correction analyses. Existing test results must be labeled historical post-hoc analysis, not independent final validation.
- Historical report conflicts and missing run-to-artifact manifests remain unresolved. Preserve original reports and result tables until a separately approved reconciliation specifies traceable corrections; do not treat matching filenames or current hashes as proof of historical run identity.

T007A's read-only audit is complete. T007B's results register and reporting dispositions are complete within documentation scope; unresolved scientific claims and evidence gaps remain recorded for future approved work.

`docs/RESULTS_REGISTER.md` now provides the evidence-linked register and proposed public-claims policy based on T007A. T007B is complete within documentation and reporting-disposition scope; no experiments or historical report edits were made. Disputed scientific interpretations, the absent grouped-correction producer, missing run manifests, and the need for an agreed independent evaluation remain open. Treat the register's dispositions as the reviewed reporting record; it does not establish newly reproduced model accuracy.

## Known bugs and blocked functionality

- T003 repaired malformed strings and the duplicate `--no-clip` help text in `scripts/run_initial_soc_inference.py`; the module now parses and imports, and CLI `--help` works.
- T005A corrected the EKF SoC state prediction in `notebooks/02_Baselines.ipynb`: negative current (discharge) now lowers SoC, and positive current (charge/regen) raises it. The RC polarization and terminal-voltage signs were reviewed and left unchanged: discharge drives polarization negative, while `I * R_IN` and `Vth` lower terminal voltage.
- T005B resolved the EKF loop's sample convention in deterministic tests: sample 0 receives a voltage measurement update without state propagation or process-noise propagation. For each `k >= 1`, state prediction uses the interval-start current `current[k-1]` and actual interval duration `dt_arr[k]`; the voltage measurement update at sample `k` uses `current[k]`.
- Deterministic tests verified measurement innovation, the OCV Jacobian, innovation covariance, Kalman gain, Joseph-form covariance update, state flow, and boundary behavior. These tests used synthetic cases; the full notebook and a real-data EKF trajectory were not run. The full real-data loop and model accuracy remain unverified.
- The EKF experiment initializes `x0 = [soc_ref[0], 0]`, so it tracks a trajectory from known reference SoC and is not directly comparable to the ML estimator's unknown initial-SoC task. Historical EKF metrics predate the T005A/T005B correctness repairs and have not been recomputed.
- The notebook replaces nonpositive timestamp intervals with the median positive interval. This remains a data-quality limitation; timestamp quality on real trajectories has not been evaluated.
- T006B replaced machine-specific or working-directory-dependent root assignments in notebooks 01–07 with marker-checked resolution from either the repository root or `notebooks/`. Other launch directories are rejected with a clear error. `scripts/run_feature_ablation.py` now derives its root from its own `__file__`.
- T006B replaced `np.trapz` with `np.trapezoid` in Notebook 04 while preserving the current array, timestamps, integration axis, sign, and downstream Ah conversion. A small synthetic irregular-time integration check passed; the notebook and historical results were not run or regenerated.
- `scripts/run_initial_soc_inference.py` requires exactly 500 rows and three finite real numeric V/I/T channels for the frozen model. Units (V, A, °C), synchronous sampling, and V/I/T ordering are caller contracts; measurement values cannot independently prove these semantics.
- Optional timestamps must be 500 finite values, strictly increasing, with elapsed time from 40 to 60 seconds and maximum gap no greater than 0.5 seconds, as specified in `docs/INITIAL_SOC_MODEL_SPEC.md` section 2. If omitted, inference reports timing as unverified. Passing these duration and gap checks does not establish uniform 10 Hz sampling; the specification does not define a per-interval tolerance around nominal 10 Hz.
- CLI CSV accepts headerless V/I/T rows, the exact `Voltage,Current,Temperature` header, or `Time,Voltage,Current,Temperature` with seconds. Other layouts are rejected rather than guessed.
- Processed windows do not retain explicit source filename/start-index identifiers; this limits durable provenance and makes trajectory grouping fragile.
- The timing contract predicts SoC at the window start but returns it after about 50 seconds. No current-scope implementation verifies that the returned state is advanced to the time of use.

## Testing and validation status

- `tests/test_inference_parity.py` provides seven focused checks for training/inference feature contract, saved preprocessing artifacts, the frozen architecture and weights, independent forward calculation, array/tuple input equivalence, and float precision paths. Run with `\.venv\Scripts\python.exe -B tests/test_inference_parity.py` from the repository root.
- `tests/test_inference_contract.py` provides 17 deterministic checks for array/tuple formats, exact sample/channel dimensions, malformed/nonnumeric/nonfinite data, timestamp rules, supported and malformed CSV layouts, CLI behavior, and frozen prediction values. The T004B run passed all 17. The T004A suite was rerun and passed all seven tests on five saved validation windows.
- T004A ran all seven checks against five saved validation windows (`X_val` indices 0, 261, 522, 783, and 1044; 500 samples each), after the test's available-memory guard allowed loading the trusted local processed pickle. Maximum observed absolute differences were: channel normalization `5.96046448e-08` (tolerance `1.2e-7`); 12 features `3.7252903e-09` (`2e-7`); standardized features `8.94069672e-07` (`4e-6`); independent unrounded forward outputs `7.39097595e-06` percentage points (`5e-5`); and rounded API output versus reference `0` (`1e-4`). These are parity checks on selected windows, not model accuracy or broad validation.
- The same seven checks previously passed through a forced deterministic synthetic-input fallback. That is execution and numerical-path evidence only; synthetic predictions are not accuracy evidence. The normal T004A-C rerun used saved validation windows, not the fallback.
- The parity tolerances use absolute comparisons (`rtol=0`). The standardized-feature bound accommodates an observed float32 endpoint-delta round-trip error amplified by the small frozen temperature-delta scaler. This bound does not establish acceptable predictive error.
- Notebooks 03 and 05 have no saved code-cell execution history/output; Notebook 04 has only one executed code cell. Notebooks 06 and 07 have saved execution counts and outputs. Other notebook outputs are mixed or partial.
- A saved dataset-validation text report records window-count reconciliation, split checks, and five example target checks; its producer was not found, so it is historical evidence rather than a repeatable test command.
- T003 smoke-tested the loaded frozen model on one saved validation window (`N=500`) and checked the API's correction/clipping switch combinations. This is a single-window execution check, not an accuracy evaluation or numerical-consistency study.
- T004A established numerical consistency within the five-window and dtype scope above. T004B completed the documented input-contract checks. No new model-accuracy results were obtained, and no scientific artifacts were modified. Inference cannot verify timing when timestamps are absent or establish channel units and semantics from numeric arrays alone. Broader model-accuracy verification was not performed by either ticket.
- T005A's 11 focused prediction/sign tests and T005B's 14 focused loop tests passed on rerun (25/25 total). The loop tests exercise deterministic synthetic cases and verify that sample 0 receives measurement correction without prediction, that later predictions use `current[k-1]` and `dt_arr[k]`, and that measurement updates use `current[k]`; they also verify innovation, Jacobian, gain, and Joseph covariance behavior. This is mathematical correctness evidence within those cases, not real-data evaluation or model-accuracy evidence.
- Independent review confirmed the notebook source changes are confined to cells 12 and 15 and preserve all other cell sources, outputs, execution counts, and metadata. No full notebook was executed and no historical EKF metrics were recomputed. Historical EKF metrics predate both correctness repairs and must not be attributed to the repaired implementation.
- T005 deterministic correctness verification is complete. Real-data EKF evaluation, OCV validity, timestamp quality, and baseline-comparison fairness remain unresolved follow-ups. The known-initial-SoC EKF setup differs from the ML initial-SoC estimator.
- T006B's seven portability tests passed, as did the 11 EKF baseline tests and 14 EKF-loop tests. Static syntax checks passed for all notebook code cells and the feature-ablation script. Notebook preservation checks found only the intended setup source cell in each notebook changed; Notebook 04 also has the approved `np.trapezoid` source edit. All notebook outputs, execution counts, identifiers, metadata, ordering, and unrelated source cells match Git HEAD. Existing `SyntaxWarning` escape messages in an unchanged Notebook 06 cell remain.
- T006B did not execute notebooks, feature ablation, training, or experiments and did not regenerate saved metrics, plots, or other historical results. The supported notebook launch directories are the repository root and `notebooks/`; running from another directory is unsupported and raises an error.
- Sensor-robustness results use synthetic perturbations, not measurements from deployment sensors.
- Current data cover one Panasonic cell and do not verify cross-cell or cross-chemistry generalization.
- Reports contain conflicting values for the frozen model versus ablation means, architecture parameter counts, training settings, capacity estimates, and experiment counts. The T007A audit and T007B register document these conflicts and proposed dispositions. Original reports remain unchanged; any further corrections require an explicitly approved ticket.

## Environment and Git status

- The existing, tested inference environment is Windows (`win32`) with Python 3.12.3, NumPy 2.5.2, PyTorch 2.6.0+cu124 (CUDA 12.4 build; CUDA was available), scikit-learn 1.9.0, SciPy 1.18.1, and joblib 1.5.3. The inference CLI defaults to CPU; CUDA use requires a compatible device/build.
- `requirements-inference.txt` pins the direct frozen-inference dependencies NumPy, PyTorch, and scikit-learn to versions verified in that existing environment. SciPy, joblib, and threadpoolctl are transitive dependencies and are not fully locked. `docs/ENVIRONMENT.md` records setup, invocation, required artifacts, and environment limitations.
- T006A reran `tests/test_inference_contract.py` (17/17 passed) and `tests/test_inference_parity.py` (7/7 passed using five saved validation windows), CLI `--help`, and `pip check`; all passed. This verifies the existing environment only. No fresh environment was created or installed, so fresh-install reproducibility has not been demonstrated. No model-accuracy results were produced.
- The inference dependency manifest covers the frozen inference path, not every historical notebook or experiment. No complete lockfile for the entire project exists.
- A fresh virtual environment has not been installed or tested. `requirements-inference.txt` documents the verified existing Windows/CUDA setup but does not fully lock transitive dependencies or establish fresh-install reproducibility.
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
