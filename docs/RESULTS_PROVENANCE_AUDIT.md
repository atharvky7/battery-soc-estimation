# T007A — Experiment Provenance Audit

**Project:** Machine Learning-Based Initial State-of-Charge Estimation for Lithium-Ion Batteries  
**Repository baseline:** `493fc674f57cae986e93882f40030e227c6f6696`  
**Scope:** Read-only inspection of source, small saved result tables, existing notebook outputs, Git metadata, and selected small artifact metadata. Only this document was created.

No training, notebook execution, model forward pass, dataset evaluation, raw MAT loading, or processed-dataset loading was performed. Arithmetic below operates on saved CSV values. Checkpoint inspection used `torch.load(..., map_location="cpu", weights_only=True)` solely to enumerate tensor shapes/counts; no experiment module was imported. Historical metrics remain historical.

## 1. Findings and evidence labels

- The frozen Group C seed-42 model has saved validation RMSE **4.3005206924 percentage points (pp)** and saved post-hoc test RMSE **10.1544424473 pp**. These are source-traceable historical results, not accuracy results reproduced by this audit.
- **4.1621638376 ± 0.1979870968 pp** is the mean and population standard deviation of five separate Group C validation runs. Saved-row arithmetic independently confirms it; it is neither the seed-42 score nor an ensemble score or confidence interval.
- Historical **4.0331125614 pp** belongs to Notebook 04's **13-feature** summary MLP, not the frozen 12-feature Group C checkpoint.
- Grouped-file correction tables report **2.853517 pp** on 315 validation evaluation windows and **7.863251 pp** on 871 post-hoc test windows. Their original producer is missing. Notebook 07 is a different random-window correction experiment.
- Several reports conflate these results. Additional conflicts concern GRU parameter counts, corrected validation aggregates, sensor-scenario counts, and strong claims of independence or physical causality.

Labels used here:

| Label | Meaning |
| --- | --- |
| **Verified artifact property** | Directly checked now: file presence/hash, saved tensor dimensions/counts, or arithmetic on saved table entries. |
| **Source-traceable only** | Identifiable code references the relevant artifact and writes the named metric table; execution-to-artifact identity and accuracy have not been independently reproduced. |
| **Provenance-incomplete** | A table or report exists, but its producing procedure or necessary run evidence is missing or inconsistent. |
| **Inference / unknown** | A conclusion suggested by code or matching values, or a question the available evidence cannot resolve. |

Notebook locations below use **zero-based cell indices** and source-line numbers within the indicated cell, after T006B. CSV `L` references are physical file lines, including the header as L1. No notebook output is treated as proof of current source execution, particularly after approved source repairs.

## 2. Selected model and data lineage

### 2.1 Frozen estimator

The designated file is `results/models/ablation_group_c_seed_42.pt`. Current tensor shapes are `(64,12)`, `(64,)`, `(32,64)`, `(32,)`, `(1,32)`, `(1,)`, matching a **12 → 64 → 32 → 1** biased MLP with ReLU hidden layers. Parameter count is `(12×64+64) + (64×32+32) + (32×1+1) = 2,945`. This verifies the saved architecture property, not the training history.

`scripts/run_feature_ablation.py:85–145` defines all 13 candidate features and Group C as the first 12, in this order:

`V0, I0, T0, Vend, Iend, Tend, dV, dI, dT, meanV, meanI, stdV`.

These are computed from channel-normalized V/I/T arrays; deltas are final minus initial, and standard deviation uses NumPy's default population definition. Group C excludes integrated current (`Q_window`). The fitted 12-feature StandardScaler is `data/ablation_scaler_group_c.pkl`; the preceding channel min/range normalizer is `data/initial_soc_scaler.npz`. Scaler fitting on training features is at `scripts/run_feature_ablation.py:188–197`; architecture is at lines 150–161. `scripts/run_initial_soc_inference.py:62–110, 189–243, 284–316` provides the current artifact-loading, feature, forward, percent-conversion, optional correction, and clipping path.

### 2.2 Dataset definition and recoverable configuration

| Item | Source and assessment |
| --- | --- |
| Primary source / split | `notebooks/03_Feature_Engineering.ipynb`, cell 3 L2–30: Panasonic drive-cycle files; train at 25/10/0 °C, validation at −10 °C, test at −20 °C. File matching/exclusion is explicit code, not a checked-in dataset manifest. |
| Reference target | Notebook 03 cell 5 L12–16, L58–63: `clip((2.9 + Ah)/2.9*100, 0, 100)` at the first sample of each window. It is a nominal-capacity Ah-based reference, not an independent absolute-SoC measurement. |
| Window / timing | Notebook 03 cell 5 L15–55 and cell 6 L2–3: nominal duration ×10 samples; stride 500; duration tolerance ±20%; reject gaps >0.5 s and NaN channels. At 50 s there are 500 samples; nominal uniform 10 Hz would span 49.9 s. Historical extraction does not enforce all newer inference-contract checks (for example strictly increasing finite timestamps). |
| Normalization | Notebook 03 cell 8: channel min/range fitted on all samples of training 50 s windows, then applied to every split/duration; target divided by 100. Cell 12 saves the normalizer and `data/initial_soc_datasets.pkl`. |
| Saved counts | `results/initial_soc_dataset_validation.txt:L15, L23, L79–85`: train 5,182, validation 1,045, test 871; total 7,098 at 50 s. The report states 46 source files; its producer was not found. Counts are saved-report evidence, not revalidated against data here. |
| Training configuration | `scripts/run_feature_ablation.py:167–172, 214–288`: seeds `[42,100,200,300,400]`, batch 64, up to 150 epochs, patience 15, Adam learning rate 0.001, weight decay 0.00001, MSE on normalized targets; select minimum validation loss and save that state. RNGs are seeded; this is not a full hardware/deterministic-run specification. |
| Single-seed selection record | `results/feature_ablation_seeds.csv:L12`: seed 42, best epoch 31, 2,945 parameters. Other Group C seeds have lower saved errors; seed 42 is the designated frozen run, not the minimum-error seed. Reason for freezing this seed rather than another is not recorded as a formal decision. |
| Metric convention | Ablation script lines 277–284: predictions and targets ×100, then RMSE/MAE; no clipping or residual correction in that evaluation. |
| Test access | Ablation script loads only training and validation arrays. Notebook 04 cell 24 and Notebooks 05–07 explicitly inspect −20 °C test data. The script's “LOCKED” comment applies to that experiment, not the project's complete history. |

Notebook 03 has no saved execution counts or outputs. Notebook 04's relevant result cells also have no saved execution evidence; Notebook 05 has none. Notebooks 06 and 07 contain saved prediction-summary and correction outputs consistent with their CSVs. These differences affect confidence in the execution trail, not whether the source exists.

### 2.3 Present artifact identity versus historical provenance

Current SHA-256 fingerprints (computed in T007A):

| Artifact | SHA-256 |
| --- | --- |
| `results/models/ablation_group_c_seed_42.pt` | `ec35dbcf3c857f5b4f16444c49674817fe7976d67d4b825630b43e516d165d5e` |
| `data/ablation_scaler_group_c.pkl` | `ca6a9c59f0c851f87bf8a425d74a076cdbc15d768e1d69cbdf10c3c50112478f` |
| `data/initial_soc_scaler.npz` | `18ceae89655e5339a899feeba0d5c6098905ca8440d774c2c93fac5562d8b165` |
| `results/models/mlp_summary_50s.pt` | `6129621f6272eda41a583128c03e28d492b7b4c8592b779573791718fac4d56b` |
| `data/initial_soc_summary_scaler_50s.pkl` | `6d638f4737814c79c724c28ee98fe99835eac67693e3d61d76b3b4fbaa42a2ef` |
| `results/models/gru_raw_50s.pt` | `f06e188f8aebd0e70e8150e6346365a8a40992865634913ce8dcec5f5f71ac01` |

The selected artifacts, legacy comparison artifacts, and result CSVs are Git-tracked. Some source reports used below, including the model specification and progress/speech/meeting reports, are currently local untracked files; their presence locally must not be represented as publication in GitHub.

The ablation script explicitly saves `ablation_group_c_seed_42.pt` and the Group C scaler via its group/seed naming templates (lines 195–197, 287–288), while writing the per-seed/summary CSVs at lines 330–335. Notebook 05 loads this checkpoint/scaler in cell 3 and writes the matching clean validation score; Notebook 06 loads the same names and has corresponding saved residual outputs. Together these support a coherent source-level chain.

**Limit:** filenames, matching numbers, Git membership, and hashes computed now do not prove that the present bytes produced the historical metrics. No original run manifest binds a data hash, code commit, environment, scaler hash, checkpoint hash, predictions, and metrics. The selected checkpoint is a state dictionary without a training-run manifest. T004 parity tests establish selected numerical-path agreement, not aggregate accuracy or original training provenance.

## 3. Major result ledger

All RMSE values are in **SoC percentage points**, not relative percentages. No row is a newly reproduced accuracy result.

| Result / exact saved value | Primary evidence | Dataset / model / artifact | Producer and recoverability | Classification |
| --- | --- | --- | --- | --- |
| Seed-42 validation **4.300520692393889** | `results/feature_ablation_seeds.csv:L12`; identical `results/robustness_clean_vs_perturbed.csv:L2`; rounded `results/residual_summary.csv:L2` | −10 °C, 1,045 windows; raw 12-feature Group C seed 42; designated checkpoint and scalers above | Ablation script lines 214–306, Notebook 05 cell 5 L43–51 and cell 15 L19–28, Notebook 06 cells 5/7. Config recoverable at source level; original run identity not bound. | **Source-traceable only**; cross-table agreement verified. |
| Five-seed Group C validation **4.16216383761665 ± 0.19798709679269677** | `results/feature_ablation_seeds.csv:L12–16`; `results/feature_ablation_summary.csv:L4` | Same validation split; five independently trained Group C checkpoints for seeds 42/100/200/300/400 | Ablation script lines 167–172, 308–335. Mean/population std calculations recoverable and checked. | **Source-traceable only** for the model runs; **verified arithmetic** for aggregation. |
| Frozen seed-42 test **10.15444244732401** | `results/robustness_clean_vs_perturbed.csv:L5`; rounded `results/residual_summary.csv:L3`, `results/grouped_correction_test_posthoc.csv:L2` | −20 °C, 871 windows; raw Group C seed 42, not a five-seed mean | Notebook 05 cell 5 L53–61 and cell 15 L3–5/L24–28; Notebook 06 cell 5 L65–71 and cell 7 L39–43. Test procedure identifiable; post-hoc use explicit. | **Source-traceable only**, historically queried test split. |
| Historical MLP Summary validation **4.033112561407553** | `results/initial_soc_model_comparison_50s.csv:L4` | −10 °C, nominal 50 s; **13** summary features including signed integrated current; `results/models/mlp_summary_50s.pt`, `data/initial_soc_summary_scaler_50s.pkl`, channel normalizer | Notebook 04 cell 3 `build_summary_features`, cell 5 scaling, cells 11–14 training/evaluation and CSV write (cell 14 L19–25). Config: seed initialized to 42 once in cell 1, Adam/batch64/max150/patience15; model creation and earlier operations consume RNG state. No independent per-model reset or multi-seed mean. No saved result-cell outputs. | **Source-traceable only**, with incomplete execution/run linkage. Must not be attributed to Group C. |
| Grouped correction validation **2.853517** | `results/grouped_correction_validation.csv:L9`, variant D, grouped evaluation split; baseline L6 **4.4961715**; un-clipped current correction L8 **2.8595421** | Claimed −10 °C evaluation: HWFET/Cycle 2/Cycle 3, 315 windows; nominal frozen seed-42 model + calibration from six other files + clipping | `results/grouped_correction_files.csv:L2,L7,L8` specifies membership. Progress report L557 names absent `scripts/run_grouped_correction.py`; no producer found. Coefficients appear in inference code, but fit and file-to-window reconstruction cannot be recovered completely. | **Provenance-incomplete**; saved improvement only. |
| Grouped correction post-hoc test **7.863251** | `results/grouped_correction_test_posthoc.csv:L5`, variant D; baseline L2 **10.154443**; un-clipped current correction L4 **7.9309535** | Claimed transfer of the grouped validation correction to −20 °C, 871 windows | Same absent grouped producer; report/spec/inference describe the intended correction, not the original producing run. | **Provenance-incomplete**, explicitly post-hoc. |

The earlier comparison's other rows are LR 8.7478574037 (L2), Ridge 8.7525883797 (L3), flattened MLP 10.0160660431 (L5), and GRU 4.5395476038 (L6). Notebook 04 cells 9 and 14 identify their producers; `lr_summary_50s.pkl`, `ridge_summary_50s.pkl`, `mlp_flattened_50s.pt`, and `gru_raw_50s.pt` are present. These are within-experiment source comparisons, not a controlled comparison of the later five-seed Group C mean against a five-seed GRU mean.

## 4. Related experiments and their evidence limits

| Saved family | Source location and result context | Status / limitation |
| --- | --- | --- |
| Duration sweep | `results/initial_soc_duration_comparison.csv:L2–7`; Notebook 04 cell 17 L15–37 writes `mlp_summary_sweep_{duration}s.pt` and metrics. 50 s = **4.3879118984**, 100 s = **4.3025652575** validation RMSE. | Source-traceable 13-feature models, not frozen Group C duration experiments. Each duration has different accepted windows (saved validation report), one run per duration in current code, and no uncertainty estimates. The 50→100 s difference is **0.0853466409 pp**, not evidence of a statistically established optimum. Additional non-`sweep` duration checkpoints exist without a located current producing loop; do not silently attach them to these rows. |
| Early Q-window ablation | `results/initial_soc_q_window_ablation_50s.csv:L2–5`; Notebook 04 cell 21 L2–9 | **Provenance-incomplete empirical claim:** code writes literal values 8.75/8.75 for LR and 4.13/4.38 for “avg 3 seeds” MLP. No three-seed computation, seed list, or matching per-seed ledger is present in this cell. The later four-group/five-seed script is a different experiment. |
| Sensor robustness | `results/robustness_validation_summary.csv`, `robustness_clean_vs_perturbed.csv`, `robustness_soc_bin_analysis.csv`; Notebook 05 cells 7, 10, 12, 15 | Source-traceable synthetic perturbations of the frozen Group C model. Current source and saved validation table contain **21 rows including baseline**, plus three post-hoc test checks in cell 15; these do not establish the reported 28 scenarios. No real sensor measurements, independent seeds/uncertainty study, or hardware latency demonstration is established. |
| Residual diagnostics | `results/residual_summary.csv`, `residual_feature_correlations.csv`, `residual_soc_bins.csv`; Notebook 06 cells 5, 7, 11, 13 | Source-traceable raw frozen predictions, residual `prediction − reference`, bias/variance and correlation analysis. Saved outputs exist. Correlation does not isolate current/temperature/ageing/SoC causality. |
| Random-window correction | `results/residual_correction_validation.csv:L2–9`, `residual_correction_test_posthoc.csv:L2–5`; Notebook 07 cells 7, 9, 11, 13 | Source-traceable and supported by saved notebook outputs, but not independently reproduced. Separate experiment: eval D **2.7414563**, test D **7.698302**. Neither is the grouped-file result. |
| Tracking baselines | `results/baseline_metrics.csv:L2–5`; Notebook 02 cell 18 saves the table, cell 15 saved output includes EKF **2.4021** | Historical outputs predate T005A/T005B correctness repairs. Current EKF starts at reference initial SoC (cell 14 L23) and tracks a full trajectory. Its 25 focused deterministic tests do not revalidate old trajectory accuracy. CC/EKF trajectory scores and ML unknown-initial-SoC window scores are different tasks/scopes. |

## 5. Saved-table arithmetic performed in T007A

### 5.1 Five-seed aggregation

Group C saved RMSEs (CSV L12–16) are:

`[4.300520692393889, 4.446507941810845, 3.972524935222171, 3.9174173866621924, 4.1738482319941514]`.

Using `statistics.mean` and `statistics.pstdev` gives **4.162163837616649** and **0.19798709679269677**. These match summary L4 within floating-point rounding (mean difference −8.88×10⁻¹⁶; std difference 0). Population std uses divisor 5, matching `np.std` in the script. It is not the sample standard deviation, a standard error, or uncertainty over new trajectories.

All groups' saved RMSE summary arithmetic also matches:

| Group | Recomputed mean from saved rows | Population std | Summary row |
| --- | ---: | ---: | --- |
| A | 7.072797583517889 | 0.34643656033598214 | `feature_ablation_summary.csv:L2` |
| B | 5.591939877054087 | 0.21393409877660005 | L3 |
| C | 4.162163837616649 | 0.19798709679269677 | L4 |
| D | 4.202027637528363 | 0.3186042299198248 | L5 |

Differences between computed means and CSV summaries are at most 8.89×10⁻¹⁶; computed population standard deviations match the parsed saved values. This verifies table aggregation, not the underlying predictions. Group D minus Group C mean is approximately **0.0398638 pp**; this small descriptive difference alone is not a significance test or proof that integrated charge is physically irrelevant.

### 5.2 Grouped file accounting and correction deltas

`results/grouped_correction_files.csv:L2–10` records nine contiguous half-open index ranges, starting at 0 and ending at 1,045. Every `End_Idx − Start_Idx` matches its `Window_Count`; consecutive ranges meet without a gap or overlap.

- Calibration: six files, **730** windows (69.8565% of windows).
- Evaluation: three files, **315** windows (30.1435%).
- Evaluation files: HWFET (95), Cycle 2 (113), Cycle 3 (107).
- This is about 70/30 **by windows**, with a 6/3 split **by files**. These counts validate the saved ledger's arithmetic only; the original mapping of processed rows to raw filenames has not been checked.

| Table / subset | Baseline A RMSE | Corrected D RMSE | D minus A from saved decimals | Interpretation |
| --- | ---: | ---: | ---: | --- |
| Grouped calibration, validation CSV L2/L5 | 4.2132907 | 2.9275434 | −1.2857473 pp | Calibration fit subset, not held-out correction performance |
| Grouped evaluation, validation CSV L6/L9 | 4.4961715 | 2.853517 | −1.6426545 pp | Reported held-out-file correction improvement; 36.5345% relative RMSE reduction |
| Grouped post-hoc test CSV L2/L5 | 10.154443 | 7.863251 | −2.2911920 pp | Reported post-hoc improvement; 22.5634% relative reduction |
| Random-window evaluation, residual validation CSV L6/L9 | 4.19799 | 2.7414563 | −1.4565337 pp | Different split/procedure from grouped evaluation |
| Random-window post-hoc test CSV L2/L5 | 10.154443 | 7.698302 | −2.4561410 pp | Different fitted correction coefficients |

All four correction tables' `Delta_RMSE` and `Delta_MAE` values agree with subtraction of the displayed metric columns within **5.31×10⁻⁷ pp**. The tiny discrepancy is consistent with saved decimal rounding. This does not verify any original inference.

Under the saved grouped membership/counts, pooling disjoint-subset RMSEs requires squared-error weighting:

`pooled_RMSE = sqrt((730 * calib_RMSE**2 + 315 * eval_RMSE**2) / 1045)`.

This yields baseline **4.3005205361 pp**, consistent with the seed-42 all-validation score; current-corrected C **3.0246775598 pp**; clipped D **2.9054278081 pp**. The latter is inconsistent with the specification's overall corrected **2.60%**. The pooled value is a derived check on saved subset aggregates, not a newly evaluated metric or proof of correct membership.

## 6. Residual correction provenance

### 6.1 Recoverable random-window experiment

Notebook 07 cell 7 L1–2 uses `train_test_split` on validation **window indices**, `test_size=0.30`, `random_state=42`, `shuffle=True`. Saved outputs report **731 calibration / 314 evaluation** windows. These are not the grouped ledger's 730/315 windows. Windows from a shared recording can appear on both sides even when individual windows do not overlap.

Cell 9 fits validation-calibration residuals with OLS on mean physical current. Saved outputs give static offset **−2.1516**, dynamic intercept **0.7846**, slope **2.3766 pp/A**, and calibration R² **0.3600** (printed precision). Cell 11 applies raw baseline, static subtraction, OLS residual subtraction, and then clipping; cell 13 reuses these parameters on the −20 °C data and writes a file explicitly marked post-hoc. No model retraining is shown in this notebook.

### 6.2 Grouped experiment has an identifiable claim, not a recoverable producer

`docs/ATHARV_GROUP1_PROGRESS_REPORT.md:L557` names **`scripts/run_grouped_correction.py`**, which is absent. A read-only search of the first-party Python files (root `analyze.py`, `scripts/`, `src/`, `tests/`) and source cells of all seven notebooks found no code writing the three `grouped_correction_*.csv` files or constructing their grouped split. The grouped plot exists but cannot substitute for producing code.

The report at L559–567 describes the membership above, OLS on 730 calibration windows, `c0 ≈ 1.0231`, `c1 ≈ 2.3575`, and R² 0.3516. The current inference script at L300–305 uses **`c0=1.023101`, `c1=2.357521`** in `prediction − (c0 + c1*mean_current)`; this is application code, not calibration evidence. Notebook 07's coefficients are different. Neither grouped CSV stores a complete fit/configuration record or checkpoint/data hashes.

**Recoverable:** reported group membership, window counts and intervals, metric variants and deltas, intended formula, and coefficients currently used by inference. **Unknown:** original reconstruction logic, authoritative index-to-source mapping, original execution code/environment, fit record, coefficient precision used for those CSVs, and a manifest linking those outputs to present artifacts. Do not create a replacement producer and describe it as the recovered historical experiment.

The reported slope difference is roughly 0.8% using the available rounded random-split slope and current grouped coefficient. Similar slopes do not prove physical causality, independence from all confounders, or unbiased generalization. Crucially, the grouped “evaluation files” belong to the same −10 °C validation pool previously used for neural-model early stopping and feature-group selection. They may be held out **from correction fitting**, but they are not globally unseen by model selection. The missing producer further limits verification of even that correction-fitting separation.

### 6.3 Post-hoc test interpretation

Current sources fit weights/scalers on training arrays and correction on validation-calibration arrays, not on test labels. This is evidence about the inspected code, not proof that no historical human choice was influenced by test results. Notebooks 04–07 and the report's own test-use chronology (progress report L590–596) show repeated test evaluation, residual inspection, and correction-transfer analysis. Describe −20 °C results as **historical post-hoc test analysis**. A future final-performance claim needs an agreed evaluation protocol and potentially new independent data, not stronger wording around the existing table.

## 7. Reporting conflicts and reconciliation

| Claim / location | Conflicting evidence / resolution | Remaining issue |
| --- | --- | --- |
| Frozen checkpoint validation RMSE 4.16: model spec L154; progress report L440/L773; speech guide L234 | Seed CSV L12 and clean robustness CSV L2 give **4.3005**. **4.1622 ± 0.1980** is the five-seed aggregate. | Historical reports need a separately approved correction; never label 4.16 as seed-42 accuracy. |
| 4.03 MLP comparison discussed as a 2,945-parameter model: progress report L20/L362/L684; meeting notes L39 | Comparison table L4 and Notebook 04 cells 3/12/14 refer to **13 features, 3,009 parameters**; saved `mlp_summary_50s.pt` dimensions confirm this. | Keep comparison and later Group C ablation separate; do not combine their metric and model size. |
| GRU has 60,869 parameters / about 20× larger: progress report L355/L361/L623, speech guide L179/L344 | Notebook 04 cell 12 defines one GRU layer with hidden size64 plus 64→16→1 head. The saved `gru_raw_50s.pt` has **14,305** tensor elements. Ratio to legacy MLP is about **4.75×**, or **4.86×** to Group C, not 20×. | A different GRU run might once have existed, but no evidence found ties 60,869 to this comparison. |
| Frozen test MAE 8.07: model spec L157, progress report L441 | Clean robustness CSV L5 and residual summary L3 give **7.880585**. | No producing table for 8.07 identified. |
| Overall/correction-subset values: model spec L153–156 gives train RMSE1.82, overall corrected2.60, calibration raw4.01/corrected2.50, evaluation raw MAE3.53 | Grouped CSV gives calibration raw **4.2132907**, corrected **2.9275434**; evaluation raw MAE **3.6964717**. Pooled clipped RMSE is **2.9054278**, not2.60. Train1.82 has no located saved aggregate producer. | Specification mixes unsupported or incompatible statistics. Treat those cells as unresolved, not authoritative evidence. |
| “28” validation sensor scenarios and broad noise impact <0.01: progress report L22/L473/L659; speech guide L243/L245 | Notebook 05 cell12 and saved robustness validation table contain **21 rows including clean baseline**. Voltage-noise 10mV row L13 changes RMSE by **0.0238851 pp**, above0.01. Moderate combined noise row L21 changes it by **0.00695336 pp**. | A limited moderate-noise statement is supported; 28 scenarios and universal <0.01 claims are not. |
| 50 s best/optimal versus 100 s: progress report L385–392; model spec L31 | Saved duration sweep has 4.3879 at50s versus4.3026 at100s; this is a different 13-feature single-run family with duration-specific window counts. Nominal49.9s is for ideal sampling, not every accepted recording. | Engineering latency trade-off is defensible; statistical optimality or confirmed exact10Hz timing is not. |
| Window counts / file splits | Saved report says 7,098 total50s windows, **1,045 validation overall**. Grouped eval315 differs from random eval314; grouped calib730 differs from random calib731. The split temperatures agree across current generation code and reports. | No contradiction in these counts once experiments are separated. Raw/source mapping and report-producer provenance remain unverified. |
| “avg3 seeds” Q-window comparison | Notebook 04 cell21 writes literal4.13/4.38 values. Later script has an actual five-seed ledger with different means. | Three-seed claim is provenance-incomplete; retain as historical table only. |
| 150 epochs, batch64, patience15 discussed as universal training facts | Current ablation script and Notebook 04 trainer agree on these settings. Seed reset, RNG consumption, model input dimension, sweep, and saved checkpoints differ by experiment. | No specific max-epoch/batch contradiction found in the inspected relevant sources; no run logs prove settings for every historical checkpoint. Do not infer identical runs from shared defaults. |
| Grouped evaluation on “completely unseen” files: progress report L561/L579; speech guide L292 | Held out from claimed correction fitting, but drawn from neural-model validation/selection files. Producer absent. | Restrict independence claim to the relevant fitting stage and mark it as reported, not independently proven. |
| Strict test independence / proof of intrinsic physics: progress report L573/L596/L716; speech guide L283/L346 | Repeated test residual and transfer analysis is explicit; similar slopes and correlations do not establish causality. | Human selection history cannot be reconstructed from current code. Further independent evaluation is needed. |
| EKF2.40 versus ML4.x: progress report L317 and speech guide L60 | Baseline CSV L5 and Notebook02 saved outputs agree historically, but predate correctness repairs and start from reference initial SoC. | Not current repaired-EKF accuracy or a fair same-task superiority comparison. OCV, timestamp quality and real-data EKF evaluation remain unresolved. |

Additional documentation maintenance: `docs/ENVIRONMENT.md` still has a T006A-era “Portability work still pending” section even though T006B completed. It was outside the authorized closeout edits and is unchanged here. Historical reports also contain machine-specific image links; avoid carrying them into public documentation. This audit does not edit any of these files.

## 8. Resume / public README evidence guidance

**Safe technical achievements now:** implementation of frozen initial-SoC inference using V/I/T windows and a 12-feature, 2,945-parameter MLP; reusable CLI/API and saved preprocessing artifacts; focused numerical-parity, input-contract, EKF-equation and portability test infrastructure; documented temperature-separated experiment design. Prior ticket test outcomes may be described with their scope; T007A did not rerun those suites. Scope remains initial estimation, with a nominal50s observation delay and prediction referring to window start.

**Source-supported historical quantities, if clearly labeled:** saved seed-42 validation4.3005/test10.1544 pp, five-seed Group C validation4.1622±0.1980 pp, and legacy13-feature MLP4.0331 versus single-run GRU4.5395 pp. Distinguish the model, split, seed statistic, post-hoc status and source table. Prominent accuracy claims still need a frozen artifact/data/run manifest and independently rerun evaluation under an approved ticket.

**Require further verification before prominent use:** grouped correction2.85/7.86 as a verified improvement; undocumented train1.82 or overall corrected2.60; three-seed Q-window averages; 20× model-size claims; 28 sensor scenarios; current EKF2.40 accuracy; hardware/embedded readiness, timing benchmarks, physical causality, cross-cell generalization or universal correction effectiveness. No latency benchmark procedure or independent hardware demonstration was established by this audit.

**README disclosures needed:** a single-cell Panasonic research dataset; nominal2.9Ah Ah-reference assumptions; restricted cold-test SoC coverage (approximately40–100%); within-trajectory correlation; temperature-specific train/validation/test roles; repeated post-hoc test inspection; correction evaluation files previously used in model selection; missing grouped producer; large datasets excluded from Git and fresh-install reproduction not demonstrated. Saved historical evidence and newly verified behavior must be labeled separately.

## 9. Follow-up decisions for an approved T007B

1. Locate the original `run_grouped_correction.py` and its fit/log/metadata records, or formally classify the grouped tables as legacy provenance-incomplete evidence. A newly implemented reconstruction would be a new experiment.
2. Approve an authoritative result register and report corrections, including seed42 versus means,13-feature comparisons, GRU count, MAE, correction aggregates and robustness counts.
3. Decide which data/version identifiers, source-file/window-index mapping, model/scaler hashes, environment and prediction records must be captured before any accuracy rerun. Preserve original artifacts and tables.
4. Agree evaluation independence and acceptance criteria before new experiments, including treatment of already-inspected −20 °C data and validation files used at multiple selection stages.
5. Retain open scientific questions: nominal capacity/ground-truth definition, low-SoC/cross-cell coverage, OCV validity and EKF baseline fairness, and unverified fresh installation.

## 10. Audit verification record

Commands were limited to `Get-Content`, `rg`/`rg --files`, read-only `git status`/`git ls-files`/`git log`/`git rev-parse`, and existing `.venv` Python with `-B` for JSON/CSV/text inspection, `statistics`/`math` arithmetic, SHA-256 fingerprints and safe tensor-metadata loading. Searches excluded the virtual environment and raw dataset payloads; no experiment script was imported or executed.

Checks completed: all four ablation-group RMSE means/population standard deviations against per-seed rows; grouped-file counts and interval accounting; correction-table delta arithmetic; grouped pooled-RMSE arithmetic; comparison checkpoint dimensions and tensor counts; source/table/output cross-references and missing grouped-producer search. These are audit checks, not fresh performance validation.

Only `docs/RESULTS_PROVENANCE_AUDIT.md` was created. Existing tracked files and experimental artifacts were not changed. Nothing was staged, committed or pushed. T007B has not begun.
