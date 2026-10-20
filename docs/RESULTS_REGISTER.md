# Authoritative Results Register and Reporting Policy

**Project:** Machine Learning-Based Initial State-of-Charge Estimation for Lithium-Ion Batteries

**Ticket:** T007B — proposed reporting dispositions for independent ChatGPT/user review.

**Evidence baseline:** `0d5e691cd7b00d347e76a4840c8729b51ea9c791`. Detailed investigation: [T007A provenance audit](RESULTS_PROVENANCE_AUDIT.md). This register is authoritative for identifying what the available historical tables report. It does not certify independently reproduced model accuracy or finalize disputed scientific interpretations. No predictions, training, notebook execution, or performance evaluation were performed for this register.

## 1. Canonical estimator and evidence vocabulary

The designated frozen estimator is the **Group C seed-42 SummaryMLP**: 12 inputs, architecture **12 → 64 → 32 → 1**, ReLU hidden layers, **2,945 parameters**. It accepts a nominal **50-second window of 500 V/I/T samples** and estimates reference SoC at the **start** of that window; the estimate becomes available after the observation period. Approximately 10 Hz sampling is assumed; duration/gap checks alone cannot establish uniform sampling.

Training temperatures are **25, 10 and 0 °C**; validation is **−10 °C**; historical test analysis is **−20 °C**. Saved 50-second counts are 5,182 training, 1,045 validation and 871 test windows. These counts come from [the saved dataset report, L15/L23](../results/initial_soc_dataset_validation.txt), whose original producer was not found; they were not revalidated against the processed dataset here. Data describe one Panasonic cell; cold-test targets cover approximately 40–100% SoC.

| Artifact / configuration | Canonical identity |
| --- | --- |
| Frozen weights | [results/models/ablation_group_c_seed_42.pt](../results/models/ablation_group_c_seed_42.pt) |
| Feature StandardScaler | [data/ablation_scaler_group_c.pkl](../data/ablation_scaler_group_c.pkl) |
| Channel min/range normalizer | [data/initial_soc_scaler.npz](../data/initial_soc_scaler.npz) |
| Ordered features | `V0, I0, T0, Vend, Iend, Tend, dV, dI, dT, meanV, meanI, stdV`, from normalized channels; final-minus-initial deltas and population voltage standard deviation |
| Source | [ablation script](../scripts/run_feature_ablation.py), L85–161 and L188–197; [inference API/CLI](../scripts/run_initial_soc_inference.py), L62–110 and L189–316 |

The reference label is `clip((2.9 + Ah) / 2.9 * 100, 0, 100)` at window start. The **nominal 2.9 Ah denominator** is a reference assumption, not independently established absolute SoC. Generation and split source: [Notebook 03](../notebooks/03_Feature_Engineering.ipynb), cells 3/5/8/12. Notebook cell numbers throughout this register are zero-based.

A saved checkpoint and its present hash do not establish original training provenance. T007A records present hashes and architecture properties, but no complete historical manifest binds dataset, code, environment, checkpoint/scalers, predictions and metrics. Optional correction and clipping are separate processing choices and must be identified explicitly; the ablation validation baseline is raw, uncorrected and unclipped.

Provenance labels retain the audit's meanings:

- **Verified artifact property / saved-row arithmetic:** a checked count, shape, file identity or aggregate calculation; never proof of accuracy.
- **Source-traceable only:** producing logic is identifiable, but historical execution-to-artifact identity and performance have not been independently reproduced.
- **Provenance-incomplete:** a saved claim exists but its producer or necessary run evidence is missing.
- **Inference / unknown:** available evidence cannot establish the proposed interpretation.

All result CSVs, cited notebooks/scripts, selected model/scaler artifacts, and the T007A audit are Git-tracked at this baseline. Historical model specification, progress report, meeting notes and speech guide are **local untracked documents**; references to their disputed claims below are mediated through audit section 7. Their local presence does not mean they are available on GitHub.

## 2. Historical accuracy register

**Units:** RMSE/MAE are SoC **percentage points (pp)**, not relative percentages. Exact means the decimal saved in the named table. Rounding does not increase evidence strength. Every row below remains historical.

| Model / experiment | Exact saved RMSE → reporting value (pp) | Split / supported N | Seed or correction protocol | Primary evidence, physical lines | Provenance | Proposed publication guidance |
| --- | --- | --- | --- | --- | --- | --- |
| Frozen Group C validation | `4.300520692393889` → **4.3005** | −10 °C; 1,045 | Single designated seed 42 | [feature_ablation_seeds.csv, L12](../results/feature_ablation_seeds.csv); [clean robustness, L2](../results/robustness_clean_vs_perturbed.csv) | Source-traceable only | May describe the saved seed-42 validation score with historical/non-reproduced qualification. |
| Group C five-seed validation | `4.16216383761665 ± 0.19798709679269677` → **4.1622 ± 0.1980** | −10 °C; same 1,045 windows per run | Seeds 42/100/200/300/400; mean ± population SD | [feature_ablation_summary.csv, L4](../results/feature_ablation_summary.csv); per-seed CSV L12–16 | Source-traceable runs; verified saved-row aggregation | State mean ± population SD of five runs; not the frozen score, an ensemble, or a confidence interval. |
| Frozen Group C historical test | `10.15444244732401` → **10.1544** | −20 °C; 871 | Single seed 42; prior test inspection | [clean robustness, L5](../results/robustness_clean_vs_perturbed.csv) | Source-traceable only; post-hoc history | Historical post-hoc test analysis; no claim of independent final validation. |
| Legacy 13-feature summary MLP validation | `4.033112561407553` → **4.0331** | −10 °C; 1,045 per shared 50-second dataset report/source | Single historical run; notebook initializes RNG once to 42, not independently per model | [model comparison, L4](../results/initial_soc_model_comparison_50s.csv); [Notebook 04](../notebooks/04_ML_Baselines.ipynb), cells 3/5/12/14 | Source-traceable only; incomplete run linkage | Identify 13 features including integrated current and 3,009 parameters. Never attribute this score to Group C. |
| Legacy raw-sequence GRU validation | `4.539547603770602` → **4.5395** | −10 °C; same historical 1,045-window context | Single notebook run; RNG shared with earlier training operations | [model comparison, L6](../results/initial_soc_model_comparison_50s.csv); Notebook 04 cells 12/14 | Source-traceable only; incomplete run linkage | A historical within-experiment comparison; not a multi-seed superiority test against later Group C. |
| Grouped current correction + clipping, evaluation | `2.853517` → **2.8535** | Reported −10 °C; 315 windows from three files | Reported fit on six other validation files / 730 windows; variant D | [grouped validation, L9](../results/grouped_correction_validation.csv); [group membership, L2–10](../results/grouped_correction_files.csv) | Provenance-incomplete | Retain in a limitations/provenance table as a reported value; withhold verified-improvement claims. |
| Grouped current correction + clipping, test | `7.863251` → **7.8633** | Reported −20 °C; 871 | Same reported correction; variant D, post-hoc transfer | [grouped post-hoc test, L5](../results/grouped_correction_test_posthoc.csv) | Provenance-incomplete; post-hoc | Same restriction, explicitly historical post-hoc; not independent generalization evidence. |
| Random-window correction + clipping, evaluation | `2.7414563` → **2.7415** | −10 °C; 314 evaluation / 731 calibration windows | Shuffled window split, `random_state=42`; OLS residual correction; variant D | [residual correction validation, L9](../results/residual_correction_validation.csv); [Notebook 07](../notebooks/07_Residual_Correction.ipynb), cells 7/9/11 | Source-traceable only; saved notebook outputs | Historical random-window analysis; shared-recording dependence and model-selection reuse limit independence. Not the grouped result. |
| Random-window correction + clipping, test | `7.698302` → **7.6983** | −20 °C; 871 | Parameters from preceding random split; variant D | [residual correction post-hoc test, L5](../results/residual_correction_test_posthoc.csv); Notebook 07 cell 13 | Source-traceable only; post-hoc | Qualified historical analysis, separate from grouped correction and independent final testing. |

The grouped ledger's 730/315 allocation is approximately 70/30 **by windows**, with six/three files; the original file-to-window mapping is not independently established. Its evaluation files were already part of neural-model validation for early stopping/feature selection. Reported separation from correction fitting is not proof of globally unseen end-to-end evaluation. Notebook 07's random-window correction has different coefficients and membership from the grouped tables.

### Duration and sensor findings

The duration sweep uses **separate 13-feature MLP runs**, not duration variants of the frozen Group C checkpoint. [Notebook 04](../notebooks/04_ML_Baselines.ipynb), cell 17 identifies producing logic. Primary metrics: [duration comparison, L2–7](../results/initial_soc_duration_comparison.csv); N values: saved dataset report L11–16. Classification: **source-traceable only**.

| Nominal duration | Exact saved validation RMSE → rounded (pp) | Saved −10 °C N |
| --- | --- | --- |
| 5 s | `5.645603339471379` → 5.6456 | 1,102 |
| 10 s | `6.6578350060491465` → 6.6578 | 1,097 |
| 20 s | `5.029471705092317` → 5.0295 | 1,079 |
| 30 s | `4.619674296932763` → 4.6197 | 1,067 |
| 50 s | `4.387911898414447` → 4.3879 | 1,045 |
| 100 s | `4.302565257471385` → 4.3026 | 984 |

Proposed wording: “The historical duration sweep explored an observation-delay/error trade-off.” Different accepted windows, single runs and absent uncertainty analysis do not establish statistical optimality of 50 seconds. Do not substitute the non-sweep 4.0331 score for the sweep's 4.3879 score.

The frozen Group C sensor study is **source-traceable only**, using synthetic perturbations; no real sensor or hardware verification is implied. [Notebook 05](../notebooks/05_Robustness_Analysis.ipynb), cells 7/10/12/15, identifies its logic.

| Saved case | Exact RMSE → rounded (pp) | Context / evidence | Reporting limit |
| --- | --- | --- | --- |
| Validation clean → combined 5 mV / 25 mA noise | `4.300520692393889` → 4.3005; `4.30747405484401` → 4.3075 | −10 °C, 1,045 windows; [robustness summary, L2/L21](../results/robustness_validation_summary.csv) | Saved delta ≈0.00695336 pp applies to this case only. |
| Validation voltage noise 10 mV | `4.324405766055284` → 4.3244 | Same validation set; robustness summary L13 | Delta ≈0.02388507 pp contradicts a blanket “noise impact <0.01” claim. |
| Test combined 5 mV / 25 mA noise | `10.16036687894419` → 10.1604 | −20 °C, 871 windows; [clean/perturbed comparison, L6](../results/robustness_clean_vs_perturbed.csv) | Historical post-hoc synthetic case. |

The validation summary has **21 rows including the clean baseline (20 perturbations)**, plus three test conditions in the separate comparison table. These do not establish “28 sensor scenarios.” No repeated-noise uncertainty study or universal robustness conclusion is established.

## 3. Model-size reporting

Counts below are properties checked in T007A against saved tensor dimensions; they are not new training evidence. See audit sections 2.3/7 and Notebook 04 cell 12.

| Model | Supported count | Supporting artifact |
| --- | ---: | --- |
| Frozen 12-feature Group C MLP | 2,945 | Canonical checkpoint above |
| Historical 13-feature summary MLP | 3,009 | [mlp_summary_50s.pt](../results/models/mlp_summary_50s.pt) |
| Present saved GRU, input 3 / hidden 64 / one layer / 64→16→1 head | 14,305 | [gru_raw_50s.pt](../results/models/gru_raw_50s.pt) |

Parameter-count ratios are `14305 / 3009 ≈ 4.75` and `14305 / 2945 ≈ 4.86`. They measure parameter elements, not deployment storage, latency or energy. The historical **60,869 / approximately 20×** claim is unsupported by the present checkpoint. No alternative GRU configuration is inferred to explain it.

## 4. Proposed dispositions of reporting conflicts

“Resolved from available evidence” resolves the identity or arithmetic issue only; it does not certify model accuracy. Original reports, specifications and CSVs remain untouched. Claim locations and detailed references are preserved in audit section 7, including local untracked report citations.

| Conflict | Disposition | Safe wording / action proposed |
| --- | --- | --- |
| Seed-42 4.16 versus five-seed mean | Resolved from available evidence | “Saved seed-42 validation RMSE: 4.3005 pp; five-run mean ± population SD: 4.1622 ± 0.1980 pp.” |
| Legacy 4.0331 / 13 features versus Group C / 12 | Resolved from available evidence | Identify the earlier 3,009-parameter MLP separately; do not combine its score with Group C's size. |
| GRU 60,869 parameters / 20× reduction | Resolved from available evidence | Present checkpoint: 14,305; parameter ratios above. Withhold 60,869/20× absent evidence for another relevant run. |
| Frozen test MAE 8.07 | Resolved from available evidence | [Clean comparison L5](../results/robustness_clean_vs_perturbed.csv) saves **7.880585193634033 pp**, report **7.8806 pp**. No producer for 8.07 was identified. |
| Overall corrected 2.60 versus 2.9054 | Provenance-incomplete | Saved subsets imply `sqrt((730×2.9275434² + 315×2.853517²)/1045) = 2.9054278081 pp`. Label this derived saved-table arithmetic; withhold 2.60. Membership/producer still unverified. |
| Missing grouped producer | Provenance-incomplete | “Grouped tables are legacy evidence with unavailable producing code in the current repository.” Inference applies coefficients; it does not document their original fit. |
| 21 saved validation rows versus 28 scenarios | Resolved from available evidence | “21 validation rows including baseline, plus three post-hoc test cases.” Withhold the 28-case claim. |
| Three-seed Q-window ablation | Provenance-incomplete | [Q-window CSV L4–5](../results/initial_soc_q_window_ablation_50s.csv) saves 4.13/4.38; Notebook 04 cell 21 writes literals. No identifiable three-run computation supports the labels; do not reuse as verified averages. |
| 50 seconds statistically optimal | Unresolved scientific question | “Nominal 50 seconds is the selected observation duration.” Require controlled repeated comparisons before claiming optimality. |
| Historical EKF 2.4021 versus repaired EKF | Historical result requiring qualification | [baseline CSV L5](../results/baseline_metrics.csv) predates sign/time repairs. The 25 deterministic tests verify scoped equations/loop behavior, not historical trajectory accuracy. EKF starts at reference SoC; avoid same-task superiority claims. |
| Strong test independence | Historical result requiring qualification | “−20 °C results were inspected post hoc.” Grouped validation files were reused in neural-model selection; do not call the entire pipeline independently validated. A new agreed protocol is required. |
| Similar correction slopes prove physical causality | Unresolved scientific question | “Residual-current association was observed in historical analyses.” Confounding and human selection history are unresolved; causal or universal-transfer claims require new evidence. |

## 5. Proposed public-claims policy

### Safe to state now

Suggested README wording: “A research prototype for initial battery SoC estimation from a nominal 50-second voltage/current/temperature window. Frozen inference uses 12 summary features and a 2,945-parameter MLP, with saved preprocessing artifacts and a reusable Python API/CLI. The estimate refers to window-start SoC.”

Suggested academic wording: “Prior scoped verification recorded 17 input-contract tests and seven numerical-parity tests; the parity checks included five saved validation windows. Twenty-five deterministic EKF tests and seven path-portability tests were also recorded. These checks establish behavior within their stated scopes.” Sources: [current state](CURRENT_STATE.md), testing section, and tracked tests. T007B did not rerun these suites. Fresh-install reproduction, hardware readiness and aggregate accuracy are not established by these checks.

### Historical findings that require qualification

Suggested README wording: “Historical saved tables report seed-42 validation RMSE of 4.3005 pp at −10 °C and a five-seed mean of 4.1622 ± 0.1980 pp. These accuracy results have not been independently reproduced. Historical −20 °C analysis reports 10.1544 pp and was inspected post hoc.” Link this register beside any metrics.

Suggested academic wording: “Earlier 13-feature MLP and GRU runs, duration sweeps and synthetic perturbation studies are separate historical experiments with incomplete run manifests. Correction analyses are exploratory: random-window results are source-traceable, while grouped-file tables lack their original producer. Reported grouped improvements are not verified accuracy improvements.” Any inclusion of correction numbers must retain the protocol, sample counts, evidence class and limitations from section 2.

Disclose the single-cell dataset, nominal 2.9 Ah target assumption, correlated windows, temperature-specific splits, cold-test SoC coverage, selection reuse and prior test inspection. Large raw/processed data are excluded from Git; current artifacts alone do not guarantee reproduction.

### Do not claim yet

Withhold verified grouped-correction gains, 20× compression, independently validated final test generalization, hardware deployment, measured deployment latency, proven physical causality, universal sensor robustness and cross-cell/chemistry generalization. Do not promote unexplained 2.60 overall correction or three-seed Q-window claims. Existing evidence supports a scoped research prototype and qualified historical findings.

## 6. Proposed minimum evidence for future accuracy claims

This is a **future standard for approval**, not a statement that the evidence already exists. Before improvement or evaluation, agree the target/reference, acceptable error/range, data partitions, selection process and test-access rules.

| Evidence to capture | Minimum practical record |
| --- | --- |
| Dataset identity | Version, source/license, source-file hashes, extraction configuration and stable source-file/window-start identifiers; version regenerated data separately. |
| Code and environment | Exact Git commit, command/config, Python/package versions and platform/device; document deterministic settings and fresh-install status. |
| Features and preprocessing | Ordered features, units/current sign, time/sample contract, train-only fit scope, output scaling, and correction/clipping switches and coefficients. |
| Frozen artifacts | Checkpoint and every scaler/normalizer SHA-256; link fitting/selection logs to these exact bytes. Present hashes alone do not reconstruct history. |
| Evaluation partition/history | Explicit files/trajectories/windows and temperatures, reference coverage, calibration/selection roles, and all prior access to validation/test labels/results. |
| Predictions and labels | Per-window prediction, reference label, source identifier/start timestamp, split, model/run identity and relevant metadata, subject to storage/privacy permissions. |
| Metric computation | Versioned calculation code, units, aggregation/weighting, clipping policy, per-split counts and subgroup errors; preserve predictions so metrics can be checked. |
| Randomness and selection | Seeds, model initialization/order, early-stopping criterion, hyperparameter/feature/correction selection procedure and all relevant attempted runs. |
| Previously inspected −20 °C data | Explicitly retain exploratory/post-hoc status. Decide prospectively whether suitable new independent trajectories/cells/data are required; do not relabel inspected data as untouched. |

Freeze the validation/test protocol before model improvement. If the grouped producer is recovered, inspect its identity and linkage before trusting it; if reconstructed instead, label and preserve it as a new experiment. Agree independence and uncertainty methods appropriate to trajectory correlation; statistical or causal claims need evidence beyond metric rounding.

## 7. Decisions for ChatGPT/user review

| Decision | Recommendation pending review | Required next authorization / evidence |
| --- | --- | --- |
| Upcoming README content | Include canonical implementation, scoped prior test achievements, limitations and this register link. If metrics are included, use qualified historical seed-42/five-seed values; separately label post-hoc test history. | Approve public wording and a later README/publication ticket. No README is created here. |
| Historical findings to retain | Keep legacy 13-feature/GRU, duration and sensor findings with their protocol/evidence limits. Put correction figures in qualified provenance discussion, not verified-improvement headlines. | Review section 2 row-by-row; retain original artifacts. |
| Report corrections | Recommend correcting seed statistics, MLP/GRU sizes and ratios, test MAE, scenario counts and optimum/independence/causality wording; flag unsupported correction aggregates and Q-window averages. | Approve exact historical report files, edit policy and traceability before changes. No report correction is performed here. |
| Questions needing evidence | Original grouped fit/window mapping, historical run manifests, independent final evaluation, reference-capacity validity, low-SoC/cross-cell coverage, causal explanation, and real-data EKF/OCV validity. | Locate original records or approve specific new verification/experiments; wording alone cannot establish these findings. |
| Grouped producer availability | `scripts/run_grouped_correction.py` remains absent from the currently available repository; T007A found no grouped CSV producer in inspected first-party code. It may exist elsewhere. | Ask for original script/logs/fit metadata or explicitly retain provenance-incomplete status. Do not describe a replacement as recovered history. |

## 8. T007B verification record

Read-only inspection used `Get-Content`, `rg`, `git status`/`git ls-files`, and existing `.venv` Python with `-B` for notebook JSON/source, small CSV arithmetic and Markdown/reference checks. Table values and split/sample labels were checked against the audit and small saved reports. Arithmetic checks cover five-seed mean/population SD, pooled grouped RMSE, parameter ratios and selected sensor deltas; none uses predictions or large datasets. Existing model-test suites were not rerun.

Only this register was created. Existing tracked files, local historical documents and scientific artifacts were preserved; no Git staging, commit or push was performed. Scientific dispositions above remain proposals for independent review.
