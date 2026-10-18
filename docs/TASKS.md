# Project Ticket Register

Future tickets remain **Proposed** until ChatGPT and the user approve them. Approval of a ticket authorizes only its stated goal and allowed-file boundaries. Work on one approved ticket at a time. This register does not authorize Git operations or project implementation by itself.

## Completed and active tickets

### T000 — Repository Audit

- **Goal:** Build an evidence-based understanding of the existing project.
- **Priority:** Complete before development.
- **Dependencies:** None.
- **Status:** Completed.
- **Acceptance criteria:** Repository, documentation, source, saved experiments, risks, environment, Git state, and follow-up questions audited without project modification.

### T001 — Documentation Foundation

- **Goal:** Establish persistent Codex instructions, factual current state, and an initial ticket register.
- **Priority:** Foundation.
- **Dependencies:** T000.
- **Status:** Completed.
- **Acceptance criteria:** Only `AGENTS.md`, `docs/CURRENT_STATE.md`, and `docs/TASKS.md` created or updated; initial-SoC-only scope recorded; evidence distinguished from fresh verification; T002–T007 and additional follow-ups listed.

### T002 — Git Initialization, .gitignore, and GitHub Setup

- **Goal:** Establish version control for the approved repository and prepare safe GitHub synchronization.
- **Priority:** High; enables reviewable checkpoints.
- **Dependencies:** T001; repository location, remote, and artifact boundaries approved.
- **Status:** Completed.
- **Acceptance criteria:** Local `main` and reviewed `.gitignore` established; 101 reviewed files committed in the initial checkpoint and pushed to the designated private repository; excluded datasets and uncertain third-party materials remain local.

### T003 — Repair Broken Inference API/CLI

- **Goal:** Restore the frozen Group C seed-42 inference module's syntax, import, and CLI help behavior.
- **Priority:** Critical; inference was blocked.
- **Dependencies:** T002; explicit approval and allowed-file boundary.
- **Status:** Completed.
- **Acceptance criteria:** Module syntax/import and CLI `--help` pass; frozen artifacts load; one saved 500-sample validation-window API smoke test completes; correction/clipping switches follow their existing formulas; no unrelated project files change.

### T004A — Training/Inference Numerical Consistency Verification

- **Goal:** Compare frozen inference preprocessing and forward outputs with an independent implementation of the saved training contract.
- **Priority:** High; prerequisite evidence for relying on inference output.
- **Dependencies:** T003; approved numerical-parity scope.
- **Status:** Completed.
- **Acceptance criteria:** Seven focused tests pass; five saved validation windows show channel, feature, standardized-feature, and forward-output parity within documented absolute tolerances; deterministic synthetic fallback remains clearly identified as non-accuracy evidence; no training or artifact modification occurs.

### T004B — Inference Input-Contract Validation

- **Goal:** Enforce the frozen estimator's documented measurement and optional timestamp contract, and reject ambiguous CLI CSV inputs.
- **Priority:** High; completes T004's inference verification.
- **Dependencies:** T003, T004A; timestamp limits from `docs/INITIAL_SOC_MODEL_SPEC.md` section 2; approved CSV layouts.
- **Status:** Completed.
- **Acceptance criteria:** Exactly 500 finite numeric V/I/T samples are required; optional timestamps are checked for shape, order, duration, and maximum gap; only approved CSV layouts are accepted; 17 contract and seven parity tests pass; no model accuracy claim or scientific artifact change is made.

### T005A — EKF State-Prediction Sign Repair and Focused Tests

- **Goal:** Correct the EKF SoC prediction sign for negative-discharge current and test the actual notebook helper in isolation.
- **Priority:** High; necessary before relying on EKF baseline claims.
- **Dependencies:** T000 current-sign finding; confirm polarization and terminal-voltage conventions.
- **Status:** Completed.
- **Acceptance criteria:** The corrected helper is used by the notebook loop; 11 focused sign, bounds, capacity-conversion, polarization, voltage-direction, and Coulomb-counting checks pass; initial-reference-SoC assumption and historical result status are stated; no full trajectory or saved result is regenerated.

### T005B — EKF Loop Timing and Measurement-Update Verification

- **Goal:** Resolve the EKF sample/time convention and verify the complete loop's deterministic prediction and measurement-update behavior.
- **Priority:** High; closes deterministic correctness verification for the baseline.
- **Dependencies:** T005A; establish whether current and timestep indices represent interval start and sample time.
- **Status:** Completed within deterministic synthetic-test scope.
- **Acceptance criteria:** Sample 0 receives measurement update without propagation; later prediction uses interval-start current and actual interval duration; measurement update uses current-sample current; innovation, Jacobian, gain, Joseph covariance and state flow pass focused deterministic tests; notebook outputs and historical metrics remain unchanged.

### T006A — Inference Environment and Dependency Reproducibility

- **Goal:** Identify and document minimum dependencies for frozen initial-SoC inference using the existing verified environment.
- **Priority:** High; establishes a reproducible starting point for inference.
- **Dependencies:** T003 inference API/CLI; existing local environment and frozen artifacts.
- **Status:** Completed for existing-environment verification and documentation.
- **Acceptance criteria:** Direct inference dependencies and existing versions are recorded; setup, CLI, artifact, and test instructions are documented; contract and parity tests, CLI help, and dependency consistency check pass in the existing environment; fresh-install reproducibility is not claimed.

### T006B — Notebook and Experiment-Script Portability

- **Goal:** Resolve repository roots portably in notebooks 01–07 and the feature-ablation script, and replace Notebook 04's incompatible trapezoidal integration call.
- **Priority:** High; removes the identified machine-specific path assumptions and NumPy API incompatibility.
- **Dependencies:** T006A; approved source-cell and script boundaries.
- **Status:** Completed.
- **Acceptance criteria:** All seven notebooks resolve from the repository root and `notebooks/` using repository markers and reject unrelated directories; the script uses `__file__`-relative resolution; Notebook 04 uses `np.trapezoid` with existing arguments; seven portability, 11 EKF baseline, and 14 EKF-loop tests pass; notebook outputs and metadata are preserved; no experiment or training is run.

## Proposed ticket sequence

### T004 — Preprocessing and Inference Verification

- **Goal:** Verify training/inference feature ordering, scaling, units, and input-contract enforcement.
- **Priority:** High; required before relying on inference output.
- **Dependencies:** T003; agreement on window length and timing contract.
- **Status:** Completed (T004A and T004B complete).
- **Acceptance criteria:** Focused checks compare training and inference preprocessing; valid 500-sample input is handled consistently; malformed, non-finite, or invalid-timing input receives documented handling; no claim of broad accuracy verification from unit checks alone.

### T005 — EKF Correctness Repair and Verification

- **Goal:** Correct the current-sign defect and verify discharge/charge behavior in the EKF process model.
- **Priority:** High; required before quoting EKF as a sound benchmark.
- **Dependencies:** Approval; confirmation of intended current and polarization sign conventions.
- **Status:** Completed within deterministic correctness-verification scope (T005A and T005B complete).
- **Acceptance criteria:** Focused prediction and loop tests show the documented sign and sample-time behavior; 11 prediction tests and 14 loop tests pass; historical metrics are not attributed to repaired code. Real-data evaluation and scientific baseline suitability remain open.

### T005 follow-up issues (not a new ticket)

- Real-data EKF trajectory evaluation and accuracy remain unverified; historical EKF metrics predate the repairs.
- OCV validity across the dataset's temperature and SoC range needs assessment.
- Timestamp quality needs assessment; the notebook currently replaces nonpositive intervals with the median positive interval.
- Baseline-comparison fairness remains open because the EKF starts from reference initial SoC while the ML model estimates unknown initial SoC from a window.

### T006 — Dependency and Path Portability

- **Goal:** Replace machine-specific project roots in approved development paths and document a reproducible environment.
- **Priority:** High.
- **Dependencies:** T002; choose the supported execution environment.
- **Status:** Completed within the approved dependency and path portability scope (T006A and T006B complete). Fresh-environment installation and verification remain outstanding.
- **Acceptance criteria:** Frozen inference requirements and setup are documented; approved notebooks and the feature-ablation entry point resolve paths portably; the identified NumPy compatibility call is repaired and focused checks pass; no complete project lockfile or fresh-install verification is claimed.

### T007 — Experiment Provenance and Results Reconciliation

- **Goal:** Trace result tables to producing code/configuration/checkpoints and reconcile conflicting reports.
- **Priority:** High; necessary for defensible academic claims.
- **Dependencies:** T002; T003–T006 as needed to identify current behavior.
- **Status:** Proposed (next ticket).
- **Acceptance criteria:** Frozen seed-42 metrics remain distinct from five-seed means; grouped-correction producer or provenance gap is recorded; conflicting tables and report claims are resolved or clearly marked historical; saved results are not silently overwritten.

## Additional proposed work identified by T000

### T008 — Focused Automated Inference and Data-Contract Checks

- **Goal:** Add small automated checks for loader behavior, inference input validation, checkpoint/scaler compatibility, and key edge cases.
- **Priority:** High.
- **Dependencies:** T003, T004, and T006; approve test files and test-running scope.
- **Status:** Proposed.
- **Acceptance criteria:** Tests check real behavior rather than reimplementing the same formulas; tests run without training, simulation, or dataset-wide processing; results are reported accurately.

### T009 — Window-Level Data Provenance

- **Goal:** Preserve source trajectory and window-start metadata so experiments can be grouped and audited reliably.
- **Priority:** Medium-high.
- **Dependencies:** T002; approved dataset/artifact boundary; T007.
- **Status:** Proposed.
- **Acceptance criteria:** A documented, repeatable mapping links each processed window to its source file and start location; existing saved datasets/results remain preserved; any regenerated artifact is versioned as a new, traceable result.

### T010 — Initial-Estimate to Current-Time Handoff Analysis

- **Goal:** Determine how a SoC estimate for window start can initialize a tracker at the end of the observation period within the approved initial-estimation scope.
- **Priority:** Medium-high; settle semantics before making operational claims.
- **Dependencies:** T000 findings; confirm target use and accepted delayed-estimate behavior.
- **Status:** Proposed.
- **Acceptance criteria:** Documented timing/index convention and a focused offline analysis explain how the estimate corresponds to the intended prediction instant; does not expand deliverables into continuous tracking or controller integration.

### T011 — Evaluation Protocol and Generalization Limits

- **Goal:** Establish an agreed evaluation protocol that addresses prior test-set inspection, trajectory correlation, cold SoC coverage, and any required external-cell validation.
- **Priority:** Medium-high; depends on academic requirements.
- **Dependencies:** T000; T007; user/guide decision on external validation and acceptance criteria.
- **Status:** Proposed.
- **Acceptance criteria:** Splits and metrics are declared before new evaluation; trajectory-level dependence and the 40–100% cold-test range are documented; external-data validation is performed only if approved and data/labels are suitable.

All tickets remain within initial SoC estimation. Continuous tracking, charger-control implementation, and hardware integration require a separately approved scope change.
