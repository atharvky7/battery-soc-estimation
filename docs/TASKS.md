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

## Proposed ticket sequence

### T002 — Git Initialization, .gitignore, and GitHub Setup

- **Goal:** Establish version control for the approved repository and prepare safe GitHub synchronization.
- **Priority:** High; enables reviewable checkpoints.
- **Dependencies:** Explicit user approval; confirm repository location, intended remote, and treatment of large data/artifacts first.
- **Status:** Proposed.
- **Acceptance criteria:** Agreed Git repository and branch exist; an appropriate `.gitignore` is reviewed; no private credentials are stored; remote points to the user-designated private repository; initial status is reviewed. Commit/push only if explicitly included in the approved ticket.

### T003 — Repair Broken Inference API/CLI

- **Goal:** Make the existing Group C seed-42 inference module syntactically valid and usable through its documented API/CLI.
- **Priority:** Critical; current inference is blocked.
- **Dependencies:** T002 recommended; approval; explicit allowed-file list.
- **Status:** Proposed.
- **Acceptance criteria:** Module parses/imports; CLI help and a documented local sample invocation operate; output fields and clipping/correction switches match documentation; no unrelated feature or report changes.

### T004 — Preprocessing and Inference Verification

- **Goal:** Verify training/inference feature ordering, scaling, units, and input-contract enforcement.
- **Priority:** High; required before relying on inference output.
- **Dependencies:** T003; agreement on window length and timing contract.
- **Status:** Proposed.
- **Acceptance criteria:** Focused checks compare training and inference preprocessing; valid 500-sample input is handled consistently; malformed, non-finite, or invalid-timing input receives documented handling; no claim of broad accuracy verification from unit checks alone.

### T005 — EKF Correctness Repair and Verification

- **Goal:** Correct the current-sign defect and verify discharge/charge behavior in the EKF process model.
- **Priority:** High; required before quoting EKF as a sound benchmark.
- **Dependencies:** Approval; confirmation of intended current and polarization sign conventions.
- **Status:** Proposed.
- **Acceptance criteria:** Focused sign/edge checks show discharge lowers predicted SoC and charging raises it under declared conventions; voltage-model conventions remain internally consistent; any renewed benchmark is tied to an explicit run and code state.

### T006 — Dependency and Path Portability

- **Goal:** Replace machine-specific project roots in approved development paths and document a reproducible environment.
- **Priority:** High.
- **Dependencies:** T002; choose the supported execution environment.
- **Status:** Proposed.
- **Acceptance criteria:** Selected entry points resolve files relative to the repository; required dependencies and versions are declared; existing environment incompatibilities are addressed without unrelated dependency additions; no environment installation unless separately authorized.

### T007 — Experiment Provenance and Results Reconciliation

- **Goal:** Trace result tables to producing code/configuration/checkpoints and reconcile conflicting reports.
- **Priority:** High; necessary for defensible academic claims.
- **Dependencies:** T002; T003–T006 as needed to identify current behavior.
- **Status:** Proposed.
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
