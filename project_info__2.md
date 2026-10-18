# Major Project — SoC Estimation — Progress Review

## Summary

This is a **Machine Learning State-of-Charge (SoC) estimator** for Li-ion batteries. It takes voltage (V), current (A), and temperature (T), and predicts the battery's charge percentage. It's your **Group 1** deliverable whose output feeds **Group 2's** Zeta-Luo multilevel charger control. Dataset: **McMaster Panasonic 18650PF** (2.9 Ah), with **LG 18650HG2** (3.0 Ah) as a generalization target.

**Verdict on your question — "are there mistakes, can I proceed?"**

There's **one critical bug** you **must fix before building the ML model**: the **EKF SoC prediction has a sign error**, which makes the battery *charge during discharge*. Your current baseline lists EKF at **RMSE 6.08%** vs **Perfect CC at 0.08%** — a working EKF should *beat* Coulomb counting, not be 75× worse. That 6.08% is **not a real result**; it's a symptom of the bug. If you build the LSTM and claim you beat "EKF at 6.08%", the comparison is meaningless.

**Bottom line:** Excellent foundation, strong methodology. Fix the EKF sign, re-run baselines to get a real ML target, then proceed to the LSTM. Everything else looks correct and well-reasoned.

---

## What Is Done & Correct (on track — no redo needed)

- **Ground-truth SoC formula** — `SoC = (Q_max + Ah)/Q_max * 100` with `Q_max = 2.9` is **correct**. The README confirms the Ah counter resets before each drive cycle and the battery is charged to 4.2V/50mA first, so each file starts at 100% with Ah=0. Your **Perfect CC RMSE of 0.08%** confirms the formula reproduces ground truth to float precision — solid validation.
- **Current sign** — negative = discharge (McMaster). CC correctly drops SoC on discharge.
- **Coulomb counting** — formula and sign are right.
- **Temperature-based split** — Train = 25/10/0°C, Val = −10°C, Test = −20°C. **Excellent methodology** — tests true cold-temperature generalization, not random-row leakage. Don't change this.
- **Normalization** — min-max from train only, applied to val/test. Correct; no leakage.
- **Sliding windows respecting file boundaries** — correct and important (prevents crossing Ah-reset discontinuities).
- **Window size 500 (50 s at 10 Hz)** — sensible; captures ~2.6× the RC time constant τ≈18.87 s.
- **EDA** — thorough and solid: multi-temp voltage-vs-SoC, usable-capacity-vs-temp, OCV-SoC from C/20, drive-cycle characterization, coulometer validation (MAE < 0.0009 Ah).

---

## 🔴 CRITICAL BUG — EKF SoC Prediction Sign Error (MUST FIX)

**File:** `notebooks/02_Baselines.ipynb` — EKF main loop

```python
soc_pred = float(np.clip(x[0] - (ETA * I * dt) / (Q_MAX * 3600) * 100.0, 0.0, 100.0))
```

**The problem:** With `I` negative during discharge, `(ETA * I * dt) / (Q_MAX * 3600) * 100.0` is **negative**. So `x[0] - (negative)` = `x[0] + positive` → **SoC increases during discharge**. Backwards — the EKF "charges" the battery while it's discharging.

**The fix — change the minus to a plus:**

```python
soc_pred = float(np.clip(x[0] + (ETA * I * dt) / (Q_MAX * 3600) * 100.0, 0.0, 100.0))
```

This matches your CC code, which correctly uses `soc[k-1] + delta_soc` where `delta_soc = (I * dt) / (Q_max * 3600) * 100`.

**Evidence it's a bug:** `results/baseline_metrics.csv` shows EKF RMSE **6.0798%** vs Perfect CC **0.0838%**. A correctly-tuned EKF with good HPPC parameters should be *better* than noisy Coulomb counting (it uses voltage feedback to correct drift). 6.08% vs 0.08% is ~75× worse — that's the SoC state being driven the wrong way, not "the flat OCV curve is hard."

**Why it matters:** Notebook 03's summary says *"Target: RMSE < 6.08% (beat EKF baseline)"*. If EKF is broken, beating 6.08% is trivial and your comparison table is invalid. After the fix, true EKF RMSE should drop to **~1–3%**. Re-run the baseline notebook, then set your real ML target (e.g., "beat the fixed EKF RMSE").

---

## 🟡 Secondary Issues / Concerns

### 1. Test set only covers 40–100% SoC (important limitation)
Per the README: drive cycles at **0°C, −10°C, −20°C** are terminated at **2.32 Ah (80% DOD), 2.03 Ah (70% DOD), 1.74 Ah (60% DOD)**. With nominal 2.9 Ah:
- **Test (−20°C):** stops at 1.74 Ah → SoC only reaches **(2.9 − 1.74)/2.9 × 100 ≈ 40%**. The model is **never evaluated below 40% SoC** in the test set.
- **Val (−10°C):** stops at 2.03 Ah → SoC only reaches ≈ **30%**.

**Implication:** Reported test RMSE only reflects mid-to-high SoC (40–100%). The model isn't tested on deep-discharge behavior at cold temps. Technically valid (lab stopped there), but state this limitation explicitly in your report so the professor doesn't assume it handles 0–40% at −20°C. Optionally, also report RMSE on 25°C/10°C full data to show the full SoC range.

### 2. Possible drive-cycle double-counting (verify in outer temp folders)
The README warns: *"many US06, HWFET, UDDS, LA92, NN drive cycles were saved in a single long file with charges/pauses... The files were also split... some drive cycle data is included twice."*
- The **25°C inner folder** contains exactly **10 clean split files** (Cycles 1–4, US06, HWFTa, HWFTb, UDDS, LA92, NN) — **no double-counting there.** ✓
- But notebook 03's `load_temp_folder(OUTER_ROOT, temp)` globs **all `*.mat`** in the **outer** `10degC/0degC/-10degC/-20degC/Drive Cycles` folders. If any contain the long contiguous file, that data loads **twice**, inflating training counts. **Check the outer Drive Cycles folders** and exclude/dedupe the long files.

### 3. Stride comment inconsistency (minor)
Prose says *"Stride = 1: ... maximum data utilization"* but code sets `STRIDE = 10`. Not a bug — docstring contradicts code. With 90% overlap, adjacent windows are nearly identical. Fine for a first pass; align the comments.

### 4. `dt_arr[0]` uses prepended median (cosmetic)
Harmless. The CC loop starts at k=1, so it never uses index 0. No action needed.

---

## ⚠️ Important — don't "fix" the wrong thing in the EKF

The EKF's **measurement model** `V_pred = ocv_lookup(x_pred[0]) + I * R_IN + x_pred[1]` and **RC update** `vth_pred = alpha * x[1] + R_TH * (1.0 - alpha) * I` use an **unusual but self-consistent sign convention** — the code's `V_th` is the *negative* of the conventional polarization voltage. The two "odd" signs cancel out, so terminal voltage is computed correctly. **Only change the SoC process sign** (`-` → `+` in the soc_pred line). Do NOT flip the V_th or measurement signs.

---

## Module Reference

| File/Path | Purpose |
|---|---|
| `src/data_loader.py` | `load_panasonic_mat()` loads .mat → clean DataFrame with ground-truth SoC. Correct. |
| `notebooks/01_EDA_Panasonic.ipynb` | EDA: structure, OCV-SoC, multi-temperature, drive cycles, coulometer validation. Solid. |
| `notebooks/02_Baselines.ipynb` | Coulomb Counting (correct) + EKF (**has the SoC sign bug**). |
| `notebooks/03_Feature_Engineering.ipynb` | Normalization + sliding-window sequence generation, temperature split, saves `.npz`. Correct methodology (minor doc/verification issues). |
| `data/panasonic_sequences.npz` | Processed sequences for LSTM training (created by notebook 03). |
| `data/scaler_params.npz` | Saved min-max scaler + window size for inverse transform. |
| `results/baseline_metrics.csv` | **Currently misleading** — EKF 6.08% is a bug, not a real baseline. |

---

## Can You Proceed? — Yes, with this sequence

1. **Fix the EKF sign** in `notebooks/02_Baselines.ipynb`.
2. **Re-run** the baseline notebook to get the real EKF RMSE (expect ~1–3%). Update `results/baseline_metrics.csv`.
3. **Update the ML target** (notebook 03 summary) to the corrected EKF value — e.g., "target RMSE < 1.5%".
4. **Check the outer temperature Drive Cycles folders** for the long contiguous files; exclude if present.
5. **Document the test-set SoC limitation (40–100% at −20°C)** in your report.
6. **Then build `04_LSTM_Baseline.ipynb`** using the already-saved `data/panasonic_sequences.npz`.

Your pipeline and methodology are genuinely strong. The bug is a single line. Get it fixed, and you're ready to move forward.

---

*Full report saved to `project_info__1.md` in the project root.*