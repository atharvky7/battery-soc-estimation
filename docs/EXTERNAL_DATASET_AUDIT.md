# External Dataset Audit & Compatibility Checklist

**Document Identifier:** `AUDIT-EXT-DATA-001`  
**Status:** ACTIVE AUDIT TEMPLATE  
**Primary Target:** CALCE INR 18650-20R (Samsung SDI 20R cylindrical cell, CALCE Battery Research Group, University of Maryland)  
**Reference Frozen Baseline:** Panasonic 18650PF Group C Estimator (`SPEC-GROUP1-SOC-001`)

---

## 1. Executive Summary & Purpose

Before evaluating the frozen Group C initial-SoC estimator on external battery data, this audit checklist establishes the criteria required for rigorous zero-shot and domain-shift evaluation.

**Methodological Boundary:**
* Do **NOT** retrain the model on candidate external data.
* Do **NOT** download external datasets automatically without verified repository storage.
* Evaluate candidate data through structured feature compatibility, physical signal scaling, and domain-shift risk classification.

---

## 2. 15-Point External Dataset Compatibility Checklist

| # | Audit Item | Frozen Panasonic Baseline | CALCE INR 18650-20R Requirement / Target | Compatibility Status / Action Required |
| :-: | :--- | :--- | :--- | :--- |
| **1** | **Terminal Voltage ($V$)** | Synchronously logged terminal voltage in Volts ($	ext{V}$). Range: $2.28\,	ext{V} - 4.32\,	ext{V}$. | Measured cell terminal voltage ($V$). Nominal $2.5\,	ext{V} - 4.2\,	ext{V}$. | **REQUIRED:** Must convert $	ext{mV} 	o 	ext{V}$ if logged in millivolts. |
| **2** | **Load Current ($I$)** | Synchronously logged load current in Amperes ($	ext{A}$). | Measured cell current ($I$). | **CRITICAL:** Verify sign convention: **Discharge must be Negative**, **Charge must be Positive**. |
| **3** | **Battery Temp ($T$)** | Synchronously logged surface temperature in $^\circ	ext{C}$. | Thermocouple surface temperature. | **REQUIRED:** Convert Kelvin/Fahrenheit to $^\circ	ext{C}$. Must be cell surface, not ambient chamber only. |
| **4** | **Sampling Interval ($\Delta t$)** | Nominal 10 Hz ($\Delta t = 0.100\,	ext{s}$), uniform time step. | Typically 1 Hz (or 10 Hz depending on cycler setup). | **CRITICAL:** If logged at 1 Hz, 50s window contains only 50 samples. Resampling/interpolation needed. |
| **5** | **Window Continuity** | 50 s nominal continuous observation ($500$ samples), $\max(\Delta t) \le 0.5\,	ext{s}$. | Continuous dynamic driving/pulsing segments $\ge 50\,	ext{s}$. | **REQUIRED:** Validate absence of cycler rest pauses or step transition gaps inside the 50s window. |
| **6** | **Initial SoC Diversity** | Windows sampled across wide initial SoC ($3.5\% 	ext{ to } 100\%$). | Dynamic schedules started at varied initial states of charge. | **REQUIRED:** Confirm test schedules begin at diverse states ($20\%, 50\%, 80\%$), not only $100\%$ full charge. |
| **7** | **SoC Reference Label** | Ah-integrated reference at exact start of window ($t=0$). | Coulomb counting reference $	ext{SoC}_{	ext{ref}}(t) = 	ext{SoC}_0 + rac{\int I dt}{Q_{	ext{nom}}}$. | **CRITICAL:** Must confirm how $	ext{SoC}_0$ and reference denominator $Q$ were established in CALCE metadata. |
| **8** | **Charge/Discharge Regime** | Dynamic automotive drive cycles (HWFET, UDDS, LA92, US06). | Dynamic stress tests (DST, FUDS, BJDST, or EV drive cycles). | **REQUIRED:** Separate dynamic driving regimes from constant-current characterization steps. |
| **9** | **Configuration Scale** | Single 18650 cylindrical cell in climate chamber. | Single 18650 cylindrical cell. | **COMPATIBLE:** Direct cell-level match (avoids pack balancing and multi-cell voltage confounding). |
| **10** | **Cell Chemistry** | NCA ($	ext{LiNiCoAlO}_2$) with graphite anode. | NMC/LMO ($	ext{LiNiMnCoO}_2 / 	ext{LiMn}_2	ext{O}_4$) blended cathode. | **POTENTIAL DOMAIN SHIFT:** Different OCV-SoC profile, slope, and phase transitions. |
| **11** | **Nominal Capacity** | $2.90\,	ext{Ah}$ ($2,900\,	ext{mAh}$). | $2.00\,	ext{Ah}$ ($2,000\,	ext{mAh}$). | **POTENTIAL DOMAIN SHIFT:** Same current produces $1.45	imes$ higher C-rate on CALCE cell. |
| **12** | **Temperature Conditions** | Train: 25, 10, 0°C; Val: -10°C; Test: -20°C. | Commonly tested at 0°C, 25°C, 45°C (verify sub-zero availability). | **REQUIRED:** Map available CALCE thermal chambers to frozen baseline test conditions. |
| **13** | **Current Dynamic Range** | Peak discharge $-21.0\,	ext{A}$ ($-7.2	ext{C}$), peak regen $+10.7\,	ext{A}$ ($+3.7	ext{C}$). | Samsung 20R rated for 20A continuous discharge. | **COMPATIBLE:** Power-cell rating accommodates high-rate dynamic pulses. |
| **14** | **Voltage Operating Limits** | Lower cutoff: $2.5\,	ext{V}$, Upper cutoff: $4.2\,	ext{V}$. | Samsung 20R cutoff: $2.5\,	ext{V}$ to $4.2\,	ext{V}$. | **COMPATIBLE:** Identical terminal voltage operating window. |
| **15** | **Label Methodology** | Ah-integration reset after CC-CV full charge. | Arbin / Maccor cycler amp-hour integration. | **REQUIRED:** Verify whether Ah reset occurred at full charge or during discharge rest periods. |

---

## 3. Unit Conversion & Normalization Protocol

Before extracting the 12 Group C features, external candidate raw signals must be mapped strictly to the input format expected by `initial_soc_scaler.npz`:

### 1. Terminal Voltage ($V$)
* **Target Unit:** Volts ($	ext{V}$).
* **Transformation:** If recorded in $	ext{mV}$:
  $$V_{	ext{target}} = rac{V_{	ext{raw}}}{1000.0}$$

### 2. Load Current ($I$)
* **Target Unit:** Amperes ($	ext{A}$).
* **Sign Convention Check:** If discharge is recorded as positive current (common in certain battery cyclers):
  $$I_{	ext{target}} = -1.0 	imes I_{	ext{raw}}$$
* **Scale Check:** If recorded in $	ext{mA}$:
  $$I_{	ext{target}} = rac{I_{	ext{raw}}}{1000.0}$$

### 3. Battery Surface Temperature ($T$)
* **Target Unit:** Degrees Celsius ($^\circ	ext{C}$).
* **Transformation:** If recorded in Kelvin:
  $$T_{	ext{target}} = T_{	ext{raw}} - 273.15$$

### 4. Sampling Uniformity & Window Interpolation
* If CALCE data is sampled at $1\,	ext{Hz}$ ($1.0\,	ext{s}$ interval):
  A 50 s window contains only 51 samples. Direct feed into a 500-sample array is **strictly invalid**.
  * **Option A:** Resample via monotonic cubic Hermite spline to 500 samples at 10 Hz.
  * **Option B (Recommended for baseline integrity):** Identify whether CALCE high-rate dynamic test logs recorded at 10 Hz exist.

---

## 4. Potential Domain Shift Analysis (External Validation Risks)

When evaluating the frozen Panasonic Group C model zero-shot on CALCE INR 18650-20R, performance degradation must be analyzed through the following explicit domain shift factors:

### POTENTIAL DOMAIN SHIFT 1: Cathode Chemistry & OCV Curvature
* **Panasonic 18650PF:** Nickel-Cobalt-Aluminum Oxide ($	ext{NCA}$). Characterized by a relatively steep OCV-SoC slope across the $30\% - 80\%$ plateau.
* **CALCE INR 18650-20R:** Nickel-Manganese-Cobalt / Lithium Manganese Oxide blend ($	ext{NMC/LMO}$). Exhibits distinct OCV plateaus, different thermodynamic equilibrium potentials, and distinct entropic heat coefficients.
* *Consequence:* An instantaneous voltage $V_0 = 3.70\,	ext{V}$ under zero/low current corresponds to a different true SoC in NMC/LMO than in NCA.

### POTENTIAL DOMAIN SHIFT 2: Capacity & C-Rate Scaling Mismatch
* **Panasonic 18650PF:** $2.90\,	ext{Ah}$ nominal.
* **CALCE 20R:** $2.00\,	ext{Ah}$ nominal.
* *Consequence:* A $-5\,	ext{A}$ discharge pulse represents a $-1.72	ext{C}$ load on the Panasonic cell, but a $-2.50	ext{C}$ load on the CALCE cell. For the same absolute current input, the CALCE cell will experience higher overpotentials and steeper polarization drop $dV$, which the frozen model may misinterpret as a lower initial SoC.

### POTENTIAL DOMAIN SHIFT 3: Internal Ohmic & Polarization Impedance
* The Samsung SDI 20R is a high-power cell (designed for power tools, lower internal resistance $\sim 15-20\,	ext{m}\Omega$) compared to the Panasonic 18650PF energy-optimized cell ($\sim 30-35\,	ext{m}\Omega$).
* *Consequence:* Under equivalent discharge current, the CALCE cell suffers smaller $IR$ drop than the training cell. Features like $V_0$, $V_{	ext{end}}$, and $	ext{mean}V$ will sit higher than expected for a given SoC and current, potentially inducing positive estimation bias.

### POTENTIAL DOMAIN SHIFT 4: Sampling Interval Mismatch (1 Hz vs 10 Hz)
* Group C summary features include dynamic endpoint differentials ($dV, dI, dT$) and voltage standard deviation ($	ext{std}V$).
* *Consequence:* If CALCE signals are acquired at 1 Hz, transient high-frequency voltage fluctuations during dynamic steps are filtered out or aliased, suppressing $	ext{std}V$ and distorting $dV/dI$ dynamic features.

### POTENTIAL DOMAIN SHIFT 5: Thermal Mass & Thermocouple Dynamics
* Surface thermocouple placement, convective heat transfer coefficient of the CALCE thermal fixture, and cell thermal time constant will alter the magnitude of $dT$ over a 50 s window.
* *Consequence:* While temperature change over 50 s is typically modest ($\sim 0.01 - 0.5^\circ	ext{C}$), discrepancies in thermocouple thermal resistance will shift normalized $T_0, T_{	ext{end}}, dT$.

### POTENTIAL DOMAIN SHIFT 6: SoC Ground Truth Reference Definition
* In Panasonic data, Reference SoC was defined as:
  $$	ext{SoC}_{	ext{ref}} = rac{2.9\,	ext{Ah} + 	ext{Ah}_{	ext{actual}}}{2.9\,	ext{Ah}} 	imes 100.0$$
* In external datasets, Reference SoC may be computed relative to:
  1. Manufacturer nominal capacity ($2.0\,	ext{Ah}$),
  2. Actual measured 1C discharge capacity at $25^\circ	ext{C}$, or
  3. Temperature-dependent discharge capacity ($Q(T)$).
* *Consequence:* A discrepancy in reference capacity definition creates a constant scaling offset in the ground-truth target itself.

### POTENTIAL DOMAIN SHIFT 7: Cell Aging & History
* The CALCE cell may have undergone prior degradation, cycle life testing, or calendar storage at elevated temperatures before the test log.
* *Consequence:* SEI growth and lithium inventory loss alter the voltage-to-SoC relationship independently of operational conditions.

---

## 5. Decision Protocol for External Evaluation

```
[Candidate Dataset Received]
            │
            ▼
[Step 1: Execute 15-Point Audit Checklist]
            │
            ├─> Missing V, I, or T? ───────────> REJECT
            ├─> Missing Continuous 50s Window? ─> REJECT
            ├─> Ambiguous Reference SoC? ──────> REJECT
            │
            ▼
[Step 2: Signal Conversion & Validation]
            │
            ├─> Ensure V [V], I [A, negative=discharge], T [°C]
            ├─> Validate sampling interval (10 Hz nominal)
            │
            ▼
[Step 3: Zero-Shot Frozen Group C Evaluation]
            │
            ▼
[Step 4: Quantify Domain Shift Residuals]
            │
            ▼
[Step 5: Compare Against Panasonic Validation (4.16%) & Test (10.15%)]
```
