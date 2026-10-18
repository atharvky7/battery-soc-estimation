# Group 1 Initial-SoC Estimation: System Integration Interface

**Document Identifier:** `ICD-GROUP1-GROUP2-001`  
**Status:** PROPOSED INTERFACE CONTRACT  
**Author:** Group 1 (State Estimation & Machine Learning)  
**Counterpart:** Group 2 (Battery Hardware, Power Electronics, BMS & Powertrain Control)  
**Baseline Model:** Group C Summary MLP (`SPEC-GROUP1-SOC-001`)

---

## 1. Scope & Objective

This Interface Control Document (ICD) establishes the functional, temporal, and data contracts between **Group 1** (Battery State of Charge Estimator) and **Group 2** (BMS Hardware, Power Converters, and Vehicle Supervisory Controller).

The current Group 1 baseline provides a **frozen cold-start initial-SoC estimator** designed to solve the initial condition problem ($t=0$) from an unconditioned early-drive observation window.

---

## 2. Group 1 Delivery Contract (Current Baseline)

```
                       ┌─────────────────────────────────────────┐
                       │           GROUP 2 HARDWARE              │
                       │    (Sensors, Pack, Power Converter)     │
                       └───────────────────┬─────────────────────┘
                                           │ Synchronous 10 Hz Streams
                                           │ V(t), I(t), T(t) [500 samples]
                                           ▼
┌────────────────────────────────────────────────────────────────────────┐
│                        GROUP 1 ESTIMATOR CORE                          │
│                                                                        │
│   50 s Ring Buffer (500x3) ──> Two-Stage Scaler ──> Group C MLP (2.9k) │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
                                   │ Single-shot Initial SoC [%]
                                   │ at t ≈ 50.0 s post key-on
                                   ▼
                       ┌─────────────────────────────────────────┐
                       │       SUPERVISORY CONTROLLER / EKF      │
                       │    (Continuous Tracking / Range Disp)   │
                       └─────────────────────────────────────────┘
```

### 2.1 Input Data Contract (Input to Group 1)
Group 1 requires three synchronous time-series channels:

1. **Terminal Voltage $V(t)$:**
   * Expected Unit: Volts ($\text{V}$)
   * Scale: Cell-level potential ($2.5\,\text{V} - 4.2\,\text{V}$)
2. **Load Current $I(t)$:**
   * Expected Unit: Amperes ($\text{A}$)
   * Sign Convention: **Negative = Discharge**, **Positive = Charge / Regeneration**
3. **Battery Temperature $T(t)$:**
   * Expected Unit: Degrees Celsius ($^\circ\text{C}$)
   * Location: Cell surface temperature

### 2.2 Observation Window Contract
* **Nominal Window Duration:** $50\,\text{s}$ nominal.
* **Sample Count:** Exactly $500$ samples.
* **Nominal Sampling Rate:** $10\,\text{Hz}$ ($\Delta t = 0.100\,\text{s}$).
* **Actual Elapsed Duration:** $\approx 49.9\,\text{s}$ ($499 \times 0.100\,\text{s}$).
* **Data Quality Constraints:**
  * Maximum inter-sample gap: $\max(\Delta t) \le 0.5\,\text{s}$.
  * Missing value tolerance: $0\%$ (no `NaN` / `Inf`).

### 2.3 Output Data Contract (Output from Group 1)
* **Estimated Quantity:** Initial State of Charge at the start of the observation window ($\widehat{\text{SoC}}_0$).
* **Output Unit:** Percent ($0.0\% \text{ to } 100.0\%$).
* **Format:** Single 32-bit floating point value (`float32`).
* **Clipping:** Enforced physical projection $[0.0, 100.0]\%$.

### 2.4 Timing & Execution Contract
* **Execution Frequency:** **Single-Shot Cold-Start Execution**.
* **Trigger:** Initiated upon vehicle "key-on" / converter activation.
* **Delivery Time:** Exactly once, immediately upon accumulation of the 500th sample ($t \approx 50.0\,\text{s}$).
* **Compute Latency:** $< 5\,\text{ms}$ on standard microcontrollers / ARM Cortex targets.
* **Role in System:** Supplies the true initial condition $\text{SoC}_0$ to initialize downstream Coulomb counting or EKF algorithms, eliminating long transient convergence delays.

---

## 3. Data Structures & API Specification

### 3.1 Python In-Memory API
```python
from scripts.run_initial_soc_inference import predict_initial_soc

# window_500x3: np.ndarray of shape (500, 3) -> [Voltage_V, Current_A, Temp_degC]
result = predict_initial_soc(window_500x3)

estimated_soc = result["predicted_soc_pct"]  # float in [0.0, 100.0]%
```

### 3.2 Embedded C/C++ Interface Concept
```c
#ifndef GROUP1_SOC_INTERFACE_H
#define GROUP1_SOC_INTERFACE_H

#include <stdint.h>
#include <stdbool.h>

#define GROUP1_WINDOW_SAMPLES 500

typedef struct {
    float voltage_V[GROUP1_WINDOW_SAMPLES];      /* Cell potential in Volts */
    float current_A[GROUP1_WINDOW_SAMPLES];      /* Amperes: neg=discharge, pos=charge */
    float temperature_C[GROUP1_WINDOW_SAMPLES];  /* Degrees Celsius */
} Group1_InputWindow_t;

typedef struct {
    float   initial_soc_pct;    /* Predicted SoC at window start [0.0 - 100.0]% */
    float   raw_model_soc_pct;  /* Unconstrained MLP prediction */
    bool    valid;              /* True if input passed quality validation */
    uint8_t error_code;         /* 0 = OK, 1 = Gap error, 2 = Range error */
} Group1_OutputEstimate_t;

/* Primary inference entry point */
void Group1_Estimate_Initial_SoC(
    const Group1_InputWindow_t* input_window,
    Group1_OutputEstimate_t*    output_estimate
);

#endif /* GROUP1_SOC_INTERFACE_H */
```

### 3.3 Hardware Memory & Compute Footprint
* **Ring Buffer RAM:** $500 \times 3 \times 4\,\text{bytes} = 6,000\,\text{bytes}$ ($\approx 5.86\,\text{KiB}$).
* **Model Parameters (Flash):** $2,945 \times 4\,\text{bytes} = 11,780\,\text{bytes}$ ($\approx 11.5\,\text{KiB}$).
* **Working RAM:** $< 1\,\text{KiB}$ during forward inference pass.
* **Arithmetic Operations:** $\approx 5,900$ FLOPs per inference call.

---

## 4. UNKNOWN — TO BE CONFIRMED WITH GROUP 2

The following parameters are currently unknown and depend entirely on Group 2's hardware, converter topology, and vehicle architecture. **These values must NOT be assumed or hardcoded by Group 1:**

| Parameter | Current Group 1 Assumption (Panasonic Prototype) | Confirmation Status | Impact on Group 1 Model |
| :--- | :--- | :--- | :--- |
| **1. Actual Battery Chemistry** | NCA ($\text{LiNiCoAlO}_2$, Panasonic 18650PF) | **UNKNOWN — TO BE CONFIRMED WITH GROUP 2** | High. OCV-SoC curve and cell impedance depend heavily on chemistry (LFP, NMC, NCA, Na-ion). |
| **2. Cell / Pack Configuration** | Single cell ($1\text{S}1\text{P}$ equivalent) | **UNKNOWN — TO BE CONFIRMED WITH GROUP 2** | High. If pack-level voltage is measured (e.g. $96\text{S}$), voltage divider / per-cell normalization is required. |
| **3. Required SoC Output Format** | Percentage $[0.0, 100.0]\%$ float | **UNKNOWN — TO BE CONFIRMED WITH GROUP 2** | Low. Trivially mapped to $[0.0, 1.0]$ fraction or CAN fixed-point (e.g. 0.1% / bit). |
| **4. Required Update Rate** | Single estimate after $50\,\text{s}$ | **UNKNOWN — TO BE CONFIRMED WITH GROUP 2** | Medium. Determines whether cold-start initialization is sufficient. |
| **5. Whether Only Initial SoC is Needed** | Yes (Initial SoC at $t=0$) | **UNKNOWN — TO BE CONFIRMED WITH GROUP 2** | Fundamental scope definition. |
| **6. Whether Continuous SoC Tracking is Required** | No (Group 1 currently provides initial estimator) | **UNKNOWN — TO BE CONFIRMED WITH GROUP 2** | Fundamental. If continuous tracking is needed, Group 1 must interface with Coulomb Counter / EKF. |
| **7. Controller Interface Protocol** | Function call / CSV file | **UNKNOWN — TO BE CONFIRMED WITH GROUP 2** | Medium. CAN bus, SPI, UART, shared memory, or ROS2 topic. |
| **8. Embedded Deployment Target** | x86_64 / PyTorch on CUDA/CPU | **UNKNOWN — TO BE CONFIRMED WITH GROUP 2** | Medium. Microcontroller target (STM32, TI TMS570, Infineon AURIX, ESP32) dictates C-code export / ONNX / TFLite Micro. |
| **9. Sensor Ranges & Tolerances** | Clean laboratory sensors (Arbin cycler) | **UNKNOWN — TO BE CONFIRMED WITH GROUP 2** | Medium. Shunt vs Hall sensor noise; ADC resolution ($12\text{-bit}$ vs $16\text{-bit}$). |
| **10. Actual Sampling Rate** | Uniform $10\,\text{Hz}$ ($\Delta t = 0.1\,\text{s}$) | **UNKNOWN — TO BE CONFIRMED WITH GROUP 2** | High. If BMS acquires at $1\,\text{Hz}$ or $100\,\text{Hz}$, resampling or buffer sizing must be adapted. |

---

## 5. Next Steps for Joint Integration

1. **Schedule Technical Handshake Meeting with Group 2:**
   Review Section 4 checklist with Group 2 leads.
2. **Obtain Sensor Hardware Specifications:**
   Acquire voltage/current ADC precision, sampling rate, and sensor noise figures to evaluate robustness against Notebook 05 findings.
3. **Establish Pack Topography:**
   Determine whether Group 1 receives individual cell voltages from an AFE (Analog Front End, e.g. BQ79616 / LTC6813) or aggregate pack voltage.
