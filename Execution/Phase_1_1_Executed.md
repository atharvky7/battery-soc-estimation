# PHASE 1.1 — Domain Knowledge & Foundations: Executed

Welcome to Phase 1.1! This document is your foundational guide to understanding the physics, chemistry, and mathematics of Lithium-ion batteries. Before you can apply Artificial Intelligence to estimate the State of Charge (SoC), you must first master the system you are modeling. 

This document synthesizes information from battery fundamentals and specifically references the multi-level charging research detailed in the project's reference papers (e.g., Soumitra 2019, 2020, 2023). Let's dive in.

---

## A1. Why Are We Doing This Project? (Context)

At the heart of every modern electronic device—from smartphones to electric vehicles—is a **Battery Management System (BMS)**. A BMS is the "brain" of the battery pack. It is an electronic system that manages a rechargeable battery (cell or battery pack), protecting the battery from operating outside its safe operating area, monitoring its state, calculating secondary data, reporting that data, controlling its environment, and authenticating it.

For a BMS to function effectively, it needs to know the battery's real-time states, which are not physically measurable:
*   **State of Charge (SoC):** "How full is the battery?" (0% to 100%). This is the equivalent of a fuel gauge.
*   **State of Health (SoH):** "How degraded is the battery?" A new battery has 100% SoH. Over time, its capacity fades and internal resistance increases.
*   **State of Power (SoP):** "How much instantaneous power can the battery deliver safely right now?"

For this project, **State of Charge (SoC)** is the most critical parameter. Here is why:
1.  **User Experience:** It drives the battery percentage display on a mobile phone.
2.  **Safety:** It prevents overcharging (which can cause thermal runaway and fires) and prevents deep discharging (which irreversibly damages the cell chemistry).
3.  **Group 2 Integration:** Group 2 is building a **Zeta-Luo multilevel charger** (as detailed in *Soumitra 2023*). Their hardware logic requires the real-time SoC to determine the optimal charging current. For example, if the SoC is low, it pumps high current; if it's near 100%, it gently trickles current.

The fundamental engineering problem you are solving is this: **SoC cannot be measured directly.** There is no physical sensor you can insert into a battery to measure its "fullness." Instead, a BMS can only measure three physical quantities: **Voltage ($V$)**, **Current ($I$)**, and **Temperature ($T$)**. Your goal is to infer SoC from $V$, $I$, and $T$ using Machine Learning.

---

## A2. Li-ion Battery Chemistry — Fundamentals

To model a battery, you must understand what it is. A **Lithium-ion (Li-ion) cell** physically consists of four main components:
1.  **Anode (Negative Electrode):** Typically made of graphite.
2.  **Cathode (Positive Electrode):** Typically a metal oxide. Common types include NMC (Nickel Manganese Cobalt, used in our LG HG2 dataset), LCO (Lithium Cobalt Oxide, common in phones), and LFP (Lithium Iron Phosphate).
3.  **Electrolyte:** A liquid or polymer gel that allows Lithium ions ($Li^+$) to flow between the anode and cathode.
4.  **Separator:** A porous membrane that physically separates the anode and cathode to prevent short circuits while allowing ions to pass.

**What happens during charging?**
When you plug in a charger, it forces electrons into the anode through the external circuit. To balance this negative charge, positive Lithium ions ($Li^+$) detach from the cathode, swim through the electrolyte, pass through the separator, and embed themselves into the graphite layers of the anode (a process called intercalation).

**What happens during discharging?**
When you use the phone, the process reverses. The $Li^+$ ions leave the anode and travel back through the electrolyte to the cathode. The electrons travel through the external circuit (powering your phone) to meet them at the cathode.

**Why does voltage change with SoC?**
As ions move from one electrode to the other, the chemical potential (Gibbs free energy) of the electrodes changes. When the anode is packed with lithium (100% SoC), the potential difference between the anode and cathode is highest. As lithium moves to the cathode, this potential difference decreases.

**Key Specifications to remember:**
*   **Nominal Voltage:** The average operating voltage, typically 3.6V to 3.7V for NMC cells.
*   **Charging Cutoff:** The maximum safe voltage, typically 4.2V. Pushing past this damages the battery.
*   **Discharge Cutoff:** The minimum safe voltage, usually 2.5V to 3.0V. Draining below this causes copper dissolution, permanently ruining the battery.
*   **Capacity:** Measured in Amp-hours ($Ah$). If a battery has a capacity of 1 Ah, it can deliver 1 Amp of continuous current for 1 hour. Our dataset cells are 2.9 Ah (Panasonic) and 3.0 Ah (LG HG2), which perfectly maps to a 2900 - 3000 mAh smartphone battery.

---

## A3. The Battery Equivalent Circuit Model (Thevenin ECM)

We cannot easily use complex electrochemical differential equations in real-time BMS code. Instead, electrical engineers use an **Equivalent Circuit Model (ECM)** to simulate the battery using standard electrical components (resistors, capacitors, and voltage sources). 

The most common model used in literature is the **First-Order Thevenin Model**.

```text
    R_in          R_th
+--/\/\/--+----/\/\/----+
|         |             |
V_OC     C_th          V_terminal
|         |             |
+---------+-------------+
```

### Components of the Thevenin Model:

1.  **$V_{OC}$ (Open Circuit Voltage):**
    *   This is an ideal voltage source representing the "true" resting voltage of the battery when zero current is flowing.
    *   It is purely a mathematical function of SoC: $V_{OC} = f(SoC)$.
    *   If you disconnect a battery and let it rest for 2+ hours, the voltage you measure at the terminals will exactly equal $V_{OC}$.

2.  **$R_{in}$ (Internal or Ohmic Resistance):**
    *   Units: Ohms ($\Omega$) or milliohms ($m\Omega$).
    *   No battery is a perfect conductor. The electrolyte, terminals, and electrodes have physical resistance.
    *   When current $I$ flows, it causes an instantaneous voltage drop (or rise during charge): $V_{drop} = I \times R_{in}$.
    *   This resistance causes the battery to heat up: $P_{heat} = I^2 \times R_{in}$. As the battery ages (SoH decreases), $R_{in}$ increases.

3.  **$R_{th}$ and $C_{th}$ (Polarization RC Network):**
    *   Chemical reactions in the battery (charge transfer and diffusion) do not happen instantly. 
    *   When you draw current, the voltage drops instantly due to $R_{in}$, but then it *continues* to drop slowly over several seconds or minutes. When you stop drawing current, the voltage slowly creeps back up to $V_{OC}$.
    *   This "chemical delay" or polarization is modeled by putting a resistor ($R_{th}$) and capacitor ($C_{th}$) in parallel.
    *   The voltage across this RC network is the polarization voltage, $V_{th}$.
    *   **Time Constant ($\tau$):** $\tau = R_{th} \times C_{th}$. This dictates how many seconds it takes for the polarization voltage to settle.

### The Master Equations:

The total measurable voltage at the battery terminals is the sum of these components:
$$V_{terminal}(t) = V_{OC}(SoC(t)) + I(t) \times R_{in} + V_{th}(t)$$
*(Note: Current $I(t)$ sign convention varies. Here, we assume $I(t)$ is positive during charge and negative during discharge, or you must adapt the signs accordingly. For the equations below, assume $I(t)$ is positive for discharge, meaning the terminal voltage drops).*
Let's define $I(t)$ as **positive during discharge** and **negative during charge**.
$$V_{terminal}(t) = V_{OC}(SoC(t)) - I(t) \times R_{in} - V_{th}(t)$$

To find $V_{th}(t)$, we use the continuous RC transient differential equation:
$$\frac{d V_{th}}{dt} = -\frac{V_{th}}{R_{th} \times C_{th}} + \frac{I(t) \times R_{th}}{R_{th} \times C_{th}}$$

**Discretized for Python implementation:**
Computers cannot solve continuous time, so we sample at intervals of $dt$ seconds. The discretized equation for the RC voltage at step $k+1$ is:
$$V_{th}[k+1] = \exp\left(-\frac{dt}{R_{th} \times C_{th}}\right) \times V_{th}[k] + R_{th} \times \left(1 - \exp\left(-\frac{dt}{R_{th} \times C_{th}}\right)\right) \times I[k]$$

```python
# Python snippet for Thevenin ECM step
import math

def calculate_next_vth(vth_prev, I_curr, dt, R_th, C_th):
    decay = math.exp(-dt / (R_th * C_th))
    vth_next = decay * vth_prev + R_th * (1 - decay) * I_curr
    return vth_next

def calculate_v_terminal(v_oc, I_curr, R_in, v_th):
    # Assuming I_curr is positive for discharge
    return v_oc - (I_curr * R_in) - v_th
```

---

## A4. The OCV-SoC Relationship

The $V_{OC}$ vs. SoC curve is the holy grail of traditional battery management. It is a lookup table or mathematical curve that relates the resting voltage to the state of charge.

**How is it measured?**
You fully charge the battery to 100%, let it rest, then discharge it extremely slowly (at a rate of $C/20$, meaning it takes 20 hours to empty). Because the current is so small, $I \times R_{in} \approx 0$ and $V_{th} \approx 0$. Therefore, $V_{terminal} \approx V_{OC}$. You can extract this curve directly from the `C/20 discharge` `.mat` file in the Panasonic dataset.

**Shape:**
For NMC chemistry, the curve is highly non-linear. It is relatively flat between 20% and 80% SoC (hovering around 3.7V - 3.9V) and then drops like a cliff as it approaches 0%. 

**The Temperature Effect:**
This is critical. The chemical potential of the battery is temperature-dependent. At $-10^\circ C$, a battery sitting at 50% SoC will have a noticeably lower $V_{OC}$ than the same battery at $25^\circ C$. 
*   **Why this matters:** If a traditional BMS only uses one OCV-SoC lookup table, it will grossly underestimate the SoC in winter (causing the phone to shut down prematurely).
*   **Why ML solves this:** Machine Learning models take Temperature as a direct input feature. The ML model implicitly learns how the OCV-SoC curve shifts across temperatures without needing multiple discrete lookup tables.

---

## A5. State of Charge — Definition & All Formulas

State of Charge (SoC) is fundamentally a ratio of remaining charge to total capacity.

**Continuous Definition:**
$$SoC(t) = SoC(0) - \left( \frac{1}{Q_{max}} \int_{0}^{t} I(\tau) d\tau \right) \times 100\%$$

*   $SoC(0)$: Initial SoC at time 0 (usually 100% after a full charge).
*   $Q_{max}$: Nominal capacity in Amp-hours ($Ah$). For Panasonic, $Q_{max} = 2.9 Ah$.
*   $I(\tau)$: Current in Amps. (Convention here: Positive $I$ means discharging).

**Discrete Formula (Coulomb Counting):**
When writing code, you sum up (integrate) the current at every timestep $dt$. Because capacity is in Amp-*hours* and $dt$ is in *seconds*, you must divide by 3600 to convert seconds to hours.
$$SoC[k+1] = SoC[k] - \frac{I[k] \times dt}{Q_{max} \times 3600} \times 100$$

```python
# Python snippet for Coulomb Counting
def coulomb_counting(current_array, dt_seconds, Q_max_Ah, initial_soc_percent=100.0):
    soc = [initial_soc_percent]
    current_soc = initial_soc_percent
    
    for I in current_array: # I is positive for discharge
        # dt / 3600 converts seconds to hours
        current_soc = current_soc - ( (I * (dt_seconds / 3600)) / Q_max_Ah ) * 100
        soc.append(current_soc)
        
    return soc
```

**From the Panasonic Dataset (`Ah` column):**
The Panasonic dataset conveniently tracks the cumulative Amp-hours in the `Ah` column (which is negative during discharge). You can instantly find the Ground Truth SoC:
$$SoC_{true}(t) = \frac{Q_{max} + Ah(t)}{Q_{max}} \times 100$$

---

## A6. Why Traditional Methods Fail

Before ML, engineers relied on three main model-based methods, all of which have severe flaws.

### Method 1: Direct Voltage Measurement
*   **The Idea:** Just read $V_{terminal}$ and look it up on the OCV-SoC curve.
*   **Why it fails:** As established in A3, $V_{terminal} \neq V_{OC}$ when the device is running. If you draw 2 Amps through a $50 m\Omega$ internal resistance, the voltage instantly drops by $100 mV$. 
*   Because the OCV-SoC curve is very flat in the middle, a $100 mV$ error in voltage can easily map to a **20% error in SoC**. Your phone would jump from 60% to 40% the second you open a game.

### Method 2: Coulomb Counting (Ampere-Hour Integration)
*   **The Idea:** Start at 100% and integrate the current over time (as shown in A5).
*   **Why it fails:** 
    1.  **Sensor Noise Drift:** Current sensors have inherent noise (e.g., $0.05 A$ offset). Integrating this noise over time causes massive drift. An error of $0.05A \times 3600s = 0.18 Ah$ error per hour. On a $3.0Ah$ battery, that is a **6% SoC error accumulating every single hour**.
    2.  **Initial SoC:** If you reboot your phone at 50% SoC, Coulomb counting doesn't know where to start the integration unless the battery has rested long enough to measure $V_{OC}$.
    3.  **Capacity Fade:** As the battery ages, $Q_{max}$ decreases. If the software still divides by the factory $2.9 Ah$, the SoC will be totally wrong.

### Method 3: Extended Kalman Filter (EKF)
*   **The Idea:** Treat the Thevenin ECM as a mathematical "system." Use Coulomb counting to predict the state, then use the measured $V_{terminal}$ to correct the prediction.
*   **Why it fails:** 
    1.  The EKF is only as good as its system model. It requires highly accurate values for $R_{in}$, $R_{th}$, and $C_{th}$. 
    2.  These parameters change drastically depending on Temperature, SoC, and battery age. Calibrating them is a nightmare.
    3.  The EKF relies on linearizing a highly non-linear system (using Taylor series expansion), introducing mathematical linearization errors.

---

## A7. Why Machine Learning is the Right Approach

Machine learning represents a massive paradigm shift in battery management: moving from **Model-Based** to **Data-Driven** estimation.

*   **Traditional:** Physicist writes equations (Thevenin ECM) $\rightarrow$ parameterize model $\rightarrow$ estimate SoC. *Problem: The model is always wrong/simplified to some degree.*
*   **ML:** Feed thousands of hours of real $(V, I, T)$ measurements into an algorithm $\rightarrow$ algorithm finds hidden patterns $\rightarrow$ estimates SoC. *Advantage: The model learns the true physics from the data, even the physics that traditional equations fail to capture.*

**Advantages for Battery SoC:**
1.  It automatically learns the non-linear $V_{OC}$-SoC curve.
2.  It learns temperature effects implicitly. You don't need separate $R_{in}$ maps for $10^\circ C$ and $25^\circ C$; the model learns how $T$ influences $V$ and $I$.
3.  No circuit parameterization is required. No need to calculate RC time constants.
4.  It generalizes perfectly to dynamic, unpredictable loads (like a user playing games, texting, and sleeping).

**Why LSTM over simple Neural Networks (FNN)?**
A Feed-Forward Neural Network (like the one in the LG HG2 dataset MATLAB script) looks at $V, I, T$ at exactly time $t$ and tries to guess SoC. 
However, **SoC is time-dependent.** A battery sitting at $3.7V$ after resting for an hour is at a very different SoC than a battery sitting at $3.7V$ right after a massive 10-Amp power spike. 
**Long Short-Term Memory (LSTM)** networks contain internal memory states. They can remember what happened 100+ timesteps ago, capturing the polarization ($V_{th}$) delay inherently!

**Why CNN-LSTM is the Ultimate Architecture:**
*   **CNN (Convolutional Neural Network):** Excellent at extracting *local spatial patterns*. In time-series, a 1D-CNN acts as a feature extractor. It learns to identify shapes in the data, like "this rapid up-and-down current pattern is typical of CPU processing."
*   **LSTM:** Takes those extracted features and learns how they evolve over time.
*   **Result:** The CNN handles the noisy, spiky short-term features, while the LSTM handles the long-range temporal dependencies (like Coulomb counting). This combination historically yields the lowest RMSE.

---

## A8. The Reference Papers — What They Contribute

Your project does not exist in a vacuum. It directly builds upon the charging architecture developed by your professor/group.

1.  **Soumitra 2019 (NPEC) - Multilevel Charging:**
    *   **Contribution:** Proposed a 5-level Constant Current (CC) charging scheme based on SoC to enable fast charging without damaging the battery.
    *   **Old Method:** Used "Modified Coulomb Counting" (CC combined with OCV corrections during intentional rest periods).
    *   **Your Role:** Our Deep Learning model *replaces* their Modified Coulomb Counting. We provide a continuous, highly accurate SoC without needing to pause the charging for OCV rests.

2.  **Soumitra 2020 (KPEC) - LMS-Based Online SoC Estimation:**
    *   **Contribution:** Upgraded the 2019 paper by replacing Modified CC with a Least Mean Square (LMS) adaptive filter to estimate SoC online. 
    *   **Limitations:** LMS is still a model-based adaptive filter and suffers from parameter tuning issues.
    *   **Your Role:** Our ML model *replaces* the LMS estimator, proving that data-driven approaches outperform adaptive filters.

3.  **Soumitra 2017:**
    *   **Contribution:** Explores the 5-level charging scheme in the context of Plug-in Hybrid EVs (PHEVs). This serves as the historical background for the group's research trajectory.

4.  **Soumitra 2023 (NPEC) - Zeta-Luo Converter:**
    *   **Contribution:** The hardware blueprint for Group 2. They are physically building a "Bridgeless Isolated Zeta-Luo converter."
    *   **Integration:** This hardware acts as the smart charger. Its microprocessor requires the real-time SoC to decide which of the 5 charging levels to output. Your ML model is the software brain feeding data into this hardware system.

---

## A9. The Full Project Pipeline (Big Picture)

To understand where you and Atharva fit into the grand scheme of the project alongside Group 2, look at this system pipeline:

```text
[Mobile Phone Battery]
          |
          |  Measure real-time Sensors
          v
   V(t), I(t), T(t)
          |
          |===========================================|
          |         GROUP 1 (ARPIT & ATHARVA)         |
          |-------------------------------------------|
          |       Deep Learning SoC Estimator         |
          |        (CNN-LSTM / BiLSTM Model)          |
          |===========================================|
          |
          |  Output: Real-time SoC(t) in %
          v
          |===========================================|
          |               GROUP 2 (A, B, C)           |
          |-------------------------------------------|
          |  Zeta-Luo Converter Control Logic Micro   |
          |                                           |
          |  if SoC < 20%  --> Level 1 (Fast Charge)  |
          |  if SoC 20-40% --> Level 2                |
          |  if SoC 40-60% --> Level 3                |
          |  if SoC 60-80% --> Level 4                |
          |  if SoC > 80%  --> Level 5 (Trickle)      |
          |===========================================|
          |
          v
[Battery Charged Safely & Faster than standard CC-CV]
```

**Your Ultimate Deliverable:** A Python function (or edge-deployed C++ script) that takes in an array of recent $V, I, T$ values and returns `float SoC_percentage`. 

---

## A10. Vocabulary Reference Table

Use this as your cheat sheet when writing your report or talking to your professor.

| Term | Definition |
| :--- | :--- |
| **SoC** | **State of Charge:** Percentage of remaining capacity in the battery (0% to 100%). |
| **SoH** | **State of Health:** Ratio of the battery's current maximum capacity to its original factory capacity (degrades over time). |
| **SoP** | **State of Power:** The maximum power the battery can safely supply or accept at the current instant. |
| **OCV / $V_{OC}$** | **Open Circuit Voltage:** The resting voltage of the battery when no load is applied. Directly correlates to SoC. |
| **ECM** | **Equivalent Circuit Model:** Using electrical components (resistors, capacitors) to simulate electrochemical battery behavior. |
| **Thevenin Model** | A specific ECM featuring a voltage source, internal resistance, and one or more RC networks. |
| **$R_{in}$** | **Internal Resistance:** The ohmic resistance of the battery; causes instant voltage drops when current flows. |
| **$V_{th}$** | **Polarization Voltage:** The transient voltage across the RC network representing the slow chemical response of the battery. |
| **RC Network** | A Resistor-Capacitor parallel pair used to model time-delayed polarization effects. |
| **Time Constant ($\tau$)** | $\tau = R \times C$. The time it takes for the polarization voltage to reach ~63.2% of its final value. |
| **Coulomb Counting** | Estimating SoC by mathematically integrating the current flowing in/out of the battery over time. |
| **EKF** | **Extended Kalman Filter:** An advanced mathematical algorithm that estimates the internal state (SoC) by comparing predictions with actual voltage measurements. |
| **C-Rate** | A measure of the rate at which a battery is discharged relative to its maximum capacity (e.g., 1C rate discharges the battery in 1 hour). |
| **$Q_{max}$** | The nominal, maximum capacity of the battery, usually in Ah. |
| **Ground Truth SoC** | The mathematically perfect, "true" SoC calculated after the fact in a lab setting, used to train ML models. |
| **RMSE** | **Root Mean Square Error:** A metric evaluating the accuracy of the ML model. Lower is better. |
| **MAE** | **Mean Absolute Error:** Average absolute difference between predicted SoC and ground truth SoC. |
| **Train/Val/Test Split** | Dividing the dataset: Train (to teach model), Validation (to tune model), Test (unseen data to prove it works). |
| **LSTM** | **Long Short-Term Memory:** A type of Recurrent Neural Network architecture designed to remember long-term dependencies in time-series data. |
| **CNN** | **Convolutional Neural Network:** A network excellent at extracting local spatial/temporal patterns from input sequences. |
| **Transfer Learning** | Training a model on one dataset (Panasonic) and fine-tuning it with a small amount of data on another (LG HG2). |
| **Drive Cycle** | A standardized profile of varying power/current demands used to simulate real-world usage. |
| **UDDS / LA92 / US06 / HWFET** | Standard automotive drive cycles adapted to battery testing to provide highly dynamic, chaotic current loads. |
| **DST** | **Dynamic Stress Test:** A standardized battery test profile involving rapid pulses of charge and discharge. |

---

### What You Have Learned
You now understand the fundamental physical and chemical mechanisms of a Li-ion battery. You know why voltage alone is an unreliable indicator of charge, why Coulomb Counting drifts, and how the Thevenin Equivalent Circuit Model attempts to mathematically represent these dynamics. More importantly, you understand the exact reasons why deep learning (specifically CNN-LSTM) represents a superior approach for non-linear, temperature-dependent SoC estimation.

### What Comes Next in Phase 1.2
Now that you have the theoretical foundation, we will get our hands dirty with the data. In Phase 1.2, we will write our first Python scripts to open the Mendeley dataset, perform Exploratory Data Analysis (EDA), visualize the OCV-SoC curve, and clean sensor anomalies.
