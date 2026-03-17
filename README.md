# Dynamic Closed-Loop Steering: Taming Overthinking in Large Reasoning Models

> **Taming System-2 Overthinking via Manifold Projection and Spherical Intervention.**

---

## 🌟 Overview

As Large Language Models (LLMs) evolve from "System-1" (rapid intuitive generation) to "System-2" (deliberate reasoning), scaling **Test-Time Compute** has become the cornerstone for unlocking complex problem-solving. However, unconstrained thinking often leads to **"Overthinking"** — a trap where models get stuck in repetitive, low-value, or erroneous logical loops.

**Dynamic Closed-Loop Steering** is a first-of-its-kind framework that transforms LLM intervention from "open-loop trial-and-error" to an **L4-level autonomous steering system**. By combining control theory with high-dimensional geometry, it detects cognitive confusion in real-time and applies corrective steering that is mathematically "pure" and "norm-preserving."

---

## 🛠️ Core Architecture (The 4-Module Pipeline)

The system operates as a real-time closed loop, monitoring the model's internal state and intervening only when necessary.

### 1. 📡 Sensor: High-Agility Multi-Dimensional Probes
*   **EMA Entropy (`state_monitor.py`)**: Monitors Shannon entropy of token distributions using **Exponential Moving Average (EMA)** to filter noise and capture genuine "cognitive confusion."
*   **ThinkBrake (`state_monitor.py`)**: Uses **Logit Margin** (the gap between top candidates and the convergence token `</think>`) to predict logical closure and instantly cut off intervention, preserving natural language flow.

### 2. 🎮 Controller: Error-Driven PD Regulator
*   **Agile PD Control (`pid_controller.py`)**: Maps cognitive deviation to intervention strength ($\alpha$) using a **Proportional-Derivative** architecture. It implements a "Steer when confused, retreat when clear" strategy.
*   **Anti-Windup**: Excludes the Integral (I) term to prevent "saturation" and ensure high responsiveness in long CoT trajectories.

### 3. 💎 Purifier: Manifold Projection
*   **PCA Logic Manifold (`manifold_utils.py`)**: Raw steering vectors are often contaminated with high-dimensional noise. We project the contrastive activation vectors onto a low-dimensional **"Logic Manifold"** fitted via PCA, ensuring the steering signal only affects cognitive depth without causing syntactic "State Shock."

### 4. 🧭 Actuator: Spherical Steering Engine
*   **Norm-Preserving SLERP (`spherical_injector.py`)**: Unlike traditional linear addition ($h + \alpha v$), we use **Spherical Linear Interpolation (SLERP)**. This performs a rotation in the latent space that strictly maintains the original vector norm, preventing the "Repetition Loops" and distribution shifts common in legacy steering methods.

---

## 📐 Mathematical Foundation

The framework is built on rigorous mathematical definitions:

*   **EMA Update**: $\text{EMA}_t = \beta \cdot H_t + (1 - \beta) \cdot \text{EMA}_{t-1}$
*   **ThinkBrake Margin**: $M_t = \log p(y_t^\star) - \log p(y_{\text{term}})$
*   **PD Strength**: $\alpha_t = K_p \cdot e_t + K_d \cdot (e_t - e_{t-1})$
*   **Spherical Rotation**: $\hat{h}_{\text{rotated}} = \cos(\theta_{\text{new}})v + \sin(\theta_{\text{new}})u$ (where $\theta_{\text{new}}$ is the error-driven target angle).

---

## 🚀 Getting Started

### 1. Configuration
Centralized hyperparameters are located in `src/config.py`. Key settings include:
- `EMA_BETA`: Smoothing factor.
- `PID_KP` / `PID_KD`: Control gains.
- `ALPHA_MAX`: Safety ceiling for rotation.

### 2. Vector Extraction
Extract the purified control vectors from your reasoning data:
```bash
python src/extract_critic_vector.py
```

### 3. Running Experiments
Orchestrate the three primary modes:
```bash
python src/run_experiment.py
```
- **Baseline**: Standard generation.
- **Continuous**: Fixed-strength global steering.
- **Dynamic_Spherical**: Our adaptive, closed-loop approach.

---

## 📊 Evaluation & Metrics

We evaluate the "Cognitive ROI" (Return on Investment) of thinking tokens:
- **Local DTR (Deep-Thinking Ratio)**: Measures the surge in deep representation post-intervention.
- **Entropy Drop**: Confirms the transition from high-entropy confusion to low-entropy convergence.
- **Language Stability**: Monitors N-gram repetition and perplexity to ensure zero "State Shock."
- **Accuracy Flip**: Analyzes the net gain in logical correctness.

---

## 📁 Repository Structure

```text
├── src/
│   ├── state_monitor.py      # Sensing (EMA & ThinkBrake)
│   ├── pid_controller.py     # Control (PD Logic)
│   ├── manifold_utils.py     # Purification (PCA Projection)
│   ├── spherical_injector.py # Actuation (SLERP Rotation)
│   ├── run_experiment.py     # Experiment Orchestration
│   ├── dtr_utils.py          # Deep-Thinking Evaluation
│   └── loaders/              # Dataset interfaces (AIME, Math500, Zebra)
├── dataset/                  # Reasoning benchmarks
├── architecture/             # Documentation & Diagrams
└── todo/                     # Research notes & Algorithm specs
```

---

## 🔗 References & Credits

This work builds upon and critically improves current SOTA methods:
- **ThinkBrake**: Mitigating Overthinking in Tool Reasoning.
- **Spherical Steering**: Geometry-Aware Activation Rotation.
- **s1/o1**: Paradigms for scaling Test-Time Compute.
- **Manifold Steering**: Mitigating Overthinking via Manifold Projection.

---
> [!TIP]
> This framework achieves **Pareto Optimality** in its current configuration—significantly boosting accuracy and reasoning efficiency with **Zero Negative Interference** to language quality.
