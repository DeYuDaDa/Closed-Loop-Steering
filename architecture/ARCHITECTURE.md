# System Architecture

The **Closed-Loop Steering System** represents a paradigm shift from traditional "experience-based trial-and-error" activation steering to a rigorous, control-theory-backed dynamic intervention pipeline for Large Language Models (LLMs). It mitigates issues like *Overthinking*, *State Shock*, and *Repetition Loops* by intervening dynamically in the model's latent space.

The system is composed of **five core modules** working in a real-time closed loop.

---

## 1. Vector Purification and Manifold Projection (`manifold_utils.py` & `extract_critic_vector.py`)

**Objective:** Clean raw Contrastive Activation Addition (CAA) vectors to remove orthogonal noise that causes model repetition and state shock.

- **Extraction:** Calculates the mean difference in hidden states between positive (strict deductive reasoning) and negative (sycophancy/hallucination) generations derived from `critic_data.json`.
- **Manifold Projection (PCA):** Natural inference representations reside in a low-dimensional manifold. High-dimensional CAA vectors often carry noise that disrupts language generation. This module fits a PCA model on natural activations, extracts the top $k$ principal components, and projects the raw CAA vector onto this purified "logic manifold".
- **Result:** A purified, mathematically robust steering vector (`critic.pt`) that only affects the cognitive axis without degrading syntactic capabilities.

## 2. Real-time State Monitor (`state_monitor.py`)

**Objective:** Act as the "EEG probe" for the model to detect when it gets confused, eliminating the lag associated with surface-level XML tags (like `<critic>`).

- **TECA (Token Entropy Cumulative Average):** At every generation step, a custom `LogitsProcessor` computes the Shannon entropy of the model's token distribution. If the entropy consistently surges (TECA spikes), it physically signals that the model is facing prior conflicts and entering "Overthinking".
- **ThinkBrake (Logit Margin):** Continuously monitors the probability gap between the top token and a convergence token (e.g., `</think>`). If the margin shrinks below $\tau=0.25$, it signals that the underlying logic has converged, even if the model continues to generate padding words.

## 3. Dynamic PID Controller (`pid_controller.py`)

**Objective:** Map the cognitive confusion (TECA) to a physical intervention strength ($\alpha$) using classical control theory.

- **Mechanics:** Receives real-time $\text{TECA}_t$ from the State Monitor.
- **Error Calculation:** Uses $e_t = \text{TECA}_t - \text{SetPoint}$.
- **PID Math:** Adjusts the rotation angle output $\alpha$ based on Proportional, Integral, and Derivative terms over the history of errors.
- **Safety:** Implements anti-windup (conditional integration) to prevent runaway feedback loops and strictly clamps $\alpha$ to a safe maximum `ALPHA_MAX` threshold to prevent model collapse.

## 4. Spherical Steering Engine (`spherical_injector.py` & `vector_injector.py`)

**Objective:** Physically rotate the model's hidden states towards the purified cognitive vector without altering absolute vector magnitudes.

- **The Problem with Linear Addition:** Standard activation steering ($h + \alpha v$) changes the L2-norm of the hidden state $h$, pushing it outside the pre-training distribution manifold, resulting in perplexity spikes ("State Shock").
- **Spherical Rotation:** Uses Gram-Schmidt orthogonalization to construct a 2D hyperplane between the current hidden state $h$ and the control vector $v$. It rotates $h$ by angle $\alpha$ toward $v$, while enforcing that $||h_{new}||_2 == ||h_{original}||_2$.
- **Integration:** This runs via a PyTorch `register_forward_hook` at decoding time, executing the rotation purely iteratively on the residual stream of the targeted transformer layer.

## 5. Multi-dimensional Evaluation & DTR (`dtr_utils.py`, `evaluation_visualizer.py`, `aime_loader.py`)

**Objective:** Establish definitive proof of efficacy without runtime regressions.

- **Deferred DTR Calculation:** Deep-Thinking Ratio (DTR) calculation relies on Jensen-Shannon Divergence across layers to identify representations that delay convergence. The `DTRCalculator` dynamically runs an offline reconstruction pass (via replay hooks) to compute Local DTR within the exact intervention windows without bottlenecking real-time generation.
- **AIME Integration:** Standardizes logical evaluation around exact-match mathematical reasoning (Pass@1) via the `aime_loader.py`.
- **Comprehensive Visualization:** Evaluates the trade-offs over four key metrics:
  1. Logical Accuracy (Does it solve the prompt?)
  2. Dynamic Entropy Drop (Did TECA plummet post-intervention?)
  3. Language Stability (Did Perplexity and N-gram repetition remain stable?)
  4. Reasoning Efficiency (Token yield vs. Accuracy).

---

## The Execution Pipeline (`run_experiment.py`)

The overarching experiment script seamlessly orchestrates these modules across three paradigms:
1. **Baseline:** Standard zero-shot generation.
2. **Continuous:** Unconditional steering with fixed alpha. Validates the raw direction but often suffers logical collapse or structural truncation.
3. **Dynamic_Spherical (Our Approach):** Connects `StateMonitor` -> `PIDController` -> `InjectionState` -> `SphericalHook`. Sequences are processed individually because each maintains its own unique PID state and temporal trajectory.
