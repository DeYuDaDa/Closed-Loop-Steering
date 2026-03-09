# Closed-Loop Steering System 

A dynamic, real-time activation steering pipeline for Large Reasoning Models (LLMs). This project transitions from traditional static activation injection (like Contrastive Activation Addition) to a rigorous **closed-loop control system**, solving critical issues like *Overthinking*, *State Shock*, and *Repetition Loops* in modern LLMs (e.g., Qwen3-8B).

## Background
In inference-time compute scaling (like Chain-of-Thought), models often fall into "System 1" semantic imitation, presenting high Deep-Thinking Ratios (DTR) that actually represent "Overthinking"—getting stuck in high-entropy local minima loops rather than performing valid reasoning. 

Standard physical interventions (like continuous vector injection or XML tag-triggered injection) often result in "State Shock" where the model's language degrades, or they happen too late to fix the logic. 

This system solves this by:
1. Monitoring the model's internal entropy in real-time.
2. Intervening at the exact moment of confusion (Anti-Overthinking).
3. Using norm-preserving spherical rotation instead of linear addition to preserve language fluency (Anti-State Shock).

## Project Structure

```text
Closed-Loop-Steering-System/
├── src/
│   ├── config.py                 # Centralized hyperparameters & paths
│   ├── run_experiment.py         # Main entry point for AIME logic evaluations
│   ├── extract_critic_vector.py  # Offline script to extract & purify CAA vectors
│   ├── state_monitor.py          # Real-time TECA & ThinkBrake monitors
│   ├── pid_controller.py         # Maps TECA entropy to rotation angle α
│   ├── spherical_injector.py     # Executes norm-preserving hidden state rotation
│   ├── manifold_utils.py         # PCA-based logic manifold projection
│   ├── vector_injector.py        # Manages loading/normalizing control vectors
│   ├── dtr_utils.py              # Evaluates Deep-Thinking Ratio & Perplexity
│   ├── evaluation_visualizer.py  # Generates publication-ready evaluation plots
│   └── aime_loader.py            # AIME benchmark dataset loader & parser
├── todo/                         # Original research & theoretical foundation docs
└── results/                      # Output directory for experiment logs & plots
```

## Setup & Usage

### 1. Extract and Purify the Control Vector
Before running experiments, extract the cognitive control vector from the dataset (`critic_data.json`) and purify it using Manifold Projection (PCA) to remove orthogonal noise:

```bash
cd src
python extract_critic_vector.py
```
*Outputs `critic.pt` and `critic_raw.pt` into the `vectors/` directory.*

### 2. Run the AIME Benchmark Experiment
Run the closed-loop evaluation pipeline. The script tests the model against the AIME dataset across three modes: `Baseline`, `Continuous`, and `Dynamic_Spherical`.

```bash
python run_experiment.py --dataset dataset/your_dataset.jsonl
```
*Outputs JSON metrics and a comprehensive 2x2 evaluation plot into `results/`.*

## Core Architecture
See `ARCHITECTURE.md` for a deep dive into the 5-module control pipeline (State Monitor, PID Controller, Spherical Steering, Manifold Projection, and Evaluator).
