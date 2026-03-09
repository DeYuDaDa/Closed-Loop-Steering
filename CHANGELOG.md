# Changelog

All notable changes to the Closed-Loop Steering System will be documented in this file.

## [Unreleased] - 2026-03-09

### Added
- **Real-time State Monitor (`state_monitor.py`)**: Implemented `StateMonitor` as a `LogitsProcessor` to compute TECA (Token Entropy Cumulative Average) and ThinkBrake margin recursively during generation without blocking the GPU pipeline.
- **PID Controller (`pid_controller.py`)**: Introduced a discrete PID controller with anti-windup protection to dynamically map TECA divergence to physical rotation angle ($\alpha$).
- **Spherical Steering Engine (`spherical_injector.py`)**: Transitioned from linear activation addition to norm-preserving spherical rotation using Gram-Schmidt orthogonalization. Resolves the "State Shock" logic collapse issue.
- **Manifold Projection (`manifold_utils.py` / `extract_critic_vector.py`)**: Added an offline PCA pipeline to project raw CAA target vectors onto the low-dimensional reasoning manifold, stripping away orthogonal noise that causes textual repetition.
- **AIME Benchmark Integration (`aime_loader.py`)**: Replaced hardcoded test questions with the official AIME jsonl datasets. Includes pass@1 evaluation, LaTeX `\boxed{}` answer extraction, and batching logic.
- **Publication-Ready Visualizer (`evaluation_visualizer.py`)**: Automated the generation of a 2x2 grid containing logical accuracy, TECA entropy drop trajectories, language stability (repetition), and token efficiency.
- **Documentation**: Drafted `README.md`, `ARCHITECTURE.md`, and this `CHANGELOG.md` to properly document the architecture and theoretical foundation inherited from the original research parameters.

### Changed
- **Experiment Execution (`run_experiment.py`)**: Radically refactored the legacy tag-based system (`run_dtr_experiments.py`). Now supports concurrent batch generation for `Baseline` and `Continuous` modes, and sequential processing for the stateful `Dynamic_Spherical` mode.
- **DTR Evaluation (`dtr_utils.py`)**: Overhauled DTR calculation to deferred/offline execution. Offloaded tensor calculation directly to CPU immediately after generation to prevent VRAM fragmentation/OOM issues, recalculating via replay hooks. 
- **CUDA Allocator**: Configured `PYTORCH_CUDA_ALLOC_CONF` for `expandable_segments:True` to fix sustained GPU utilization degradation and memory fragmentation bugs during long evaluation runs.

### Fixed
- Fixed an `AttributeError` in `run_experiment.py` where `state.active` was erroneously referenced instead of the initialized state attributes.
- Fixed a bug causing previous experiment results to be overwritten by injecting timestamped directory generation for the `results/` folder outputs.
- Enforced `do_sample=True` explicitly in `run_experiment.py` during inference to match proper sampling constraints required by the TECA algorithms.
