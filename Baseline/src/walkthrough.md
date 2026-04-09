# Walkthrough - Replicating Baseline Algorithms

I have successfully adapted and implemented the three baseline steering algorithms (SEAL, Spherical Steering, and CAA) to support the AIME and MATH500 datasets using your local loaders and vectors.

## Summary of Changes

### 1. SEAL-Steering Adaptation
The SEAL evaluation pipeline has been updated to use your local dataset loaders and the PCA-purified `critic.pt` vector.

- **[eval_MATH_steering.py](file:///f:/academic/Closed-Loop-Steering-System-writing/Baseline/src/SEAL-main/SEAL-main/eval_MATH_steering.py)**: Now supports `args.dataset == "AIME"` and uses `util.loaders` for data ingestion.
- **[get_math_results.py](file:///f:/academic/Closed-Loop-Steering-System-writing/Baseline/src/SEAL-main/SEAL-main/get_math_results.py)**: Updated to use the specific extraction and checking logic from `util.loaders` for more accurate evaluation on AIME/MATH500.

### 2. Spherical Steering Implementation
Since Spherical Steering requires two separate prototypes (`mu_T` and `mu_H`), I created an extraction tool and a new evaluation script.

- **[extract_prototypes.py](file:///f:/academic/Closed-Loop-Steering-System-writing/Baseline/src/Spherical-Steering-main/Spherical-Steering-main/extract_prototypes.py)**: A new utility that re-runs the extraction logic on `critic_data.json` to save the separate means as `prototypes.pt`.
- **[eval_math_spherical.py](file:///f:/academic/Closed-Loop-Steering-System-writing/Baseline/src/Spherical-Steering-main/Spherical-Steering-main/eval_math_spherical.py)**: A math-specific evaluation pipeline that uses `baukit.TraceDict` to apply the spherical rotation intervention during generation.

### 3. CAA (Contrastive Activation Addition) Baseline
Implemented a standard CAA baseline evaluation to compare against the more complex steering methods.

- **[eval_caa_math.py](file:///f:/academic/Closed-Loop-Steering-System-writing/Baseline/src/CCA/eval_caa_math.py)**: Implements linear steering on the last token of each generation step, following the original CAA implementation principles.

### 4. Manifold Steering Implementation
Implemented the Manifold Steering baseline which uses layer-wise ablation to suppress "overthinking" directions.

- **[extract_all_layers.py](file:///f:/academic/Closed-Loop-Steering-System-writing/Baseline/src/manifold/extract_all_layers.py)**: A utility to extract and purify CAA vectors for ALL 32 layers of the model.
- **[eval_math_manifold.py](file:///f:/academic/Closed-Loop-Steering-System-writing/Baseline/src/manifold/eval_math_manifold.py)**: Applies a layer-wise ablation hook ($h' = h - \alpha \cdot \text{Proj}_r(h)$) to all transformer blocks during generation.

## Next Steps for Execution

To run the full evaluation, follow these commands:

### A. Extract Prototypes (for Spherical Steering)
```bash
cd src/Spherical-Steering-main/Spherical-Steering-main/
python extract_prototypes.py
```

### B. Run SEAL Baseline
```bash
cd src/SEAL-main/SEAL-main/
python eval_MATH_steering.py --dataset AIME --steering --steering_vector ../../util/vectors/qwen3-8b/critic.pt --steering_layer 24 --steering_coef 1.0 --model_name_or_path /root/autodl-tmp/qwen3-8b
```

### C. Run Spherical Steering Baseline
```bash
cd src/Spherical-Steering-main/Spherical-Steering-main/
python eval_math_spherical.py --dataset AIME --prototypes_path ../../util/vectors/qwen3-8b/prototypes.pt --layer 24 --alpha 0.6 --model_name_or_path /root/autodl-tmp/qwen3-8b
```

### D. Run CAA Baseline
```bash
cd src/CCA/
python eval_caa_math.py --dataset AIME --vector_path ../util/vectors/qwen3-8b/critic.pt --multiplier 1.0 --layer 24 --model_name_or_path /root/autodl-tmp/qwen3-8b
```

### E. Run Manifold Steering Baseline
First, extract layer-wise vectors:
```bash
cd src/manifold/
python extract_all_layers.py
```
Then run evaluation (applying intervention to all layers):
```bash
python eval_math_manifold.py --dataset AIME --vectors_dir ../util/vectors/qwen3-8b/layer_wise/ --alpha 0.3 --model_name_or_path /root/autodl-tmp/qwen3-8b
```

> [!NOTE]
> Ensure that the `MODEL_PATH` in `src/config.py` correctly points to your local Qwen3-8b model before running.
