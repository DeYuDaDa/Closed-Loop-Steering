# Analysis and Investigation of Potential Answer Leakage

The user reported an unexpectedly high accuracy (>80%) on the AIME 2025 dataset using a Qwen3-8B model, suggesting a potential answer leak in the evaluation or prompt construction pipeline. After a code-level audit, I have identified a critical architectural bug in the inference engine and several areas to verify for potential leakage.

## User Review Required

> [!IMPORTANT]
> **Confirmed Architectural Bug (Cross-Slot Interference)**  
> The `run_continuous_batching_generation` function (second version) registers a steering hook to the model's target layer for *every* active inference slot but **fails to remove/isolate them**. This means when Slot 1 (e.g., Baseline) is being processed, steering hooks from Slot 2, Slot 3, etc. are also executed on its activations. While this is a serious bug, it usually causes output corruption rather than consistent answer leakage (unless the steering vectors themselves are contaminated).

> [!NOTE]
> **Accuracy Discrepancy**  
> Initial inspection of existing result files (`src/results/aime2025/experiment_results.json`) shows a Baseline accuracy of ~13%, which is much more realistic. I need to confirm if the user is running a newer or different version of the dataset/model that has triggered the 80%+ report.

## Proposed Changes

### 1. Diagnostic Investigation (Verification)

I will create a temporary diagnostic script `check_prompt_leak.py` to:
- Load the AIME 2025 dataset and build prompts exactly as `run_experiment.py` does.
- Run the tokenizer's chat template and print the final input string.
- Explicitly check if the string "answer" or the ground truth value appears in the input context.

### 2. Continuous Batching Engine Fix

#### [MODIFY] [run_experiment.py](file:///f:/academic/Closed-Loop-Steering-System/src/run_experiment.py)
- Refactor the hook registration to be slot-aware (e.g., by passing the current slot's batch index to the hook and having the hook only operate if it matches).
- Alternatively, ensure only the current slot's hook is registered on the model before its forward pass and removed immediately after. (Given the loop structure in the second version of `run_continuous_batching_generation`, this is the most robust fix).

### 3. Dataset Integrity Audit
- Verify if `load_aime_dataset` or its dependent loaders have been modified to append ground truth to the `problem` field during loading.

## Open Questions

- **Model Version**: Can you confirm if `qwen3-8b` is a base model or a math-tuned variant (like Qwen2.5-Math)? High accuracy on AIME is more common for math-tuned models using CoT.
- **Result Log**: Even though you said checking the results is difficult, can you provide a few lines of the text where the model starts its reasoning? If it immediately says "The answer is X", the leak is in the prompt. If it reasons correctly to the answer, it might just be model capability or contamination.

## Verification Plan

### Automated Tests
- Run `check_prompt_leak.py` to audit the final tokenized input.
- Run the code with `--max_concurrent_seqs 1` to isolate the effects of the cross-slot interference bug and see if accuracy changes.

### Manual Verification
- Review the printed prompts for any unintentional inclusions of the `answer` field.
