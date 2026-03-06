"""
Closed-Loop Steering System — Unified Experiment Runner
==========================================================
Replaces the old tag-based `run_dtr_experiments.py`.

Runs three experiment modes:
  1. Baseline: No intervention at all.
  2. Continuous: Fixed-strength spherical rotation at every decoding step.
  3. Dynamic_Spherical: TECA-driven PID → spherical rotation (our method).

Pipeline per experiment:
  Load model → Load AIME dataset → Run generation with hooks →
  Collect metrics (DTR, PPL, Repetition, Pass@1 Accuracy, Trajectories) →
  Generate visualizations.

AIME Evaluation Protocol:
  - Standard pass@1 exact-match integer scoring (0-999)
  - Answer extraction via \\boxed{} regex (academic standard)
  - Datasets run separately for parallel GPU execution

Batch Processing Strategy:
  - Baseline & Continuous: batched generation (batch_size from config)
  - Dynamic_Spherical: sequential (per-sequence PID state)
"""

import os
import sys
import json
import argparse
import torch
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    LogitsProcessorList,
)

from config import (
    MODEL_PATH,
    LAYER_ID,
    VECTOR_DIR,
    MAX_NEW_TOKENS,
    EXPERIMENT_MODES,
    RESULTS_DIR,
    ALPHA_MAX,
    DATASET_DIR,
    AIME_MAX_TOKENS,
    BATCH_SIZE,
)
from state_monitor import InjectionState, StateMonitor
from pid_controller import PIDController
from spherical_injector import create_steering_hook
from vector_injector import VectorInjector
from dtr_utils import (
    DTRCalculator,
    calculate_ppl,
    calculate_repetition_rate,
)
from evaluation_visualizer import PlotVisualizer
from aime_loader import (
    load_aime_dataset,
    list_aime_datasets,
    build_aime_prompt,
    extract_answer,
    check_answer,
    collate_prompts,
)


def load_control_vector(vector_dir: str, device: str, dtype) -> torch.Tensor | None:
    """Load and normalize the critic control vector."""
    try:
        injector = VectorInjector(vector_dir, device=device, model_dtype=dtype)
        if injector.activate("critic", coeff=1.0):
            v = injector.get_normalized_vector()  # shape [1, 1, d]
            # Normalize to unit vector
            v_flat = v.view(-1)
            v_normalized = v_flat / v_flat.norm()
            v_normalized = v_normalized.view(v.shape)
            injector.deactivate()
            return v_normalized
    except Exception as e:
        print(f"⚠️  Failed to load control vector: {e}")
    return None


# ======================== Single-Sequence Generation ========================
# Used for Dynamic_Spherical mode (per-sequence PID state)

def run_single_generation(
    model,
    tokenizer,
    prompt: str,
    mode: str,
    control_vector: torch.Tensor | None,
) -> dict:
    """
    Run generation for a single prompt under a given experiment mode.

    Returns a dict with:
        text, tokens, teca_trajectory, alpha_trajectory,
        history_hidden, intervention_start, intervention_end
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs.input_ids.shape[1]

    # ---- Set up the closed-loop pipeline ----
    state = InjectionState()

    # PID controller (only for Dynamic_Spherical)
    pid = PIDController() if mode == "Dynamic_Spherical" else None

    # Resolve </think> token ID for ThinkBrake
    term_token_id = None
    try:
        term_ids = tokenizer.encode("</think>", add_special_tokens=False)
        if term_ids:
            term_token_id = term_ids[-1]
    except Exception:
        pass

    # State monitor (LogitsProcessor)
    monitor = StateMonitor(
        state=state,
        pid_controller=pid,
        term_token_id=term_token_id,
    )
    processors = LogitsProcessorList([monitor])

    # Steering hook
    if control_vector is not None:
        hook_fn, history_hidden = create_steering_hook(
            state=state,
            control_vector=control_vector,
            mode=mode,
            continuous_alpha=0.15,
        )
    else:
        # Fallback: no-op hook that still records hidden states
        history_hidden = []

        def hook_fn(module, args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            history_hidden.append(hidden[:, -1, :].detach().cpu())
            return output

    # Register hook at target layer
    layer = model.model.layers[LAYER_ID]
    handle = layer.register_forward_hook(hook_fn)

    # ---- Generate ----
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=AIME_MAX_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            logits_processor=processors,
        )

    handle.remove()

    # ---- Extract results ----
    generated_ids = output_ids[:, input_len:]
    gen_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    tokens = [
        tokenizer.decode([t]).replace("\n", "↵") for t in generated_ids[0]
    ]

    return {
        "text": gen_text,
        "tokens": tokens,
        "num_tokens": len(tokens),
        "output_ids": output_ids,
        "input_len": input_len,
        "teca_trajectory": state.teca_trajectory,
        "alpha_trajectory": state.alpha_trajectory,
        "entropy_trajectory": state.entropy_trajectory,
        "history_hidden": history_hidden,
        "intervention_start": state.intervention_start_step,
        "intervention_end": state.intervention_end_step,
    }


# ======================== Batched Generation ========================
# Used for Baseline and Continuous modes

def run_batched_generation(
    model,
    tokenizer,
    prompts: list[str],
    mode: str,
    control_vector: torch.Tensor | None,
    batch_size: int = BATCH_SIZE,
) -> list[dict]:
    """
    Run batched generation for Baseline or Continuous modes.

    These modes use uniform intervention (none or fixed-alpha), so
    batch processing is safe.

    Args:
        model: The loaded causal LM.
        tokenizer: The tokenizer.
        prompts: List of prompt strings.
        mode: "Baseline" or "Continuous".
        control_vector: The steering vector (or None).
        batch_size: Number of sequences per batch.

    Returns:
        List of result dicts (same format as run_single_generation).
    """
    all_results = []

    for batch_start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[batch_start:batch_start + batch_size]
        actual_bs = len(batch_prompts)

        # Tokenize with padding
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(model.device)

        # For batched mode, we use a simple shared state (Baseline/Continuous only)
        state = InjectionState()
        if mode == "Continuous":
            state.active = True
            state.alpha = 0.15  # Fixed continuous alpha

        # Steering hook (same alpha for all sequences in batch)
        history_hidden = []
        if control_vector is not None and mode == "Continuous":
            hook_fn, history_hidden = create_steering_hook(
                state=state,
                control_vector=control_vector,
                mode=mode,
                continuous_alpha=0.15,
            )
        else:
            def hook_fn(module, args, output):
                hidden = output[0] if isinstance(output, tuple) else output
                history_hidden.append(hidden[:, -1, :].detach().cpu())
                return output

        # Register hook
        layer = model.model.layers[LAYER_ID]
        handle = layer.register_forward_hook(hook_fn)

        # Generate
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=AIME_MAX_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        handle.remove()

        # Extract per-sequence results
        for i in range(actual_bs):
            # Find where the actual input ends (skip padding tokens)
            input_mask = inputs.attention_mask[i]
            input_len = input_mask.sum().item()

            generated_ids = output_ids[i, input_len:]
            # Remove padding tokens from generated output
            if tokenizer.pad_token_id is not None:
                generated_ids = generated_ids[
                    generated_ids != tokenizer.pad_token_id
                ]

            gen_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            tokens = [
                tokenizer.decode([t]).replace("\n", "↵")
                for t in generated_ids
            ]

            all_results.append({
                "text": gen_text,
                "tokens": tokens,
                "num_tokens": len(tokens),
                "output_ids": output_ids[i:i+1],
                "input_len": input_len,
                "teca_trajectory": [],
                "alpha_trajectory": [],
                "entropy_trajectory": [],
                "history_hidden": [],
                "intervention_start": None,
                "intervention_end": None,
            })

    return all_results


# ======================== Full Experiment Pipeline ========================

def run_full_experiment(
    model,
    tokenizer,
    control_vector: torch.Tensor | None,
    dtr_calc: DTRCalculator,
    dataset: list[dict],
    dataset_name: str = "AIME",
    modes: list[str] | None = None,
):
    """
    Run the full AIME benchmark across all modes.

    Uses batched generation for Baseline/Continuous and sequential
    generation for Dynamic_Spherical.

    Args:
        model: Loaded causal LM.
        tokenizer: Tokenizer.
        control_vector: Steering vector (or None).
        dtr_calc: DTR calculator instance.
        dataset: List of AIME problem dicts.
        dataset_name: Human-readable label for the dataset.
        modes: Experiment modes to run (defaults to config).

    Returns:
        experiment_results dict with per-mode metrics and per-problem details.
    """
    if modes is None:
        modes = EXPERIMENT_MODES

    experiment_results = {}

    for mode in modes:
        print(f"\n{'='*60}")
        print(f"  [{dataset_name}] EXPERIMENT MODE: {mode}")
        print(f"{'='*60}")

        # Build all prompts
        prompts = collate_prompts(dataset)

        # ---- Run generation ----
        if mode in ("Baseline", "Continuous"):
            print(f"  Using BATCHED generation (batch_size={BATCH_SIZE})...")
            results_list = run_batched_generation(
                model, tokenizer, prompts, mode, control_vector
            )
        else:
            # Dynamic_Spherical: sequential
            print(f"  Using SEQUENTIAL generation (per-sequence PID)...")
            results_list = []
            for idx, prompt in enumerate(prompts):
                print(f"    [{mode}] Problem {idx+1}/{len(prompts)}: "
                      f"id={dataset[idx]['id']}...")
                result = run_single_generation(
                    model, tokenizer, prompt, mode, control_vector
                )
                results_list.append(result)

        # ---- Evaluate each problem ----
        mode_correct = 0
        mode_total = len(dataset)
        mode_tokens_total = 0
        mode_repetitions = []
        mode_ppls = []
        mode_local_dtrs = []
        per_problem_details = []

        # Save first problem's trajectory for visualization
        first_teca_traj = []
        first_alpha_traj = []

        for i, (eval_item, result) in enumerate(zip(dataset, results_list)):
            # Extract and check answer
            predicted = extract_answer(result["text"])
            expected = eval_item["answer"]
            is_correct = check_answer(predicted, expected)

            mode_correct += int(is_correct)
            mode_tokens_total += result["num_tokens"]

            # Repetition rate
            rep = calculate_repetition_rate(result["text"])
            mode_repetitions.append(rep)

            # PPL
            try:
                ppl = calculate_ppl(model, tokenizer, result["text"])
                mode_ppls.append(ppl)
            except Exception:
                mode_ppls.append(float("nan"))

            # Local DTR
            try:
                if result["intervention_start"] and result["intervention_end"]:
                    w_start = result["input_len"] + result["intervention_start"]
                    w_end = result["input_len"] + result["intervention_end"]
                    local_dtr = dtr_calc.calculate_local_dtr(
                        result["output_ids"], w_start, w_end
                    )
                else:
                    dtr_scores, _ = dtr_calc.calculate(result["output_ids"])
                    local_dtr = dtr_scores[0]
                mode_local_dtrs.append(local_dtr)
            except Exception as e:
                mode_local_dtrs.append(0.0)

            # Save first trajectory
            if i == 0:
                first_teca_traj = result["teca_trajectory"]
                first_alpha_traj = result["alpha_trajectory"]

            # Per-problem detail record
            per_problem_details.append({
                "id": eval_item["id"],
                "expected": expected,
                "predicted": predicted,
                "correct": is_correct,
                "num_tokens": result["num_tokens"],
                "repetition": rep,
            })

            status = "✅" if is_correct else "❌"
            print(f"  {status} Problem {eval_item['id']}: "
                  f"pred={predicted} expected={expected} | "
                  f"Tokens={result['num_tokens']} | Rep={rep:.3f}")

        # ---- Aggregate mode results ----
        accuracy = mode_correct / mode_total if mode_total > 0 else 0
        avg_rep = np.mean(mode_repetitions) if mode_repetitions else 0
        avg_ppl = np.nanmean(mode_ppls) if mode_ppls else float("nan")
        avg_tokens = mode_tokens_total // max(mode_total, 1)
        avg_dtr = np.mean(mode_local_dtrs) if mode_local_dtrs else 0

        experiment_results[mode] = {
            "accuracy": accuracy,
            "correct_count": mode_correct,
            "total_count": mode_total,
            "repetition": avg_rep,
            "ppl": avg_ppl,
            "tokens": avg_tokens,
            "local_dtr": avg_dtr,
            "teca_trajectory": first_teca_traj,
            "alpha_trajectory": first_alpha_traj,
            "per_problem": per_problem_details,
        }

        print(f"\n  [{mode}] SUMMARY: "
              f"Pass@1={accuracy:.2%} ({mode_correct}/{mode_total}) | "
              f"AvgTokens={avg_tokens} | Rep={avg_rep:.3f} | "
              f"PPL={avg_ppl:.2f} | DTR={avg_dtr:.3f}")

    return experiment_results


def main():
    """Main entry point for the AIME benchmark experiment."""
    parser = argparse.ArgumentParser(
        description="Closed-Loop Steering System — AIME Benchmark Evaluation"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to a specific AIME .jsonl file. "
             "If not set, lists available datasets and prompts selection.",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=None,
        help="Experiment modes to run (default: all).",
    )
    args = parser.parse_args()

    # ---- Determine dataset ----
    if args.dataset:
        dataset_path = args.dataset
    else:
        # List available datasets and let user choose
        available = list_aime_datasets(DATASET_DIR)
        if not available:
            print(f"❌ No .jsonl files found in {DATASET_DIR}")
            sys.exit(1)
        elif len(available) == 1:
            dataset_path = available[0]
        else:
            print("Available AIME datasets:")
            for i, f in enumerate(available):
                print(f"  [{i}] {os.path.basename(f)}")
            choice = input("Select dataset index: ").strip()
            dataset_path = available[int(choice)]

    dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
    print(f"\n📂 Loading dataset: {dataset_path}")
    dataset = load_aime_dataset(dataset_path)
    print(f"   Loaded {len(dataset)} problems from {dataset_name}")

    # ---- Load model ----
    print(f"\n🔧 Loading model from {MODEL_PATH}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ---- Load control vector ----
    control_vector = load_control_vector(
        VECTOR_DIR,
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype=torch.bfloat16,
    )
    if control_vector is None:
        print("⚠️  No control vector loaded. "
              "Continuous and Dynamic modes will have no effect.")

    # ---- DTR calculator ----
    dtr_calc = DTRCalculator(model)

    # ---- Run experiments ----
    modes = args.modes if args.modes else None
    experiment_results = run_full_experiment(
        model, tokenizer, control_vector, dtr_calc,
        dataset=dataset,
        dataset_name=dataset_name,
        modes=modes,
    )

    # ---- Save results ----
    results_subdir = os.path.join(RESULTS_DIR, dataset_name)
    os.makedirs(results_subdir, exist_ok=True)

    # Serialize results
    serializable_results = {}
    for mode, data in experiment_results.items():
        serializable_results[mode] = {
            k: v for k, v in data.items()
            if isinstance(v, (int, float, str, list, dict, bool, type(None)))
        }

    results_path = os.path.join(results_subdir, "experiment_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)
    print(f"\n📊 Raw results saved to {results_path}")

    # ---- Generate visualization ----
    visualizer = PlotVisualizer(save_dir=results_subdir)
    visualizer.generate_comprehensive_report(experiment_results)

    # ---- Print final report ----
    print(f"\n{'='*60}")
    print(f"  AIME BENCHMARK REPORT — {dataset_name}")
    print(f"{'='*60}")
    for mode, data in experiment_results.items():
        acc = data["accuracy"]
        n_correct = data["correct_count"]
        n_total = data["total_count"]
        print(f"  {mode:<22} Pass@1: {acc:.2%} ({n_correct}/{n_total})")
    print(f"{'='*60}")

    print("\n🎉 Experiment completed successfully!")


if __name__ == "__main__":
    main()
