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
import gc
import time
import json
import warnings
from datetime import datetime
import argparse

# ---- CUDA Allocator Anti-Fragmentation ----
# Must be set BEFORE importing torch. Expandable segments prevent the
# caching allocator from creating permanent fragmentation holes when
# tensors of varying sizes (KV caches, hidden states) are allocated
# and freed in a loop.
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True,garbage_collection_threshold:0.7"
)

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
    TEMPERATURE,
    TOP_P,
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
            # Log raw norm for coeff calibration
            raw_norm = injector.get_raw_norm()
            print(f"  📏 Vector raw norm (before normalization): {raw_norm:.4f}")

            v = injector.get_normalized_vector()  # shape [1, 1, d]
            # Normalize to unit vector
            v_flat = v.view(-1)
            v_normalized = v_flat / v_flat.norm()
            v_normalized = v_normalized.view(v.shape)
            print(f"  📏 Vector final norm (after normalization): {v_normalized.float().view(-1).norm().item():.4f}")
            print(f"  ℹ️  Steering uses unit-direction vector; "
                  f"effective coeff is controlled by alpha (PID output).")
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
    calc_dtr: bool = True, # New argument
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
            capture_hidden_states=calc_dtr, # Only capture if DTR is calculated
        )
    else:
        # Fallback: no-op hook (avoid storing huge history_hidden arrays to save RAM)
        history_hidden = []

        def hook_fn(module, args, output):
            return output

    # Register hook at target layer
    layer = model.model.layers[LAYER_ID]
    handle = layer.register_forward_hook(hook_fn)

    # ---- Generate ----
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=AIME_MAX_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
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

    # Move output_ids to plain Python to sever ALL PyTorch/CUDA references.
    # Using .tolist() instead of .cpu() ensures zero hidden CUDA event refs.
    output_ids_list = output_ids[0].cpu().tolist()  # list[int]
    # Explicitly break reference cycles to allow PyTorch to free the computation graph
    teca_traj = list(state.teca_trajectory)
    alpha_traj = list(state.alpha_trajectory)
    entropy_traj = list(state.entropy_trajectory)
    inv_start = state.intervention_start_step
    inv_end = state.intervention_end_step
    conv = state.converged
    
    del output_ids
    del generated_ids
    del inputs
    del state
    del monitor
    del processors
    del pid
    del hook_fn
    del history_hidden
    # Synchronize CUDA stream before freeing — ensures all async ops are done
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "text": gen_text,
        "tokens": tokens,
        "num_tokens": len(tokens),
        "output_ids": output_ids_list,  # plain Python list[int], no PyTorch refs
        "input_len": input_len,
        "teca_trajectory": teca_traj,
        "alpha_trajectory": alpha_traj,
        "entropy_trajectory": entropy_traj,
        "history_hidden": [],
        "intervention_start": inv_start,
        "intervention_end": inv_end,
        "convergence": conv,
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
    calc_dtr: bool = True, # New argument
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
        calc_dtr: Whether DTR will be calculated, influences history_hidden capture.

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
            state.intervention_active = True
            state.alpha = 0.15  # Fixed continuous alpha

        # Steering hook (same alpha for all sequences in batch)
        history_hidden = []
        if control_vector is not None and mode == "Continuous":
            hook_fn, history_hidden = create_steering_hook(
                state=state,
                control_vector=control_vector,
                mode=mode,
                continuous_alpha=0.15,
                capture_hidden_states=calc_dtr, # Only capture if DTR is calculated
            )
        else:
            def hook_fn(module, args, output):
                return output

        # Register hook
        layer = model.model.layers[LAYER_ID]
        handle = layer.register_forward_hook(hook_fn)

        # Generate
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=AIME_MAX_TOKENS,
                do_sample=True,
                temperature=TEMPERATURE,
                top_p=TOP_P,
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

            # Convert to plain Python list to sever CUDA references
            all_results.append({
                "text": gen_text,
                "tokens": tokens,
                "num_tokens": len(tokens),
                "output_ids": output_ids[i].cpu().tolist(),  # plain list[int]
                "input_len": input_len,
                "teca_trajectory": [],
                "alpha_trajectory": [],
                "entropy_trajectory": [],
                "history_hidden": [],
                "intervention_start": None,
                "intervention_end": None,
            })

        # Free GPU memory after extracting results from this batch
        del output_ids
        del inputs
        del state
        del hook_fn
        del history_hidden
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()

    return all_results


# ======================== Full Experiment Pipeline ========================

def run_full_experiment(
    model,
    tokenizer,
    dataset: list[dict],
    dataset_name: str = "AIME",
    modes: list[str] | None = None,
    control_vector: torch.Tensor | None = None,
    batch_size: int = BATCH_SIZE,
    calc_dtr: bool = True,
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
        batch_size: Number of sequences per batch for batched modes.
        calc_dtr: Whether DTR will be calculated.

    Returns:
        experiment_results dict with per-mode metrics and per-problem details.
    """
    if modes is None:
        modes = EXPERIMENT_MODES

    experiment_results = {}

    # Initialize DTR calculator once for all modes
    dtr_calc = DTRCalculator(model)

    for mode in modes:
        print(f"\n{'='*60}")
        print(f"  [{dataset_name}] EXPERIMENT MODE: {mode}")
        print(f"{'='*60}")

        # Build all prompts
        prompts = collate_prompts(dataset)

        # ---- Run generation ----
        if mode in ("Baseline", "Continuous"):
            print(f"  Using BATCHED generation (batch_size={batch_size})...")
            results_list = run_batched_generation(
                model, tokenizer, prompts, mode, control_vector,
                batch_size=batch_size, calc_dtr=calc_dtr
            )
        else:
            # Dynamic_Spherical: sequential
            print(f"  Using SEQUENTIAL generation (per-sequence PID)...")
            results_list = []
            for idx, prompt in enumerate(prompts):
                print(f"    [{mode}] Problem {idx+1}/{len(prompts)}: "
                      f"id={dataset[idx]['id']}...")
                result = run_single_generation(
                    model, tokenizer, prompt, mode, control_vector, calc_dtr=calc_dtr
                )
                results_list.append(result)
                # Aggressive memory cleanup between sequential problems
                torch.cuda.synchronize()
                gc.collect()
                torch.cuda.empty_cache()

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

            # PPL — skip during evaluation loop, will be calculated in deferred pass
            # (to not interleave forward passes with generation)
            mode_ppls.append(float("nan"))

            # Save first trajectory
            if i == 0:
                first_teca_traj = result["teca_trajectory"]
                first_alpha_traj = result["alpha_trajectory"]

            status = "✅" if is_correct else "❌"
            print(f"  {status} Problem {eval_item['id']}: "
                  f"pred={predicted} expected={expected} | "
                  f"Tokens={result['num_tokens']} | Rep={rep:.3f}")

            # Per-problem detail record
            detail = {
                "id": eval_item["id"],
                "expected": expected,
                "predicted": predicted,
                "correct": is_correct,
                "num_tokens": result["num_tokens"],
                "repetition": rep,
                "full_response": result["text"],
            }

            # Module diagnostics (Dynamic_Spherical: per-problem trajectories)
            if mode == "Dynamic_Spherical":
                alpha_traj = result["alpha_trajectory"]
                detail["teca_trajectory"] = result["teca_trajectory"]
                detail["alpha_trajectory"] = alpha_traj
                detail["entropy_trajectory"] = result["entropy_trajectory"]
                detail["intervention_start"] = result["intervention_start"]
                detail["intervention_end"] = result["intervention_end"]
                detail["convergence"] = result.get("convergence", None)
                detail["alpha_active_steps"] = sum(
                    1 for a in alpha_traj if a > 0
                )
                detail["alpha_max_value"] = (
                    max(alpha_traj) if alpha_traj else 0.0
                )
                detail["alpha_mean_value"] = (
                    (sum(alpha_traj) / len(alpha_traj))
                    if alpha_traj
                    else 0.0
                )

            per_problem_details.append(detail)

        # ---- Deferred DTR & PPL Calculation ----
        # Run ALL heavy forward passes AFTER generation is complete,
        # so GPU stays at full utilization during the generation phase.
        if calc_dtr:
            print(f"\n  📐 [{mode}] Calculating DTR (deferred, {len(results_list)} problems)...")
        for i, result in enumerate(results_list):
            # DTR
            if calc_dtr:
                try:
                    # Reconstruct output_ids as a GPU tensor from the plain Python list
                    output_ids_gpu = torch.tensor(
                        [result["output_ids"]], dtype=torch.long, device=model.device
                    )  # shape [1, seq_len]

                    # Construct replay trajectory
                    alpha_traj = None
                    if mode == "Continuous":
                        alpha_traj = [0.15] * (output_ids_gpu.shape[1] - result["input_len"])
                    elif mode == "Dynamic_Spherical":
                        alpha_traj = result["alpha_trajectory"]

                    replay_args = {
                        "control_vector": control_vector if mode in ("Continuous", "Dynamic_Spherical") else None,
                        "alpha_trajectory": alpha_traj,
                        "input_len": result["input_len"],
                        "layer_id": LAYER_ID
                    }

                    if result["intervention_start"] and result["intervention_end"]:
                        w_start = result["input_len"] + result["intervention_start"]
                        w_end = result["input_len"] + result["intervention_end"]
                        if w_start >= w_end:
                            dtr_scores, _ = dtr_calc.calculate(output_ids_gpu, **replay_args)
                            local_dtr = dtr_scores[0]
                        else:
                            local_dtr = dtr_calc.calculate_local_dtr(
                                output_ids_gpu, w_start, w_end, **replay_args
                            )
                    else:
                        dtr_scores, _ = dtr_calc.calculate(output_ids_gpu, **replay_args)
                        local_dtr = dtr_scores[0]
                    mode_local_dtrs.append(local_dtr)
                    print(f"    DTR[{i+1}/{len(results_list)}] = {local_dtr:.4f}")

                    del output_ids_gpu
                    torch.cuda.synchronize()
                    gc.collect()
                    torch.cuda.empty_cache()
                except Exception as e:
                    print(f"    ⚠️ DTR error [{i+1}]: {e}")
                    mode_local_dtrs.append(float("nan"))
            else:
                mode_local_dtrs.append(float("nan"))

            # PPL (also deferred)
            try:
                ppl = calculate_ppl(model, tokenizer, result["text"])
                mode_ppls[i] = ppl
            except Exception:
                pass  # already nan

        # ---- Aggregate mode results ----
        accuracy = mode_correct / mode_total if mode_total > 0 else 0
        avg_rep = np.mean(mode_repetitions) if mode_repetitions else 0
        avg_ppl = np.nanmean(mode_ppls) if mode_ppls else float("nan")
        avg_tokens = mode_tokens_total // max(mode_total, 1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            avg_dtr = float(np.nanmean(mode_local_dtrs)) if mode_local_dtrs else float("nan")

        mode_result = {
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

        # Module-level diagnostics summary (Dynamic_Spherical only)
        if mode == "Dynamic_Spherical":
            active_steps_list = [
                d.get("alpha_active_steps", 0) for d in per_problem_details
            ]
            total_steps_list = [
                d.get("num_tokens", 1) for d in per_problem_details
            ]
            max_alpha_list = [
                d.get("alpha_max_value", 0.0) for d in per_problem_details
            ]
            mode_result["module_diagnostics"] = {
                "problems_with_intervention": sum(
                    1 for s in active_steps_list if s > 0
                ),
                "problems_with_convergence": sum(
                    1 for d in per_problem_details
                    if d.get("convergence")
                ),
                "mean_alpha_active_ratio": float(np.mean([
                    a / max(t, 1)
                    for a, t in zip(active_steps_list, total_steps_list)
                ])),
                "mean_max_alpha": float(np.mean(max_alpha_list)),
            }
            # Print diagnostic summary
            diag = mode_result["module_diagnostics"]
            print(f"\n  📊 [{mode}] MODULE DIAGNOSTICS:")
            print(f"    StateMonitor (TECA):     "
                  f"All {mode_total} problems have TECA trajectories")
            print(f"    PID Controller:          "
                  f"{diag['problems_with_intervention']}/{mode_total} "
                  f"problems triggered intervention (α>0)")
            print(f"    Spherical Steering:      "
                  f"Mean active ratio = "
                  f"{diag['mean_alpha_active_ratio']:.2%}, "
                  f"Mean max α = {diag['mean_max_alpha']:.4f}")
            print(f"    ThinkBrake Convergence:  "
                  f"{diag['problems_with_convergence']}/{mode_total} "
                  f"problems reached convergence")

        experiment_results[mode] = mode_result

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
    parser.add_argument(
        "--no_calc_dtr", action="store_true",
        help="Disable deep-thinking ratio (DTR) calculation to save time and VRAM",
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
        model, tokenizer,
        dataset=dataset,
        dataset_name=dataset_name,
        modes=modes,
        control_vector=control_vector,
        calc_dtr=not args.no_calc_dtr,
    )

    # ---- Save results ----
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_subdir = os.path.join(RESULTS_DIR, f"{dataset_name}_{timestamp}")
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
