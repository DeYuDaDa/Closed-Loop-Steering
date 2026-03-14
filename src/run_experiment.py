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
import config
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    config.PYTORCH_CUDA_ALLOC_CONF
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
    DEFAULT_DTYPE,
    DEVICE_MAP,
    DO_SAMPLE,
    ENDOFTEXT_ID,
    SAFE_SCORE_RANGE,
    CONTINUOUS_ALPHA,
    CAPTURE_HIDDEN_STATES,
    RESULTS_TIMESTAMP_FMT,
    JSON_INDENT,
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
from loaders.aime_loader import (
    load_aime_dataset,
    list_aime_datasets,
    build_aime_prompt,
    extract_answer as extract_answer_aime,
    check_answer as check_answer_aime,
    collate_prompts as collate_prompts_aime,
)
from loaders.math500_loader import (
    load_math500_dataset,
    build_math500_prompt,
    extract_answer_math500,
    check_answer_math500,
)
from loaders.zebra_logic_loader import (
    load_zebra_dataset,
    build_zebra_prompt,
    extract_answer_zebra,
    check_answer_zebra,
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


# Single sequence generation used to be here, but has been removed
# because all modes (including Dynamic_Spherical) now use fully batched tensor math.


# ======================== Batched Generation ========================
# Used for Baseline and Continuous modes

def run_batched_generation(
    model,
    tokenizer,
    prompts: list[list[dict]],
    mode: str,
    control_vector: torch.Tensor | None,
    batch_size: int = BATCH_SIZE,
) -> list[dict]:
    """
    Run batched generation for ALL modes.
    
    Batched state handling and tensor operations natively support Dynamic_Spherical
    speedup without cross-sequence pollution.

    Args:
        model: The loaded causal LM.
        tokenizer: The tokenizer.
        prompts: List of prompt message dict lists.
        mode: "Baseline", "Continuous", or "Dynamic_Spherical".
        control_vector: The steering vector (or None).
        batch_size: Number of sequences per batch.

    Returns:
        List of result dicts.
    """
    all_results = []
    
    # Resolve </think> token ID for ThinkBrake
    term_token_id = None
    try:
        term_ids = tokenizer.encode("</think>", add_special_tokens=False)
        if term_ids:
            term_token_id = term_ids[-1]
    except Exception:
        pass

    for batch_start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[batch_start:batch_start + batch_size]
        actual_bs = len(batch_prompts)

        # Apply chat template securely to get IDs without text-splitting issues.
        # NOTE: We call apply_chat_template once per message list (not on the whole batch)
        # because some transformers versions return a flat list[int] for batch inputs,
        # causing totally wrong tensor construction.
        think_ids = tokenizer.encode("<think>\n", add_special_tokens=False)
        
        encoded_prompts = []
        for messages in batch_prompts:
            ids = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
            )
            # Manually append <think>\n to force chain-of-thought prefill
            encoded_prompts.append(ids + think_ids)

        # Tokenize with left-padding to keep alignments simple
        tokenizer.padding_side = 'left'
        
        # VERY IMPORTANT: The pad_token_id in input_ids must be a valid vocabulary index,
        # otherwise CUDA embedding lookup (ScatterGather) will cause an OutOfBounds device assert.
        # We use eos_token_id if pad_token_id is None, but ensure it's a valid integer > 0.
        valid_pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
        if tokenizer.pad_token_id is not None and tokenizer.pad_token_id >= 0:
            valid_pad_id = tokenizer.pad_token_id
            
        max_len = max(len(ids) for ids in encoded_prompts)
        
        padded_ids = []
        attention_mask = []
        
        for ids in encoded_prompts:
            pad_len = max_len - len(ids)
            padded_ids.append([valid_pad_id] * pad_len + ids)
            attention_mask.append([0] * pad_len + [1] * len(ids))
            
        inputs = {
            "input_ids": torch.tensor(padded_ids, dtype=torch.long, device=model.device),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long, device=model.device)
        }
        
        # Reset to right padding just in case it's assumed elsewhere
        tokenizer.padding_side = 'right'

        # Initialize batched state
        state = InjectionState(batch_size=actual_bs, device=model.device)
        pid = None
        processors = LogitsProcessorList()

        # Protection against fp16/bf16 left-padding NaN generation bugs (FlashAttention)
        class InfNanProtectionProcessor:
            def __call__(self, input_ids, scores):
                if torch.isnan(scores).any() or torch.isinf(scores).any():
                    scores = torch.nan_to_num(scores, nan=-SAFE_SCORE_RANGE, posinf=SAFE_SCORE_RANGE, neginf=-SAFE_SCORE_RANGE)
                return scores

        processors.append(InfNanProtectionProcessor())
        
        if mode == "Continuous":
            state.intervention_active.fill_(True)
            state.alpha.fill_(CONTINUOUS_ALPHA)  # Use specialized continuous alpha
        elif mode in ("Dynamic_Spherical",):
            # PID controller mapped to batch size
            pid = PIDController(batch_size=actual_bs, device=model.device)
            
        # Calculate actual input lengths per sequence in batch
        input_lens = inputs["attention_mask"].sum(dim=1).tolist()

        if mode != "Baseline":
            monitor = StateMonitor(
                state=state,
                pid_controller=pid,
                term_token_id=term_token_id,
            )
            
            # Since generation doesn't expose sequence completion easily, we add a
            # quick custom logits processor that examines input_ids to update the active_mask
            class ActiveMaskProcessor:
                def __init__(self, state, tokenizer, input_lens):
                    self.state = state
                    self.eos_id = tokenizer.eos_token_id
                    self.input_lens = input_lens
                    
                def __call__(self, input_ids, scores):
                    if self.eos_id is not None:
                        for i in range(self.state.batch_size):
                            gen_part = input_ids[i, self.input_lens[i]:]
                            has_eos = (gen_part == self.eos_id).any()
                            self.state.active_mask[i] = ~has_eos
                    return scores
            
            processors.append(ActiveMaskProcessor(state, tokenizer, input_lens))
            processors.append(monitor)

        # Steering hook 
        history_hidden = []
        if control_vector is not None and mode in ("Continuous", "Dynamic_Spherical"):
            hook_fn, history_hidden = create_steering_hook(
                state=state,
                control_vector=control_vector,
                mode=mode,
                continuous_alpha=CONTINUOUS_ALPHA,
                capture_hidden_states=CAPTURE_HIDDEN_STATES, # We use offline DTR script now
            )
            layer = model.model.layers[LAYER_ID]
            handle = layer.register_forward_hook(hook_fn)
        else:
            handle = None

        # Generate
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=AIME_MAX_TOKENS,
                do_sample=DO_SAMPLE,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                pad_token_id=tokenizer.pad_token_id,
                logits_processor=processors,
            )

        if handle is not None:
            handle.remove()

        # Extract per-sequence results
        for i in range(actual_bs):
            # Find where the actual input ends (skip padding tokens)
            input_mask = inputs["attention_mask"][i]
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

            # Extract specific trajectories for this problem
            teca_traj = state.teca_trajectory[i] if mode != "Baseline" else []
            alpha_traj = state.alpha_trajectory[i] if mode != "Baseline" else []
            entropy_traj = state.entropy_trajectory[i] if mode != "Baseline" else []
            inv_start = state.intervention_start_step[i] if mode != "Baseline" else None
            inv_end = state.intervention_end_step[i] if mode != "Baseline" else None
            conv = state.converged[i].item() if mode != "Baseline" else False

            # Convert to plain Python list to sever CUDA references
            all_results.append({
                "text": gen_text,
                "tokens": tokens,
                "num_tokens": len(tokens),
                "output_ids": output_ids[i].cpu().tolist(),  # plain list[int]
                "input_len": input_len,
                "teca_trajectory": teca_traj,
                "alpha_trajectory": alpha_traj,
                "entropy_trajectory": entropy_traj,
                "history_hidden": [],
                "intervention_start": inv_start,
                "intervention_end": inv_end,
                "convergence": conv,
            })

        # Free GPU memory after extracting results from this batch
        del output_ids
        del inputs
        del state
        if pid is not None:
            del pid
        if handle is not None:
            del hook_fn
        del history_hidden
        del processors
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
    dataset_type: str = "aime",
):
    """
    Run the full AIME benchmark across all modes.

    Uses batched generation for Baseline/Continuous and sequential
    generation for Dynamic_Spherical.

    Args:
        model: Loaded causal LM.
        tokenizer: Tokenizer.
        control_vector: Steering vector (or None).
        dataset: List of AIME problem dicts.
        dataset_name: Human-readable label for the dataset.
        modes: Experiment modes to run (defaults to config).
        batch_size: Number of sequences per batch for batched modes.

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

        # Build all prompts based on dataset type
        if dataset_type == "math500":
            prompts = [build_math500_prompt(p["problem"]) for p in dataset]
        elif dataset_type == "zebralogic":
            prompts = [build_zebra_prompt(p["puzzle"], p["question"]) for p in dataset]
        else:
            prompts = collate_prompts_aime(dataset)

        # ---- Run generation ----
        print(f"  Using BATCHED generation (batch_size={batch_size}) for ALL modes...")
        results_list = run_batched_generation(
            model, tokenizer, prompts, mode, control_vector,
            batch_size=batch_size
        )

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
            # Extract and check answer based on dataset type
            if dataset_type == "math500":
                predicted = extract_answer_math500(result["text"])
                expected = eval_item["answer"]
                is_correct = check_answer_math500(predicted, expected)
            elif dataset_type == "zebralogic":
                predicted = extract_answer_zebra(result["text"])
                expected = eval_item["answer"]
                is_correct = check_answer_zebra(predicted, expected)
            else:
                predicted = extract_answer_aime(result["text"])
                expected = eval_item["answer"]
                is_correct = check_answer_aime(predicted, expected)

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
                "output_ids": result["output_ids"],
                "input_len": result["input_len"],
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
                def _val(x):
                    return x.item() if hasattr(x, "item") else x

                detail["alpha_active_steps"] = sum(
                    1 for a in alpha_traj if _val(a) > 0
                )
                
                # Handle max safely for empty lists or lists of tensors
                max_val = 0.0
                if alpha_traj:
                    max_val = max(_val(a) for a in alpha_traj)
                detail["alpha_max_value"] = max_val
                
                # Handle mean safely
                mean_val = 0.0
                if alpha_traj:
                    mean_val = sum(_val(a) for a in alpha_traj) / len(alpha_traj)
                detail["alpha_mean_value"] = mean_val

            per_problem_details.append(detail)

        # ---- Aggregate mode results ----
        accuracy = mode_correct / mode_total if mode_total > 0 else 0
        avg_rep = np.mean(mode_repetitions) if mode_repetitions else 0
        avg_tokens = mode_tokens_total // max(mode_total, 1)

        mode_result = {
            "accuracy": accuracy,
            "correct_count": mode_correct,
            "total_count": mode_total,
            "repetition": avg_rep,
            "ppl": float("nan"), # Will be calculated offline
            "tokens": avg_tokens,
            "local_dtr": float("nan"), # Will be calculated offline
            "teca_trajectory": first_teca_traj,
            "alpha_trajectory": first_alpha_traj,
            "per_problem": per_problem_details,
        }

        # Module-level diagnostics summary (Dynamic_Spherical only)
        if mode == "Dynamic_Spherical":
            active_steps_list = [
                d.get("alpha_active_steps", 0) for d in per_problem_details if isinstance(d, dict)
            ]
            total_steps_list = [
                d.get("num_tokens", 1) for d in per_problem_details if isinstance(d, dict)
            ]
            max_alpha_list = [
                d.get("alpha_max_value", 0.0) for d in per_problem_details if isinstance(d, dict)
            ]
            mode_result["module_diagnostics"] = {
                "problems_with_intervention": sum(
                    1 for s in active_steps_list if isinstance(s, (int, float)) and s > 0
                ),
                "problems_with_convergence": sum(
                    1 for d in per_problem_details
                    if isinstance(d, dict) and d.get("convergence", False)
                ),
                "mean_alpha_active_ratio": float(np.mean([
                    (a / max(t, 1)) if isinstance(a, int) and isinstance(t, int) else 0.0
                    for a, t in zip(active_steps_list, total_steps_list)
                ])),
                "mean_max_alpha": float(np.mean(max_alpha_list) if isinstance(max_alpha_list, list) and len(max_alpha_list) > 0 else 0.0),
            }
            # Print diagnostic summary
            diag = mode_result["module_diagnostics"]
            p_interv = diag.get("problems_with_intervention", 0)
            a_ratio = diag.get("mean_alpha_active_ratio", 0.0)
            m_alpha = diag.get("mean_max_alpha", 0.0)
            p_conv = diag.get("problems_with_convergence", 0)
            print(f"\n  📊 [{mode}] MODULE DIAGNOSTICS:")
            print(f"    StateMonitor (TECA):     "
                  f"All {mode_total} problems have TECA trajectories")
            print(f"    PID Controller:          "
                  f"{p_interv}/{mode_total} "
                  f"problems triggered intervention (α>0)")
            print(f"    Spherical Steering:      "
                  f"Mean active ratio = "
                  f"{a_ratio:.2%}, "
                  f"Mean max α = {m_alpha:.4f}")
            print(f"    ThinkBrake Convergence:  "
                  f"{p_conv}/{mode_total} "
                  f"problems reached convergence")

        experiment_results[mode] = mode_result

        print(f"\n  [{mode}] SUMMARY: "
              f"Pass@1={accuracy:.2%} ({mode_correct}/{mode_total}) | "
              f"AvgTokens={avg_tokens} | Rep={avg_rep:.3f} | "
              f"PPL=N/A | DTR=N/A")

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
    dataset_path_lower = dataset_path.lower()
    
    if "math500" in dataset_path_lower:
        dataset_type = "math500"
        print(f"\n📂 Loading MATH500 dataset: {dataset_path}")
        dataset = load_math500_dataset(dataset_path)
    elif "zebralogic" in dataset_path_lower:
        dataset_type = "zebralogic"
        print(f"\n📂 Loading ZebraLogic dataset: {dataset_path}")
        dataset = load_zebra_dataset(dataset_path)
    else:
        dataset_type = "aime"
        print(f"\n📂 Loading AIME dataset: {dataset_path}")
        dataset = load_aime_dataset(dataset_path)
        
    print(f"   Loaded {len(dataset)} problems from {dataset_name} ({dataset_type})")

    model_dtype = getattr(torch, DEFAULT_DTYPE)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=model_dtype,
        device_map=DEVICE_MAP,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    # Ensure pad token is set and DIFFERS from eos_token.
    # Qwen3 eos = im_end (151645); native pad = endoftext (151643).
    # If pad == eos, batched generate() cannot stop at EOS.
    if tokenizer.pad_token_id is None or tokenizer.pad_token_id == tokenizer.eos_token_id:
        tokenizer.pad_token_id = ENDOFTEXT_ID
        tokenizer.pad_token = tokenizer.convert_ids_to_tokens(ENDOFTEXT_ID)

    # ---- Load control vector ----
    control_vector = load_control_vector(
        VECTOR_DIR,
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype=model_dtype,
    )
    if control_vector is None:
        print("⚠️  No control vector loaded. "
              "Continuous and Dynamic modes will have no effect.")

    # ---- Run experiments ----
    modes = args.modes if args.modes else None
    experiment_results = run_full_experiment(
    model, tokenizer,
    dataset=dataset,
    dataset_name=dataset_name,
    modes=modes,
    control_vector=control_vector,
    dataset_type=dataset_type,
)

    # ---- Save results ----
    timestamp = datetime.now().strftime(RESULTS_TIMESTAMP_FMT)
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
        json.dump(serializable_results, f, indent=JSON_INDENT, ensure_ascii=False)
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
