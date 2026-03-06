"""
Closed-Loop Steering System — Unified Experiment Runner
==========================================================
Replaces the old tag-based `run_dtr_experiments.py`.

Runs three experiment modes:
  1. Baseline: No intervention at all.
  2. Continuous: Fixed-strength spherical rotation at every decoding step.
  3. Dynamic_Spherical: TECA-driven PID → spherical rotation (our method).

Pipeline per experiment:
  Load model → Prepare prompts → Run generation with hooks →
  Collect metrics (DTR, PPL, Repetition, Accuracy, Trajectories) →
  Generate visualizations.
"""

import os
import json
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


# ======================== Evaluation Prompts ========================
# Logic puzzles that require genuine reasoning.
EVAL_PROMPTS = [
    {
        "question": (
            "In the fictional 'Xylophone' language, 'Apple' is classified as a verb "
            "because it implies the action of falling. Based on this rule, 'Banana' "
            "(which also falls) must also be a verb. Confirm this classification."
        ),
        "expected_keywords": ["verb", "falling", "action"],
        "accept_answer": lambda text: "verb" in text.lower(),
    },
    {
        "question": (
            "A farmer has 17 sheep. All but 9 die. How many sheep are left alive?"
        ),
        "expected_keywords": ["9"],
        "accept_answer": lambda text: "9" in text,
    },
    {
        "question": (
            "If it takes 5 machines 5 minutes to make 5 widgets, "
            "how long would it take 100 machines to make 100 widgets?"
        ),
        "expected_keywords": ["5 minutes", "5"],
        "accept_answer": lambda text: "5" in text and "minute" in text.lower(),
    },
    {
        "question": (
            "Three people check into a hotel room that costs $30. They each "
            "contribute $10. Later, the manager realizes the room costs only $25 "
            "and gives $5 to the bellboy to return. The bellboy keeps $2 and gives "
            "back $1 to each person. Now each person paid $9 (total $27), the "
            "bellboy has $2. $27 + $2 = $29. Where did the missing $1 go?"
        ),
        "expected_keywords": ["no missing", "fallacy", "error", "misleading"],
        "accept_answer": lambda text: any(
            w in text.lower()
            for w in ["no missing", "fallacy", "error", "misleading", "incorrect", "trick"]
        ),
    },
    {
        "question": (
            "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than "
            "the ball. How much does the ball cost?"
        ),
        "expected_keywords": ["$0.05", "5 cents", "0.05"],
        "accept_answer": lambda text: any(
            w in text for w in ["0.05", "5 cents", "$0.05"]
        ),
    },
]


def build_prompt(question: str) -> str:
    """Construct the full chat prompt for a question."""
    return (
        f"<|im_start|>system\n"
        f"You are a logical reasoning assistant. Solve the following puzzle step by step.\n"
        f"Use <solver> tags for your deduction and <critic> tags for self-verification.\n"
        f"Provide a rigorous logical derivation before giving your final answer.<|im_end|>\n"
        f"<|im_start|>user\n{question}<|im_end|>\n"
        f"<|im_start|>assistant\n<think>\n"
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
            max_new_tokens=MAX_NEW_TOKENS,
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


def evaluate_accuracy(gen_text: str, eval_item: dict) -> bool:
    """Check if generated text contains the expected answer."""
    return eval_item["accept_answer"](gen_text)


def run_full_experiment(
    model,
    tokenizer,
    control_vector: torch.Tensor | None,
    dtr_calc: DTRCalculator,
    modes: list[str] | None = None,
):
    """
    Run the full experiment pipeline across all modes and prompts.

    Returns experiment_results dict suitable for PlotVisualizer.
    """
    if modes is None:
        modes = EXPERIMENT_MODES

    experiment_results = {}

    for mode in modes:
        print(f"\n{'='*60}")
        print(f"  EXPERIMENT MODE: {mode}")
        print(f"{'='*60}")

        mode_correct = 0
        mode_total = 0
        mode_tokens_total = 0
        mode_repetitions = []
        mode_ppls = []
        mode_local_dtrs = []

        # Use trajectories from the first prompt for visualization
        first_teca_traj = []
        first_alpha_traj = []

        for i, eval_item in enumerate(EVAL_PROMPTS):
            prompt = build_prompt(eval_item["question"])
            print(f"\n  [{mode}] Prompt {i+1}/{len(EVAL_PROMPTS)}: "
                  f"{eval_item['question'][:60]}...")

            # Run generation
            result = run_single_generation(
                model, tokenizer, prompt, mode, control_vector
            )

            # Check accuracy
            is_correct = evaluate_accuracy(result["text"], eval_item)
            mode_correct += int(is_correct)
            mode_total += 1
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

            # Local DTR (within intervention window if available)
            try:
                if result["intervention_start"] and result["intervention_end"]:
                    w_start = result["input_len"] + result["intervention_start"]
                    w_end = result["input_len"] + result["intervention_end"]
                    local_dtr = dtr_calc.calculate_local_dtr(
                        result["output_ids"], w_start, w_end
                    )
                else:
                    # Use full generated sequence
                    dtr_scores, _ = dtr_calc.calculate(result["output_ids"])
                    local_dtr = dtr_scores[0]
                mode_local_dtrs.append(local_dtr)
            except Exception as e:
                print(f"    ⚠️  DTR calculation failed: {e}")
                mode_local_dtrs.append(0.0)

            # Save first prompt's trajectory
            if i == 0:
                first_teca_traj = result["teca_trajectory"]
                first_alpha_traj = result["alpha_trajectory"]

            status = "✅" if is_correct else "❌"
            print(f"    {status} Correct={is_correct} | "
                  f"Tokens={result['num_tokens']} | Rep={rep:.3f}")

        # Aggregate mode results
        accuracy = mode_correct / mode_total if mode_total > 0 else 0
        avg_rep = np.mean(mode_repetitions) if mode_repetitions else 0
        avg_ppl = np.nanmean(mode_ppls) if mode_ppls else float("nan")
        avg_tokens = mode_tokens_total // max(mode_total, 1)
        avg_dtr = np.mean(mode_local_dtrs) if mode_local_dtrs else 0

        experiment_results[mode] = {
            "accuracy": accuracy,
            "repetition": avg_rep,
            "ppl": avg_ppl,
            "tokens": avg_tokens,
            "local_dtr": avg_dtr,
            "teca_trajectory": first_teca_traj,
            "alpha_trajectory": first_alpha_traj,
        }

        print(f"\n  [{mode}] SUMMARY: Acc={accuracy:.2f} | "
              f"AvgTokens={avg_tokens} | Rep={avg_rep:.3f} | "
              f"PPL={avg_ppl:.2f} | DTR={avg_dtr:.3f}")

    return experiment_results


def main():
    """Main entry point for the experiment."""
    print(f"Loading model from {MODEL_PATH}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    # Load control vector
    control_vector = load_control_vector(
        VECTOR_DIR,
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype=torch.bfloat16,
    )

    if control_vector is None:
        print("⚠️  No control vector loaded. Continuous and Dynamic modes will have no effect.")

    # DTR calculator
    dtr_calc = DTRCalculator(model)

    # Run experiments
    experiment_results = run_full_experiment(
        model, tokenizer, control_vector, dtr_calc
    )

    # Save raw results (without lambdas)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Serialize results (remove non-serializable fields)
    serializable_results = {}
    for mode, data in experiment_results.items():
        serializable_results[mode] = {
            k: v for k, v in data.items()
            if isinstance(v, (int, float, str, list, dict, type(None)))
        }

    results_path = os.path.join(RESULTS_DIR, "experiment_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)
    print(f"\n📊 Raw results saved to {results_path}")

    # Generate visualization
    visualizer = PlotVisualizer(save_dir=RESULTS_DIR)
    visualizer.generate_comprehensive_report(experiment_results)

    print("\n🎉 All experiments completed successfully!")


if __name__ == "__main__":
    main()
