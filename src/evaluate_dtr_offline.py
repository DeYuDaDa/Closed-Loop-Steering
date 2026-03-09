"""
evaluate_dtr_offline.py

Standalone script to calculate DTR (Deep-Thinking Ratio) and PPL on the 
already generated output IDs from `run_experiment.py`. By completely decoupling
this from the generation phase, we prevent CUDA allocator fragmentation and
guarantee 100% stable GPU utilization during inference.
"""

import sys
import gc
import json
import argparse
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import MODEL_PATH, LAYER_ID, VECTOR_DIR
from dtr_utils import DTRCalculator, calculate_ppl
from run_experiment import load_control_vector

# Allow massive segments to prevent DTR fragmentation
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,garbage_collection_threshold:0.7"
)


def load_json_results(path: str) -> dict:
    if not os.path.exists(path):
        print(f"❌ Result file not found: {path}")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json_results(data: dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Offline DTR & PPL Calculator for AIME Results"
    )
    parser.add_argument(
        "--results",
        type=str,
        required=True,
        help="Path to the JSON results file created by run_experiment.py",
    )
    args = parser.parse_args()

    results_data = load_json_results(args.results)

    # 1. Load Model
    print(f"🔧 Loading model from {MODEL_PATH}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Load Control Vector (for Intervention Replay!)
    control_vector = load_control_vector(
        VECTOR_DIR,
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype=torch.bfloat16,
    )

    dtr_calc = DTRCalculator(model)

    # 3. Process each mode
    for mode, mode_data in results_data.items():
        print(f"\n========================================================")
        print(f"  [{mode}] Processing DTR and PPL")
        print(f"========================================================")

        problems = mode_data.get("per_problem", [])
        if not problems:
            print(f"  ⚠️ No problems found in mode {mode}.")
            continue

        mode_local_dtrs = []
        mode_ppls = []

        for i, prob in enumerate(problems):
            # If the script stopped halfway, some lists might be shorter. Guard against it.
            if "output_ids" not in prob:
                print(f"  ⚠️ outputs missing for problem {i+1}, skipping...")
                mode_local_dtrs.append(float("nan"))
                mode_ppls.append(float("nan"))
                continue

            # Load plain Python list into GPU tensor
            output_ids_gpu = torch.tensor(
                [prob["output_ids"]], dtype=torch.long, device=model.device
            )

            input_len = prob["input_len"]
            full_text = prob.get("full_response", prob.get("text", ""))

            # Calculate PPL
            try:
                ppl = calculate_ppl(model, tokenizer, full_text)
            except BaseException:
                ppl = float("nan")

            # Intervention Replay Args
            alpha_traj = None
            if mode == "Continuous":
                # Continuous mode alpha was always 0.15 for generated tokens
                alpha_traj = [0.15] * (output_ids_gpu.shape[1] - input_len)
            elif mode == "Dynamic_Spherical":
                alpha_traj = prob.get("alpha_trajectory", [])

            replay_args = {
                "control_vector": control_vector if mode in ("Continuous", "Dynamic_Spherical") else None,
                "alpha_trajectory": alpha_traj,
                "input_len": input_len,
                "layer_id": LAYER_ID,
            }

            w_start = prob.get("intervention_start")
            w_end = prob.get("intervention_end")

            try:
                if w_start is not None and w_end is not None:
                    abs_w_start = input_len + w_start
                    abs_w_end = input_len + w_end
                    if abs_w_start >= abs_w_end:
                        dtr_scores, _ = dtr_calc.calculate(output_ids_gpu, **replay_args)
                        local_dtr = dtr_scores[0]
                    else:
                        local_dtr = dtr_calc.calculate_local_dtr(
                            output_ids_gpu, abs_w_start, abs_w_end, **replay_args
                        )
                else:
                    dtr_scores, _ = dtr_calc.calculate(output_ids_gpu, **replay_args)
                    local_dtr = dtr_scores[0]
                
                print(f"    Prob {i+1:02d}: DTR = {local_dtr:.4f}  |  PPL = {ppl:.2f}")

            except Exception as e:
                print(f"    ⚠️ DTR error for Prob {i+1}: {e}")
                local_dtr = float("nan")

            mode_local_dtrs.append(local_dtr)
            mode_ppls.append(ppl)

            # Strict memory flush!
            del output_ids_gpu
            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()

        # Update mode aggregated statistics
        import numpy as np
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            avg_dtr = float(np.nanmean(mode_local_dtrs)) if mode_local_dtrs else float("nan")
            avg_ppl = float(np.nanmean(mode_ppls)) if mode_ppls else float("nan")

        mode_data["local_dtr"] = avg_dtr
        mode_data["ppl"] = avg_ppl

        for j, prob in enumerate(problems):
            prob["local_dtr"] = mode_local_dtrs[j]
            prob["ppl"] = mode_ppls[j]

        print(f"\n  ✅ {mode} Updated: Avg DTR = {avg_dtr:.4f} | Avg PPL = {avg_ppl:.2f}")

    # 4. Save and Overwrite JSON
    print(f"\n💾 Overwriting {args.results} with offline computed scores...")
    save_json_results(results_data, args.results)
    
    # Optional: trigger evaluation visualizer
    print(f"📊 Generating final graphs...")
    from evaluation_visualizer import PlotVisualizer
    viz = PlotVisualizer()
    viz.generate_comprehensive_report(results_data)


if __name__ == "__main__":
    main()
