import json

new_code = r'''# ======================== Full Experiment Pipeline ========================

def run_full_experiment(
    model,
    tokenizer,
    dataset: list[dict],
    dataset_name: str = "AIME",
    modes: list[str] | None = None,
    control_vector = None,
    batch_size: int = 1,
    dataset_type: str = "aime",
    results_path: str = None,
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
        dataset_type: Type of dataset ("aime", "math500", "zebralogic").
        results_path: Path to save the incremental results JSON.

    Returns:
        experiment_results dict with per-mode metrics and per-problem details.
    """
    if modes is None:
        try:
            from config import EXPERIMENT_MODES
            modes = EXPERIMENT_MODES
        except ImportError:
            modes = ["Baseline"]

    experiment_results = {}
    results_queue = queue.Queue()

    # Async worker to perform answer extraction, evaluate and save to JSON
    def result_saver_worker():
        while True:
            item = results_queue.get()
            if item is None:
                results_queue.task_done()
                break
                
            mode_name, batch_dataset, batch_results, mode_stats, mode_data = item
            
            for i, (eval_item, result) in enumerate(zip(batch_dataset, batch_results)):
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
                    # using fallback in standard run_experiment if dataset lacks check
                    is_correct = check_answer_aime(predicted, expected)

                mode_stats["mode_correct"] += int(is_correct)
                mode_stats["mode_tokens_total"] += result["num_tokens"]

                rep = calculate_repetition_rate(result["text"])
                mode_stats["mode_repetitions"].append(rep)

                status = "✅" if is_correct else "❌"
                tqdm.write(f"  {status} Problem {eval_item['id']}: "
                           f"pred={predicted} expected={expected} | "
                           f"Tokens={result['num_tokens']} | Rep={rep:.3f}")

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

                if mode_name == "Dynamic_Spherical":
                    alpha_traj = result["alpha_trajectory"]
                    detail["ema_trajectory"] = result["ema_trajectory"]
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
                    
                    max_val = 0.0
                    if alpha_traj:
                        max_val = max(_val(a) for a in alpha_traj)
                    detail["alpha_max_value"] = max_val
                    
                    mean_val = 0.0
                    if alpha_traj:
                        mean_val = sum(_val(a) for a in alpha_traj) / len(alpha_traj)
                    detail["alpha_mean_value"] = mean_val

                mode_data["per_problem"].append(detail)

                # Save first problem's trajectory for visualization
                if len(mode_data["per_problem"]) == 1:
                    mode_data["ema_trajectory"] = result["ema_trajectory"]
                    mode_data["alpha_trajectory"] = result["alpha_trajectory"]

            # Serialize results
            serializable_results = {}
            for m, data in experiment_results.items():
                serializable_results[m] = {
                    k: v for k, v in data.items()
                    if isinstance(v, (int, float, str, list, dict, bool, type(None)))
                }
            try:
                from config import JSON_INDENT
            except ImportError:
                JSON_INDENT = 4
                
            if results_path is not None:
                with open(results_path, "w", encoding="utf-8") as f:
                    json.dump(serializable_results, f, indent=JSON_INDENT, ensure_ascii=False)
            
            results_queue.task_done()

    # Start the worker thread
    worker_thread = threading.Thread(target=result_saver_worker, daemon=True)
    worker_thread.start()

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

        print(f"  Using BATCHED generation (batch_size={batch_size}) for ALL modes...")
        
        mode_data = {
            "accuracy": 0.0,
            "correct_count": 0,
            "total_count": len(dataset),
            "repetition": 0.0,
            "ppl": float("nan"),
            "tokens": 0,
            "local_dtr": float("nan"),
            "ema_trajectory": [],
            "alpha_trajectory": [],
            "per_problem": [],
        }
        experiment_results[mode] = mode_data
        
        mode_stats = {
            "mode_correct": 0,
            "mode_tokens_total": 0,
            "mode_repetitions": []
        }

        # Initialize progress bar
        pbar = tqdm(total=len(dataset), desc=f"Evaluating {mode}", unit="sample")

        # Create generator
        gen_iterator = run_batched_generation(
            model, tokenizer, prompts, mode, control_vector,
            batch_size=batch_size
        )

        batch_start = 0
        for batch_results in gen_iterator:
            batch_end = batch_start + len(batch_results)
            batch_dataset = dataset[batch_start:batch_end]
            
            # Submits to queue to be processed asynchronously
            results_queue.put((mode, batch_dataset, batch_results, mode_stats, mode_data))
            
            pbar.update(len(batch_results))
            batch_start = batch_end
            
        pbar.close()

        # Wait for all background saving logic for this mode to complete before finalizing
        results_queue.join()

        # ---- Aggregate mode results ----
        mode_total = len(dataset)
        import numpy as np
        accuracy = mode_stats["mode_correct"] / mode_total if mode_total > 0 else 0
        avg_rep = np.mean(mode_stats["mode_repetitions"]) if mode_stats["mode_repetitions"] else 0
        avg_tokens = mode_stats["mode_tokens_total"] // max(mode_total, 1)

        mode_data["accuracy"] = float(accuracy)
        mode_data["correct_count"] = mode_stats["mode_correct"]
        mode_data["repetition"] = float(avg_rep)
        mode_data["tokens"] = int(avg_tokens) 

        per_problem_details = mode_data["per_problem"]

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
            mode_data["module_diagnostics"] = {
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
                ]) if total_steps_list else 0.0),
                "mean_max_alpha": float(np.mean(max_alpha_list) if isinstance(max_alpha_list, list) and len(max_alpha_list) > 0 else 0.0),
            }
            # Print diagnostic summary
            diag = mode_data["module_diagnostics"]
            p_interv = diag.get("problems_with_intervention", 0)
            a_ratio = diag.get("mean_alpha_active_ratio", 0.0)
            m_alpha = diag.get("mean_max_alpha", 0.0)
            p_conv = diag.get("problems_with_convergence", 0)
            print(f"\n  📊 [{mode}] MODULE DIAGNOSTICS:")
            print(f"    StateMonitor (EMA):      "
                  f"All {mode_total} problems have EMA trajectories")
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

        print(f"\n  [{mode}] SUMMARY: "
              f"Pass@1={accuracy:.2%} ({mode_stats['mode_correct']}/{mode_total}) | "
              f"AvgTokens={avg_tokens} | Rep={avg_rep:.3f} | "
              f"PPL=N/A | DTR=N/A")

        # Give it a final save with updated metrics
        serializable_results = {}
        for m, data in experiment_results.items():
            serializable_results[m] = {
                k: v for k, v in data.items()
                if isinstance(v, (int, float, str, list, dict, bool, type(None)))
            }
        try:
            from config import JSON_INDENT
        except ImportError:
            JSON_INDENT = 4
        if results_path is not None:
            with open(results_path, "w", encoding="utf-8") as f:
                json.dump(serializable_results, f, indent=JSON_INDENT, ensure_ascii=False)

    # Clean up worker thread permanently at the very end
    results_queue.put(None)
    worker_thread.join()

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

    try:
        from config import (
            DATASET_DIR, DEFAULT_DTYPE, MODEL_PATH, DEVICE_MAP, ENDOFTEXT_ID,
            VECTOR_DIR, RESULTS_TIMESTAMP_FMT, RESULTS_DIR, BATCH_SIZE
        )
    except ImportError:
        pass

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

    # ---- Save results setup ----
    timestamp = datetime.now().strftime(RESULTS_TIMESTAMP_FMT)
    results_subdir = os.path.join(RESULTS_DIR, f"{dataset_name}_{timestamp}")
    os.makedirs(results_subdir, exist_ok=True)
    results_path = os.path.join(results_subdir, "experiment_results.json")

    # ---- Run experiments ----
    modes = args.modes if args.modes else None
    experiment_results = run_full_experiment(
        model, tokenizer,
        dataset=dataset,
        dataset_name=dataset_name,
        modes=modes,
        control_vector=control_vector,
        batch_size=BATCH_SIZE,
        dataset_type=dataset_type,
        results_path=results_path,
    )

    print(f"\n📊 Final results saved to {results_path}")

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
'''

with open("f:/academic/Closed-Loop-Steering-System/src/run_experiment.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if line.startswith("# ======================== Full Experiment Pipeline ========================"):
        split_idx = i
        break

with open("f:/academic/Closed-Loop-Steering-System/src/run_experiment.py", "w", encoding="utf-8") as f:
    f.writelines(lines[:split_idx])
    f.write(new_code)
