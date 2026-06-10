#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Parameter Sensitivity Analysis Sweep Script
===========================================
This script fits PCA with varying values of components (k) and evaluates 
the model's reasoning performance (accuracy, generated tokens, and repetition rate) 
on a validation dataset using the Dynamic_Spherical mode.

It supports parallel executions (auto-merging results via file locking/retries) 
and automatically generates a Markdown summary table and a dual-Y metric plot.

Usage:
    # Run a subset of k values in Terminal 1
    python src/run_sensitivity_analysis.py --k_values 1 10 --dataset src/dataset/MATH500_40.jsonl
    
    # Run another subset of k values in Terminal 2
    python src/run_sensitivity_analysis.py --k_values 50 raw --dataset src/dataset/MATH500_40.jsonl
"""

import os
import sys
import json
import time
import random
import argparse
from datetime import datetime
import torch
import numpy as np
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add src to python path if not present
sys.path.append(os.path.dirname(__file__))

import config
from config import (
    MODEL_PATH,
    VECTOR_DIR,
    LAYER_ID,
    GLOBAL_SEED,
    ACTIVE_MODEL,
    DEFAULT_DTYPE,
    DEVICE_MAP,
    ATTN_IMPLEMENTATION,
)
from run_experiment import run_full_experiment
from extract_critic_vector import extract_caa_vector
from manifold_utils import ManifoldProjector

# Import loaders directly
from loaders.aime_loader import load_aime_dataset
from loaders.math500_loader import load_math500_dataset
from loaders.zebra_logic_loader import load_zebra_dataset
from loaders.boolean_expressions_loader import load_boolean_expressions_dataset
from loaders.cruxeval_loader import load_cruxeval_dataset

# Resolve imports from HF
from transformers import AutoTokenizer, AutoModelForCausalLM


def set_seed(seed: int):
    """Fix all relevant RNG sources for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"Global seed fixed to {seed}")



def _normalize_vector(v: torch.Tensor) -> torch.Tensor:
    """Flatten, L2-normalize, and restore shape [1,1,d] as unit vector."""
    v_flat = v.view(-1)
    v_norm = v_flat.norm()
    if v_norm > 1e-8:
        v_flat = v_flat / v_norm
    return v_flat.view(1, 1, -1)


def load_dataset_by_path(dataset_path: str):
    """Detect dataset type from filename and load it."""
    dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
    dataset_path_lower = dataset_path.lower()
    
    if "math500" in dataset_path_lower:
        dataset_type = "math500"
        print(f"📂 Loading MATH500 dataset: {dataset_path}")
        dataset = load_math500_dataset(dataset_path)
    elif "zebralogic" in dataset_path_lower:
        dataset_type = "zebralogic"
        print(f"📂 Loading ZebraLogic dataset: {dataset_path}")
        dataset = load_zebra_dataset(dataset_path)
    elif "boolean_expressions" in dataset_path_lower:
        dataset_type = "boolean_expressions"
        print(f"📂 Loading Boolean Expressions dataset: {dataset_path}")
        dataset = load_boolean_expressions_dataset(dataset_path)
    elif "cruxeval" in dataset_path_lower:
        if "input" in dataset_path_lower:
            dataset_type = "cruxeval_input"
        else:
            dataset_type = "cruxeval_output"
        print(f"📂 Loading CruxEval dataset: {dataset_path} ({dataset_type})")
        dataset = load_cruxeval_dataset(dataset_path)
    else:
        dataset_type = "aime"
        print(f"📂 Loading AIME dataset: {dataset_path}")
        dataset = load_aime_dataset(dataset_path)
        
    return dataset, dataset_name, dataset_type


def load_and_merge_results(json_path: str, k_str: str, metrics_dict: dict) -> dict:
    """Safe results merging with retry loops to support concurrent runs."""
    for attempt in range(15):
        try:
            data = {}
            if os.path.exists(json_path):
                with open(json_path, "r", encoding="utf-8") as f:
                    try:
                        data = json.load(f)
                    except json.JSONDecodeError:
                        data = {}
            
            data[k_str] = metrics_dict
            
            # Write to a temporary file first, then replace atomically
            tmp_path = json_path + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            
            if os.path.exists(json_path):
                os.remove(json_path)
            os.rename(tmp_path, json_path)
            return data
        except Exception as e:
            print(f"⚠️ [Conflict] Process failed writing to {json_path}, retrying ({attempt+1}/15)... Error: {e}")
            time.sleep(1.0 + random.uniform(0.1, 1.5))
            
    raise RuntimeError(f"Could not merge results to {json_path} after 15 attempts.")


def generate_plots_and_tables(results_dict: dict, output_dir: str):
    """Generate summary Markdown table and plot the sensitivity trends."""
    if not results_dict:
        return

    # Sort keys: numeric keys sorted numerically first, followed by 'raw'
    keys = list(results_dict.keys())
    numeric_keys = sorted([k for k in keys if k.isdigit()], key=int)
    has_raw = "raw" in keys
    sorted_keys = numeric_keys + (["raw"] if has_raw else [])

    # Write Markdown summary
    md_path = os.path.join(output_dir, "sensitivity_summary.md")
    lines = [
        "# Parameter Sensitivity Analysis Summary",
        f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "| PCA Components (k) | Pass@1 Accuracy | Avg Generated Tokens | Repetition Rate (N-gram) | Evaluated Samples |",
        "| :--- | :---: | :---: | :---: | :---: |"
    ]
    
    for k in sorted_keys:
        m = results_dict[k]
        acc_str = f"{m['accuracy']:.2%}"
        rep_str = f"{m['repetition']:.3f}"
        lines.append(
            f"| **{k}** | {acc_str} ({m['correct_count']}/{m['total_count']}) | {m['tokens']:.1f} | {rep_str} | {m['total_count']} |"
        )
    
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    
    print("\n" + "="*80)
    print(f"{'SENSITIVITY ANALYSIS SUMMARY TABLE':^80}")
    print("="*80)
    print("\n".join(lines[3:]))
    print("="*80)
    print(f"💾 Markdown summary saved to {md_path}\n")

    # Generate Plot
    try:
        fig, ax1 = plt.subplots(figsize=(8, 5))
        
        x_indices = range(len(sorted_keys))
        x_labels = [f"k={k}" if k.isdigit() else "Raw (No-PCA)" for k in sorted_keys]

        accuracies = [results_dict[k]["accuracy"] * 100 for k in sorted_keys]
        repetitions = [results_dict[k]["repetition"] for k in sorted_keys]

        # Draw Accuracy (Bar or line)
        color = "tab:blue"
        ax1.set_xlabel("PCA Dimensionality (k)", fontsize=12, fontweight="bold")
        ax1.set_ylabel("Pass@1 Accuracy (%)", color=color, fontsize=12, fontweight="bold")
        line1 = ax1.plot(x_indices, accuracies, color=color, marker="o", linewidth=3, markersize=8, label="Accuracy (%)")
        ax1.tick_params(axis="y", labelcolor=color)
        ax1.set_xticks(x_indices)
        ax1.set_xticklabels(x_labels, rotation=15)
        ax1.grid(True, linestyle="--", alpha=0.5)

        # Draw Repetition Rate
        ax2 = ax1.twinx()
        color = "tab:red"
        ax2.set_ylabel("Repetition Rate (N-gram)", color=color, fontsize=12, fontweight="bold")
        line2 = ax2.plot(x_indices, repetitions, color=color, marker="s", linewidth=3, linestyle="--", markersize=8, label="Repetition Rate")
        ax2.tick_params(axis="y", labelcolor=color)

        # Title and Legends
        plt.title(f"Parameter Sensitivity Analysis ({ACTIVE_MODEL})", fontsize=14, fontweight="bold", pad=15)
        lines_combined = line1 + line2
        labels_combined = [l.get_label() for l in lines_combined]
        ax1.legend(lines_combined, labels_combined, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, "sensitivity_plot.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"📊 Live sensitivity plot updated at {plot_path}\n")
    except Exception as e:
        print(f"⚠️ Failed to generate plot: {e}")


def main():
    set_seed(GLOBAL_SEED)

    parser = argparse.ArgumentParser(
        description="Run Parameter Sensitivity Sweep over PCA Dimensionality k"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="src/dataset/MATH500_40.jsonl",
        help="Path to evaluation dataset JSONL.",
    )
    parser.add_argument(
        "--k_values",
        nargs="+",
        default=["1", "10", "50", "raw"],
        help="List of k values to run (e.g. 1 10 50 raw).",
    )
    parser.add_argument(
        "--skip_extraction",
        action="store_true",
        help="Skip activation extraction if cached activations exist.",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="./results/sensitivity",
        help="Directory to store sensitivity results.",
    )
    args = parser.parse_args()

    # ---- Setup Paths ----
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(VECTOR_DIR, exist_ok=True)
    
    raw_vector_path = os.path.join(VECTOR_DIR, "critic_raw.pt")
    activations_path = os.path.join(VECTOR_DIR, "all_activations.npy")
    shared_json_path = os.path.join(args.results_dir, "sensitivity_results.json")

    # ---- Step 1: Resolve Activation & Raw Vector ----
    model = None
    tokenizer = None
    all_activations = None
    v_raw = None

    # Determine if we can skip extraction
    can_skip = args.skip_extraction and os.path.isfile(raw_vector_path) and os.path.isfile(activations_path)
    
    if can_skip:
        print(f"📂 [Cache] Loading raw vector and activations from cache...")
        v_raw = torch.load(raw_vector_path, map_location="cpu", weights_only=True)
        all_activations = np.load(activations_path)
        print(f"  Loaded raw shape={list(v_raw.shape)}, activations shape={all_activations.shape}")
    else:
        print(f"🔬 Starting model activation extraction to compute raw CAA vector...")
        # Load model & tokenizer
        model_dtype = getattr(torch, DEFAULT_DTYPE)
        print(f"Loading model from {MODEL_PATH}...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=model_dtype,
            device_map=DEVICE_MAP,
            attn_implementation=ATTN_IMPLEMENTATION,
        )
        model.eval()
        
        # Ensure pad token differs from EOS
        if tokenizer.pad_token_id is None or tokenizer.pad_token_id == tokenizer.eos_token_id:
            tokenizer.pad_token_id = config.ENDOFTEXT_ID
            tokenizer.pad_token = tokenizer.convert_ids_to_tokens(config.ENDOFTEXT_ID)

        # Load critic data
        critic_data_path = os.path.join(os.path.dirname(__file__), "critic_data.json")
        print(f"Loading critic pairs from {critic_data_path}...")
        with open(critic_data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Extract
        v_raw, all_activations = extract_caa_vector(model, tokenizer, data, LAYER_ID)
        
        # Cache them
        torch.save(v_raw, raw_vector_path)
        np.save(activations_path, all_activations)
        print(f"💾 Saved extracted artifacts to VECTOR_DIR for future runs.")

    # ---- Step 2: Load Evaluation Dataset ----
    eval_dataset, dataset_name, dataset_type = load_dataset_by_path(args.dataset)
    print(f"Loaded {len(eval_dataset)} evaluation samples from {dataset_name} ({dataset_type})")

    # ---- Load model if not already loaded (required for generation) ----
    if model is None:
        model_dtype = getattr(torch, DEFAULT_DTYPE)
        print(f"\nLoading model for evaluation: {MODEL_PATH}...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=model_dtype,
            device_map=DEVICE_MAP,
            attn_implementation=ATTN_IMPLEMENTATION,
        )
        model.eval()
        if tokenizer.pad_token_id is None or tokenizer.pad_token_id == tokenizer.eos_token_id:
            tokenizer.pad_token_id = config.ENDOFTEXT_ID
            tokenizer.pad_token = tokenizer.convert_ids_to_tokens(config.ENDOFTEXT_ID)

    # Convert v_raw to active model configuration shape/device/dtype
    v_raw = v_raw.to(device=model.device, dtype=model.dtype)

    # ---- Step 3: Run Sensitivity Sweep ----
    print(f"\n🚀 Running sensitivity sweep over k values: {args.k_values}")
    
    for k_val in args.k_values:
        print(f"\n{'*'*60}")
        print(f"  Evaluating PCA k = {k_val} ...")
        print(f"{'*'*60}\n")
        
        # Resolve vector for this configuration
        if k_val.lower() == "raw":
            v_k = _normalize_vector(v_raw.view(1, 1, -1))
            print(f"Using Raw CAA Vector (No PCA) - norm: {v_k.float().view(-1).norm().item():.4f}")
        else:
            k = int(k_val)
            print(f"Fitting PCA with k = {k} components...")
            projector = ManifoldProjector(n_components=k)
            projector.fit(all_activations)
            
            # Project and normalize
            v_purified = projector.purify_vector(v_raw)
            v_k = _normalize_vector(v_purified.view(1, 1, -1))
            print(f"PCA Purified Vector (k={k}) - norm: {v_k.float().view(-1).norm().item():.4f}")

        # Package into control_vectors
        control_vectors = {
            "purified": v_k,
            "raw": v_k
        }

        # Run experiment
        # Using AIME_MAX_TOKENS and MAX_CONCURRENT_SEQS from config
        results = run_full_experiment(
            model=model,
            tokenizer=tokenizer,
            dataset=eval_dataset,
            dataset_name=dataset_name,
            modes=["Dynamic_Spherical"],
            control_vectors=control_vectors,
            max_concurrent_seqs=config.MAX_CONCURRENT_SEQS,
            dataset_type=dataset_type,
            results_path=None,  # Do not overwrite other experiment results files
            use_batch=True,
        )

        # Extract metrics
        stats = results["Dynamic_Spherical"]
        metrics = {
            "accuracy": float(stats["accuracy"]),
            "repetition": float(stats["repetition"]),
            "tokens": float(stats["tokens"]),
            "correct_count": int(stats["correct_count"]),
            "total_count": int(stats["total_count"])
        }

        # Safe merge with parallel processes
        print(f"Merging k={k_val} results to {shared_json_path}...")
        merged_results = load_and_merge_results(shared_json_path, str(k_val), metrics)

        # Plot current cumulative results
        generate_plots_and_tables(merged_results, args.results_dir)

    print("\n🎉 Sensitivity sweep execution finished for this process!")


if __name__ == "__main__":
    main()
