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
import threading
import queue
from datetime import datetime
import argparse
from tqdm import tqdm

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
from dataclasses import dataclass
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
    MAX_CONCURRENT_SEQS,
    TEMPERATURE,
    TOP_P,
    DEFAULT_DTYPE,
    USE_FP8,
    USE_FLASH_ATTENTION,
    RESTACK_INTERVAL,
    DEVICE_MAP,
    DO_SAMPLE,
    ENDOFTEXT_ID,
    SAFE_SCORE_RANGE,
    CONTINUOUS_ALPHA,
    CONTINUOUS_LINEAR_ALPHA,
    CAPTURE_HIDDEN_STATES,
    RESULTS_TIMESTAMP_FMT,
    JSON_INDENT,
    ENABLE_THINKING,
)
from state_monitor import InjectionState, StateMonitor
from pid_controller import PIDController
from tae_controller import TAEController
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


def _normalize_vector(v: torch.Tensor) -> torch.Tensor:
    """Flatten, L2-normalize, and restore shape [1,1,d] as unit vector."""
    orig_dtype = v.dtype
    v_f32 = v.float().view(-1)
    v_normalized = v_f32 / (v_f32.norm() + 1e-8)
    return v_normalized.view(v.shape).to(orig_dtype)


def load_control_vectors(vector_dir: str, device: str, dtype, pca_coeff: float = 1.0) -> dict[str, torch.Tensor]:
    """
    Load and normalize both the PCA-purified ('purified') and the raw ('raw')
    critic control vectors from disk, returning them in a dict.

    Returns:
        {
            "purified": Tensor [1,1,d]  (unit-normalized PCA vector)
            "raw":      Tensor [1,1,d]  (unit-normalized CAA vector, no PCA)
        }
        Either key may be absent if the corresponding file does not exist.
    """
    vectors: dict[str, torch.Tensor] = {}
    injector = VectorInjector(vector_dir, device=device, model_dtype=dtype)

    # ---- Purified (PCA-projected) vector ----
    try:
        if injector.activate("critic", coeff=pca_coeff):
            raw_norm = injector.get_raw_norm()
            print(f"  📏 [purified] raw norm: {raw_norm:.4f}")
            v = injector.get_normalized_vector()
            v_norm = _normalize_vector(v)
            print(f"  📏 [purified] final norm: {v_norm.float().view(-1).norm().item():.4f}")
            vectors["purified"] = v_norm
            injector.deactivate()
    except Exception as e:
        print(f"⚠️  Failed to load purified control vector: {e}")

    # ---- Raw (no-PCA) vector ----
    try:
        raw_path = os.path.join(vector_dir, "critic_raw.pt")
        if os.path.isfile(raw_path):
            v_raw = torch.load(raw_path, map_location="cpu", weights_only=True)
            v_raw = v_raw.to(device=device, dtype=dtype)
            v_raw_norm = _normalize_vector(v_raw.view(1, 1, -1))
            print(f"  📏 [raw] final norm: {v_raw_norm.float().view(-1).norm().item():.4f}")
            vectors["raw"] = v_raw_norm
        else:
            print(f"  ⚠️  [raw] critic_raw.pt not found in {vector_dir} — w/o Manifold ablation unavailable.")
    except Exception as e:
        print(f"⚠️  Failed to load raw control vector: {e}")

    if not vectors:
        print("⚠️  No control vectors loaded. Continuous and Dynamic modes will have no effect.")
    else:
        print(f"  ℹ️  Loaded vectors: {list(vectors.keys())}")
    return vectors


# Single sequence generation used to be here, but has been removed
# because all modes (including Dynamic_Spherical) now use fully batched tensor math.


# ======================== Batched Generation ========================
# Used for Baseline and Continuous modes

# All modes that use a dynamic controller (PID or TAE)
_DYNAMIC_MODES = frozenset([
    "Dynamic_Spherical",
    "Dynamic_Spherical_No_Manifold",
    "Dynamic_Spherical_No_ThinkBrake",
    "Dynamic_Spherical_No_EMA",
    "Dynamic_Linear",
    "True_TAE",
    "TAE_Spherical",
])

# Modes that attach a steering hook
_HOOK_MODES = frozenset([
    "Continuous",
    "Continuous_Linear",
    "Dynamic_Spherical",
    "Dynamic_Spherical_No_Manifold",
    "Dynamic_Spherical_No_ThinkBrake",
    "Dynamic_Spherical_No_EMA",
    "Dynamic_Linear",
    "True_TAE",
    "TAE_Spherical",
])

# Modes that log trajectories
_TRAJECTORY_MODES = frozenset(_HOOK_MODES)


def run_batched_generation(
    model,
    tokenizer,
    prompts: list[str],
    mode: str,
    control_vectors: dict,
    batch_size: int = BATCH_SIZE,
) -> list[dict]:
    """
    Run batched generation for ALL modes, including ablation variants.

    Args:
        model: The loaded causal LM.
        tokenizer: The tokenizer.
        prompts: List of prompt strings.
        mode: One of Baseline, Continuous, Continuous_Linear, Dynamic_Spherical,
              Dynamic_Spherical_No_Manifold, Dynamic_Linear,
              Dynamic_Spherical_No_ThinkBrake, Dynamic_Spherical_No_EMA.
        control_vectors: Dict with keys 'purified' and/or 'raw' tensors [1,1,d].
        batch_size: Number of sequences per batch.

    Returns:
        Generator of list[dict] batch results.
    """
    # ---- Resolve which control vector to use for this mode ----
    # True_TAE and w/o Manifold use raw (no-PCA) vector
    # all others use purified (PCA-projected) vector
    if mode in ("True_TAE", "Dynamic_Spherical_No_Manifold"):
        control_vector = control_vectors.get("raw", None)
    else:
        control_vector = control_vectors.get("purified", None)

    # ---- Resolve </think> token ID for ThinkBrake
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

        batch_results = []
        formatted_prompts = []
        for p in batch_prompts:
            # Qwen's template already ends with assistant\n when add_generation_prompt=True
            # Fast tokenizer will handle special tokens correctly from the message list
            text = tokenizer.apply_chat_template(
                p, 
                tokenize=False, 
                add_generation_prompt=True,
                enable_thinking=ENABLE_THINKING
            )
            formatted_prompts.append(text)

        # High-performance fast tokenizer call (strings -> tensors with padding)
        original_padding_side = getattr(tokenizer, "padding_side", "right")
        tokenizer.padding_side = 'left'
        inputs = tokenizer(
            formatted_prompts,
            padding=True,
            return_tensors="pt",
        ).to(model.device)
        
        # Reset to original padding
        tokenizer.padding_side = original_padding_side

        initial_seq_len = inputs.input_ids.shape[1]

        # Initialize batched state
        state = InjectionState(batch_size=actual_bs, device=model.device)
        pid = None
        processors = LogitsProcessorList()

        # Protection against fp16/bf16 left-padding NaN generation bugs (FlashAttention)
        class InfNanProtectionProcessor:
            def __init__(self, eos_id):
                self.eos_id = eos_id if isinstance(eos_id, int) else (eos_id[0] if isinstance(eos_id, list) else 0)

            def __call__(self, input_ids, scores):
                # Replace NaNs/Infs in the logits to prevent PyTorch multinomial crash
                # Replace NaNs with a very negative value so they are safely ignored by softmax
                torch.nan_to_num_(scores, nan=-SAFE_SCORE_RANGE, posinf=SAFE_SCORE_RANGE, neginf=-SAFE_SCORE_RANGE)
                
                # Check for COMPLETE sequence collapse (i.e. all valid logits became strongly negative)
                max_scores, _ = scores.max(dim=-1)
                collapsed_mask = max_scores <= (-SAFE_SCORE_RANGE + 1.0)
                
                if collapsed_mask.any():
                    # Record warnings
                    collapsed_indices = collapsed_mask.nonzero(as_tuple=True)[0].tolist()
                    seq_len = input_ids.shape[1]
                    for idx in collapsed_indices:
                        print(f"  [Warning] 🚨 Sequence {idx} mathematically collapsed at length {seq_len} (NaN generated). Forcing EOS.")
                    
                    # Force fully corrupted sequences to generate EOS safely instead of uniformly sampling from padding
                    scores[collapsed_mask, :] = -SAFE_SCORE_RANGE
                    scores[collapsed_mask, self.eos_id] = SAFE_SCORE_RANGE
                    
                return scores

        processors.append(InfNanProtectionProcessor(tokenizer.eos_token_id)) # Using eos_token_id to terminate safely
        
        if mode == "Continuous":
            state.intervention_active.fill_(True)
            state.alpha.fill_(CONTINUOUS_ALPHA)  # Use specialized continuous alpha
        elif mode == "Continuous_Linear":
            state.intervention_active.fill_(True)
            state.alpha.fill_(CONTINUOUS_LINEAR_ALPHA)  # Pre-calibrated linear coefficient
        elif mode in _DYNAMIC_MODES:
            # Dynamic controller — PID for closed-loop modes, TAE for open-loop
            if mode in ("True_TAE", "TAE_Spherical"):
                pid = TAEController(batch_size=actual_bs, device=model.device)
            else:
                pid = PIDController(batch_size=actual_bs, device=model.device)
            
        # Calculate actual input lengths per sequence in batch
        input_lens = inputs.attention_mask.sum(dim=1).tolist()

        if mode != "Baseline":
            # --- Ablation-specific StateMonitor overrides ---
            # w/o ThinkBrake: set margin_tau to -inf so the latch can never trigger
            monitor_margin_tau = -9999.0 if mode == "Dynamic_Spherical_No_ThinkBrake" else None
            # w/o EMA: ema_beta=1.0 means 100% current entropy, 0% history
            monitor_ema_beta = 1.0 if mode == "Dynamic_Spherical_No_EMA" else None
            # TAE modes: pass raw H_t to controller, not EMA
            use_raw_entropy = (mode in ("True_TAE", "TAE_Spherical"))

            monitor_kwargs = dict(
                state=state,
                pid_controller=pid,
                term_token_id=term_token_id,
                use_raw_entropy=use_raw_entropy,
            )
            if monitor_margin_tau is not None:
                monitor_kwargs["margin_tau"] = monitor_margin_tau
            if monitor_ema_beta is not None:
                monitor_kwargs["ema_beta"] = monitor_ema_beta

            monitor = StateMonitor(**monitor_kwargs)
            
            # Since generation doesn't expose sequence completion easily, we add a
            # quick custom logits processor that examines input_ids to update the active_mask
            class ActiveMaskProcessor:
                def __init__(self, state, tokenizer, initial_seq_len):
                    self.state = state
                    self.eos_id = tokenizer.eos_token_id
                    if isinstance(self.eos_id, list): self.eos_id = self.eos_id[0]
                    self.initial_seq_len = initial_seq_len
                    
                def __call__(self, input_ids, scores):
                    if self.eos_id is not None:
                        # Slice from initial_seq_len instead of input_lens to only check newly generated tokens
                        if input_ids.shape[1] > self.initial_seq_len:
                            gen_part = input_ids[:, self.initial_seq_len:]
                            has_eos = (gen_part == self.eos_id).any(dim=1)
                            self.state.active_mask = ~has_eos
                    return scores
            
            processors.append(ActiveMaskProcessor(state, tokenizer, initial_seq_len))
            processors.append(monitor)

        # Steering hook 
        history_hidden = []
        if control_vector is not None and mode in _HOOK_MODES:
            hook_fn, history_hidden = create_steering_hook(
                state=state,
                control_vector=control_vector,
                mode=mode,
                continuous_alpha=CONTINUOUS_ALPHA,
                continuous_linear_alpha=CONTINUOUS_LINEAR_ALPHA,
                capture_hidden_states=CAPTURE_HIDDEN_STATES,
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

            # Extract specific trajectories for this problem
            has_state = (mode != "Baseline")
            ema_traj = state.ema_trajectory[i] if has_state else []
            alpha_traj = state.alpha_trajectory[i] if has_state else []
            entropy_traj = state.entropy_trajectory[i] if has_state else []
            inv_start = state.intervention_start_step[i] if has_state else None
            inv_end = state.intervention_end_step[i] if has_state else None
            conv = state.is_converged[i].item() if has_state else False

            # Convert to plain Python list to sever CUDA references
            batch_results.append({
                "text": gen_text,
                "tokens": tokens,
                "num_tokens": len(tokens),
                "output_ids": output_ids[i].cpu().tolist(),  # plain list[int]
                "input_len": input_len,
                "ema_trajectory": ema_traj,
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
        del formatted_prompts, input_lens
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()

        yield batch_results


# ======================== Continuous Batching Generation ========================
# Replaces run_batched_generation as the primary inference engine.
#
# Strategy B: Batched KV-Cache Continuous Batching.
# Active slots are physically batched via left-padding of KV caches each step,
# eliminating serial Python dispatch overhead and maintaining ~100% GPU utilization.
# Finished slots are dynamically refilled to eliminate straggler wait time.

@dataclass
class _Slot:
    """One active decoding slot."""
    prompt_idx: int
    input_ids: torch.Tensor                 # [1, seq_len]
    attention_mask: torch.Tensor            # [1, seq_len]
    past_key_values: object                 # Tuple of tuples of KV tensors
    input_len: int
    n_generated: int = 0
    done: bool = False

def _stack_and_pad_kv_caches(slots: list[_Slot]):
    """Left-pad and batch KV caches for a single batched forward."""
    if not slots:
        return None, 0
    # Current sequence length before this decoding step is max of (input_ids - 1)
    max_len = max(s.input_ids.shape[1] - 1 for s in slots)
    num_layers = len(slots[0].past_key_values)
    batched_pkv = []
    
    for layer_idx in range(num_layers):
        layer_k, layer_v = [], []
        for s in slots:
            k, v = s.past_key_values[layer_idx]
            pad_left = max_len - k.shape[2]
            if pad_left > 0:
                k = torch.nn.functional.pad(k, (0, 0, pad_left, 0), value=0.0).to(k.dtype)
                v = torch.nn.functional.pad(v, (0, 0, pad_left, 0), value=0.0).to(v.dtype)
            layer_k.append(k)
            layer_v.append(v)
        batched_k = torch.cat(layer_k, dim=0).to(layer_k[0].dtype)
        batched_v = torch.cat(layer_v, dim=0).to(layer_v[0].dtype)
        batched_pkv.append((batched_k, batched_v))
    return tuple(batched_pkv), max_len

def _unpad_and_split_kv_caches(batched_pkv, slots: list[_Slot]):
    """Extract individual unpadded KV caches from the batched model output."""
    num_layers = len(batched_pkv)
    for i, s in enumerate(slots):
        # s.input_ids already contains the newly sampled token which hasn't been fed to model yet,
        # so the valid KV length in batched_pkv is exactly s.input_ids.shape[1] - 1
        valid_kv_len = s.input_ids.shape[1] - 1
        slot_pkv = []
        for layer_idx in range(num_layers):
            k = batched_pkv[layer_idx][0][i:i+1, :, -valid_kv_len:, :]
            v = batched_pkv[layer_idx][1][i:i+1, :, -valid_kv_len:, :]
            slot_pkv.append((k, v))
        s.past_key_values = tuple(slot_pkv)

def _stack_and_pad_attention_masks(slots: list[_Slot]):
    """Left-pad attention masks to match the current target sequence lengths."""
    max_total_len = max(s.input_ids.shape[1] for s in slots)
    batched_mask = []
    for s in slots:
        pad_left = max_total_len - s.input_ids.shape[1]
        mask = s.attention_mask
        if pad_left > 0:
            mask = torch.nn.functional.pad(mask, (pad_left, 0), value=0)
        batched_mask.append(mask)
    return torch.cat(batched_mask, dim=0)

def _build_global_components(mode: str, term_token_id, device: str, batch_size: int):
    """Instantiate shared InjectionState, PID, and StateMonitor for the max capacity."""
    state = InjectionState(batch_size=batch_size, device=device)
    pid = None

    if mode == "Continuous":
        state.intervention_active.fill_(True)
        state.alpha.fill_(CONTINUOUS_ALPHA)
    elif mode == "Continuous_Linear":
        state.intervention_active.fill_(True)
        state.alpha.fill_(CONTINUOUS_LINEAR_ALPHA)
    elif mode in _DYNAMIC_MODES:
        if mode in ("True_TAE", "TAE_Spherical"):
            pid = TAEController(batch_size=batch_size, device=device)
        else:
            pid = PIDController(batch_size=batch_size, device=device)

    monitor = None
    if mode != "Baseline":
        monitor_margin_tau = -9999.0 if mode == "Dynamic_Spherical_No_ThinkBrake" else None
        monitor_ema_beta   = 1.0     if mode == "Dynamic_Spherical_No_EMA"              else None
        use_raw_entropy    = (mode in ("True_TAE", "TAE_Spherical"))

        monitor_kwargs = dict(
            state=state,
            pid_controller=pid,
            term_token_id=term_token_id,
            use_raw_entropy=use_raw_entropy,
        )
        if monitor_margin_tau is not None:
            monitor_kwargs["margin_tau"] = monitor_margin_tau
        if monitor_ema_beta is not None:
            monitor_kwargs["ema_beta"] = monitor_ema_beta

        monitor = StateMonitor(**monitor_kwargs)

    return state, pid, monitor

def _slot_to_result(slot: _Slot, state, slot_idx: int, tokenizer) -> dict:
    """Convert a finished slot into a result dict, pulling from global state slice."""
    generated_ids = slot.input_ids[0, slot.input_len:]
    if tokenizer.pad_token_id is not None:
        mask = generated_ids != tokenizer.pad_token_id
        generated_ids = generated_ids[mask]

    gen_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    tokens   = [tokenizer.decode([t]).replace("\n", "↵") for t in generated_ids]

    has_state = state is not None
    return {
        "text":               gen_text,
        "tokens":             tokens,
        "num_tokens":         len(tokens),
        "output_ids":         slot.input_ids[0].cpu().tolist(),
        "input_len":          slot.input_len,
        "ema_trajectory":     list(state.ema_trajectory[slot_idx]) if has_state else [],
        "alpha_trajectory":   list(state.alpha_trajectory[slot_idx]) if has_state else [],
        "entropy_trajectory": list(state.entropy_trajectory[slot_idx]) if has_state else [],
        "history_hidden":     [],
        "intervention_start": state.intervention_start_step[slot_idx] if has_state else None,
        "intervention_end":   state.intervention_end_step[slot_idx] if has_state else None,
        "convergence":        state.is_converged[slot_idx].item() if has_state else False,
    }

def _safe_score_range_clean(scores: torch.Tensor, eos_id: int) -> None:
    fp8_safe_min = -400.0
    fp8_safe_max = 400.0
    if USE_FP8:
        scores.clamp_(min=fp8_safe_min, max=fp8_safe_max)

    torch.nan_to_num_(
        scores, 
        nan=fp8_safe_min if USE_FP8 else -SAFE_SCORE_RANGE, 
        posinf=fp8_safe_max if USE_FP8 else SAFE_SCORE_RANGE, 
        neginf=fp8_safe_min if USE_FP8 else -SAFE_SCORE_RANGE
    )
    max_scores, _ = scores.max(dim=-1)
    collapsed_threshold = fp8_safe_min + 1.0 if USE_FP8 else (-SAFE_SCORE_RANGE + 1.0)
    collapsed = max_scores <= collapsed_threshold
    
    if collapsed.any():
        scores[collapsed, :] = fp8_safe_min if USE_FP8 else -SAFE_SCORE_RANGE
        scores[collapsed, eos_id] = fp8_safe_max if USE_FP8 else SAFE_SCORE_RANGE

def run_continuous_batching_generation(
    model,
    tokenizer,
    prompts: list,
    mode: str,
    control_vectors: dict,
    max_concurrent_seqs: int = MAX_CONCURRENT_SEQS,
    continuous_alpha: float = CONTINUOUS_ALPHA,
    continuous_linear_alpha: float = CONTINUOUS_LINEAR_ALPHA,
    max_new_tokens: int = AIME_MAX_TOKENS,
    restack_interval: int = RESTACK_INTERVAL,
) -> "Generator[list[dict], None, None]":
    device = model.device

    if mode in ("True_TAE", "Dynamic_Spherical_No_Manifold"):
        control_vector = control_vectors.get("raw", None)
    else:
        control_vector = control_vectors.get("purified", None)

    term_token_id = None
    try:
        term_ids = tokenizer.encode("</think>", add_special_tokens=False)
        if term_ids:
            term_token_id = term_ids[-1]
    except Exception:
        pass

    eos_id = tokenizer.eos_token_id
    if isinstance(eos_id, list): eos_id = eos_id[0]

    from collections import deque
    pending = deque(range(len(prompts)))
    
    # Pre-allocate global state
    state, pid, monitor = _build_global_components(mode, term_token_id, device, max_concurrent_seqs)
    
    hook_handle = None
    try:
        # Global steering hook
        if control_vector is not None and mode in _HOOK_MODES:
            hook_fn, _ = create_steering_hook(
                state=state,
                control_vector=control_vector,
                mode=mode,
                continuous_alpha=continuous_alpha,
                continuous_linear_alpha=continuous_linear_alpha,
                capture_hidden_states=False,
            )
            layer = model.model.layers[LAYER_ID]
            hook_handle = layer.register_forward_hook(hook_fn)

        slots: list[_Slot | None] = [None] * max_concurrent_seqs

        def _sample_batch_tokens(logits_2d: torch.Tensor, do_sample: bool, temperature: float, top_p: float) -> torch.Tensor:
            if do_sample:
                logits_scaled = logits_2d / max(temperature, 1e-6)
                sorted_logits, sorted_idx = torch.sort(logits_scaled, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_remove = cumulative_probs - torch.softmax(sorted_logits, dim=-1) > top_p
                sorted_logits[sorted_remove] = -float("inf")
                logits_final = torch.full_like(logits_scaled, -float("inf"))
                logits_final.scatter_(1, sorted_idx, sorted_logits)
                probs = torch.softmax(logits_final, dim=-1)
                next_tok = torch.multinomial(probs, num_samples=1)
            else:
                next_tok = logits_2d.argmax(dim=-1, keepdim=True)
            return next_tok # [K, 1]

        def _reset_state_slot(idx: int):
            if state is not None:
                state.active_mask[idx] = True
                state.is_converged[idx] = False
                state.ema_entropy[idx] = 0.0
                state.ema_trajectory[idx] = []
                state.alpha_trajectory[idx] = []
                state.entropy_trajectory[idx] = []
                state.intervention_start_step[idx] = None
                state.intervention_end_step[idx] = None
                if mode == "Continuous":
                    state.alpha[idx] = continuous_alpha
                elif mode == "Continuous_Linear":
                    state.alpha[idx] = continuous_linear_alpha
                else:
                    state.alpha[idx] = 0.0
            if pid is not None and hasattr(pid, 'integral'):
                pid.integral[idx] = 0.0
                pid.prev_error[idx] = 0.0

        def _prefill_slot(slot_idx: int, prompt_idx: int) -> _Slot:
            p = prompts[prompt_idx]
            text = tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True, enable_thinking=ENABLE_THINKING)
            enc = tokenizer(text, return_tensors="pt").to(device)
            
            _reset_state_slot(slot_idx)
            
            if state is not None:
                state.active_batch_indices = [slot_idx]
                
            with torch.no_grad():
                if USE_FP8:
                    from torch.cuda.amp import autocast
                    with autocast(dtype=torch.float8_e4m3fn):
                        out = model(input_ids=enc.input_ids, attention_mask=enc.attention_mask, use_cache=True, return_dict=True)
                else:
                    out = model(input_ids=enc.input_ids, attention_mask=enc.attention_mask, use_cache=True, return_dict=True)
                
            first_logits = out.logits[:, -1, :].clone()
            _safe_score_range_clean(first_logits, eos_id)
            
            slot = _Slot(
                prompt_idx=prompt_idx,
                input_ids=enc.input_ids,
                attention_mask=enc.attention_mask,
                past_key_values=out.past_key_values,
                input_len=enc.input_ids.shape[1],
            )

            if monitor is not None:
                dummy_ids = torch.empty((max_concurrent_seqs, enc.input_ids.shape[1]), dtype=torch.long, device=device)
                dummy_ids[slot_idx] = enc.input_ids[0]
                dummy_logits = torch.empty((max_concurrent_seqs, first_logits.shape[1]), dtype=first_logits.dtype, device=device)
                dummy_logits[slot_idx] = first_logits[0]
                
                saved_mask = state.active_mask.clone()
                state.active_mask.fill_(False)
                state.active_mask[slot_idx] = True
                
                monitor(dummy_ids, dummy_logits)
                state.active_mask = saved_mask
                
            next_tok = _sample_batch_tokens(first_logits, DO_SAMPLE, TEMPERATURE, TOP_P) # [1, 1]
            slot.input_ids = torch.cat([slot.input_ids, next_tok], dim=1)
            slot.attention_mask = torch.ones(1, slot.input_ids.shape[1], dtype=torch.long, device=device)
            slot.n_generated = 1
            if (next_tok.item() == eos_id) or (slot.n_generated >= max_new_tokens):
                slot.done = True
                
            return slot

        # 1. Fill initial slot pool
        for i in range(min(max_concurrent_seqs, len(pending))):
            slot = _prefill_slot(i, pending.popleft())
            if slot.done:
                yield [_slot_to_result(slot, state, i, tokenizer)]
                slot.past_key_values = None
                slots[i] = None
            else:
                slots[i] = slot

        # 2. Main decode loop
        # [Strategy C]: Establish steady-state batched tensors OUTSIDE the inner step loop
        active_indices = [i for i, s in enumerate(slots) if s is not None]
        active_list = [slots[i] for i in active_indices]
        
        batched_pkv, _ = _stack_and_pad_kv_caches(active_list)
        batched_mask = _stack_and_pad_attention_masks(active_list)
        
        step_counter = 0
        while any(s is not None for s in slots):
            step_counter += 1
            # Extract last tokens for current step (O(K))
            batched_last_tokens = torch.cat([s.input_ids[:, -1:] for s in active_list], dim=0) # [K, 1]
            
            if state is not None:
                state.active_batch_indices = active_indices
                
            with torch.no_grad():
                if USE_FP8:
                    from torch.cuda.amp import autocast
                    with autocast(dtype=torch.float8_e4m3fn):
                        out = model(
                            input_ids=batched_last_tokens,
                            attention_mask=batched_mask,
                            past_key_values=batched_pkv,
                            use_cache=True,
                            return_dict=True,
                        )
                else:
                    out = model(
                        input_ids=batched_last_tokens,
                        attention_mask=batched_mask,
                        past_key_values=batched_pkv,
                        use_cache=True,
                        return_dict=True,
                    )
                
            # [Strategy C]: In-place reception of appended KV caches, avoiding Python-level pad & cat overhead
            batched_pkv = out.past_key_values
            
            logits_2d = out.logits[:, -1, :].clone() # [K, V]
            _safe_score_range_clean(logits_2d, eos_id) 
            
            if monitor is not None:
                max_len = batched_mask.shape[1]
                dummy_ids = torch.empty((max_concurrent_seqs, max_len), dtype=torch.long, device=device)
                dummy_logits = torch.empty((max_concurrent_seqs, logits_2d.shape[1]), dtype=logits_2d.dtype, device=device)
                
                for list_i, slot_i in enumerate(active_indices):
                    slen = active_list[list_i].input_ids.shape[1]
                    dummy_ids[slot_i, :slen] = active_list[list_i].input_ids[0]
                    dummy_logits[slot_i] = logits_2d[list_i]
                    
                saved_mask = state.active_mask.clone()
                state.active_mask.fill_(False)
                state.active_mask[active_indices] = True
                
                monitor(dummy_ids, dummy_logits)
                state.active_mask = saved_mask
                
            next_tokens = _sample_batch_tokens(logits_2d, DO_SAMPLE, TEMPERATURE, TOP_P) # [K, 1]
            
            has_finished = False
            for list_i, slot_i in enumerate(active_indices):
                s = active_list[list_i]
                nxt = next_tokens[list_i:list_i+1] # [1, 1]
                s.input_ids = torch.cat([s.input_ids, nxt], dim=1)
                # Just append to track the mask lengths locally for each slot
                s.attention_mask = torch.cat([s.attention_mask, torch.ones(1, 1, dtype=torch.long, device=device)], dim=1)
                s.n_generated += 1
                
                if (nxt.item() == eos_id) or (s.n_generated >= max_new_tokens):
                    s.done = True
                    has_finished = True

            # Trigger periodic garbage collection / padding compaction
            if step_counter % restack_interval == 0:
                has_finished = True

            # Efficiently prepare mask for the NEXT token generation by appending 1s
            ones_col = torch.ones((len(active_list), 1), dtype=batched_mask.dtype, device=device)
            batched_mask = torch.cat([batched_mask, ones_col], dim=1)
            
            # [Strategy C]: Rare-Restacking
            # Only incur memory rebuilding overhead if a sequence finished and slots were swapped
            if has_finished:
                # 1. Extract the ground-truth PKVs from the huge batched_pkv before we destroy it
                _unpad_and_split_kv_caches(batched_pkv, active_list)
                
                # 2. Process completions and refill slots
                for list_i, slot_i in enumerate(active_indices):
                    s = active_list[list_i]
                    if s.done:
                        yield [_slot_to_result(s, state, slot_i, tokenizer)]
                        
                        # Free slot memory
                        s.past_key_values = None
                        s.input_ids = None
                        s.attention_mask = None
                        
                        if pending:
                            new_slot = _prefill_slot(slot_i, pending.popleft())
                            if new_slot.done:  # Edge case: prefill immediately generated EOS
                                yield [_slot_to_result(new_slot, state, slot_i, tokenizer)]
                                new_slot.past_key_values = None
                                slots[slot_i] = None
                            else:
                                slots[slot_i] = new_slot
                        else:
                            slots[slot_i] = None
                            
                # 3. Establish a pristine state for the next steady-state batch
                active_indices = [i for i, s in enumerate(slots) if s is not None]
                if not active_indices:
                    break
                
                active_list = [slots[i] for i in active_indices]
                batched_pkv, _ = _stack_and_pad_kv_caches(active_list)
                batched_mask = _stack_and_pad_attention_masks(active_list)
    finally:
        if hook_handle is not None:
            hook_handle.remove()
        del state, pid, monitor
        torch.cuda.empty_cache()
        gc.collect()


# ======================== Full Experiment Pipeline ========================

def run_full_experiment(
    model,
    tokenizer,
    dataset: list[dict],
    dataset_name: str = "AIME",
    modes: list[str] | None = None,
    control_vectors: dict | None = None,
    batch_size: int = BATCH_SIZE,
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
        control_vectors: Dict with 'purified' and/or 'raw' steering vectors.
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
        modes = EXPERIMENT_MODES
    if control_vectors is None:
        control_vectors = {}

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

                if mode_name in ("Dynamic_Spherical", "Continuous", "Continuous_Linear"):
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

        print(f"  Using CONTINUOUS BATCHING (max_concurrent_seqs={batch_size}) for ALL modes...")
        
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

        # Create generator — continuous batching eliminates straggler padding waste
        gen_iterator = run_continuous_batching_generation(
            model, tokenizer, prompts, mode, control_vectors,
            max_concurrent_seqs=batch_size
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
        accuracy = mode_stats["mode_correct"] / mode_total if mode_total > 0 else 0
        avg_rep = np.mean(mode_stats["mode_repetitions"]) if mode_stats["mode_repetitions"] else 0
        avg_tokens = mode_stats["mode_tokens_total"] // max(mode_total, 1)

        mode_data["accuracy"] = float(accuracy)
        mode_data["correct_count"] = mode_stats["mode_correct"]
        mode_data["repetition"] = float(avg_rep)
        mode_data["tokens"] = int(avg_tokens) 

        per_problem_details = mode_data["per_problem"]

        # Module-level diagnostics summary (all Dynamic_* modes)
        if mode in _DYNAMIC_MODES:
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
    
    model_kwargs = {
        "pretrained_model_name_or_path": MODEL_PATH,
        "device_map": DEVICE_MAP,
    }
    
    model_kwargs["torch_dtype"] = model_dtype
    if USE_FP8:
        try:
            import torch._inductor.config
            torch._inductor.config.fp8_e4m3fn = True
            torch._inductor.config.use_mixed_mm = True
        except ImportError:
            pass

    if USE_FLASH_ATTENTION:
        model_kwargs["attn_implementation"] = "flash_attention_2"
        
    print(f"\n🚀 Loading model {MODEL_PATH}")
    print(f"   FP8 Enabled: {USE_FP8} | Flash Attention 2: {USE_FLASH_ATTENTION} | Dtype: {model_kwargs.get('torch_dtype')}")
    model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    # Ensure pad token is set and DIFFERS from eos_token.
    # Qwen3 eos = im_end (151645); native pad = endoftext (151643).
    # If pad == eos, batched generate() cannot stop at EOS.
    if tokenizer.pad_token_id is None or tokenizer.pad_token_id == tokenizer.eos_token_id:
        tokenizer.pad_token_id = ENDOFTEXT_ID
        tokenizer.pad_token = tokenizer.convert_ids_to_tokens(ENDOFTEXT_ID)

    # ---- Load control vectors (purified + raw) ----
    print("\n🔬 Loading control vectors...")
    control_vectors = load_control_vectors(
        VECTOR_DIR,
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype=model_dtype,
    )

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
        control_vectors=control_vectors,
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
