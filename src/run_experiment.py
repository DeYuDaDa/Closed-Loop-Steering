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
  - Answer extraction via \boxed{} regex (academic standard)
  - Datasets run separately for parallel GPU execution

Batch Processing Strategy:
  - ALL modes utilize a unified Continuous Batching engine.
  - Generates sequences dynamically to maintain near 100% GPU utilization.
  - Active slots are batched and processed concurrently, avoiding stride fragmentation.
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
from collections import deque
from dataclasses import dataclass, field

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
from transformers.cache_utils import DynamicCache

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
    TOP_K,
    MIN_P,
    DEFAULT_DTYPE,
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
    v_flat = v.view(-1)
    v_normalized = v_flat / v_flat.norm()
    return v_normalized.view(v.shape)


def load_control_vectors(vector_dir: str, device: str, dtype) -> dict[str, torch.Tensor]:
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
        if injector.activate("critic", coeff=1.0):
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

# ======================== Continuous Batching Generation ========================
# Primary inference engine.  All K active slots are forwarded in a SINGLE model()
# call per decode step, eliminating K-fold VRAM reloading and Python-launch overhead.
#
# Isolation contract: each physical slot index i in the global InjectionState owns
# exclusive state fields (alpha[i], ema_entropy[i], is_converged[i], prev_error[i]).
# The single shared hook reads alpha[active_batch_indices] → applies per-row SLERP.
# StateMonitor is called once per step with the full [K, V] logit matrix.

from dataclasses import dataclass, field


@dataclass
class _Slot:
    """One active decoding slot in the batched continuous-batching pool."""
    prompt_idx: int                   # Index into the original prompts list
    input_ids: torch.Tensor           # [1, seq_len]  current full sequence
    attention_mask: torch.Tensor      # [1, seq_len]
    past_key_values: object           # per-slot KV cache
    input_len: int                    # length of the original prompt tokens
    n_generated: int = 0
    done: bool = False


# def _stack_and_pad_kv_caches(slots: list):
#     """Left-pad and batch KV caches for a single batched forward pass.
#     Returns (batched_pkv, max_kv_len).
#     """
#     if not slots:
#         return None, 0
#     # KV length = current full sequence length minus the last token
#     # (the last token is the one we are about to feed in this step)
#     max_len = max(s.input_ids.shape[1] - 1 for s in slots)
    
#     pkv_0 = slots[0].past_key_values
#     is_dynamic = hasattr(pkv_0, "key_cache")
#     num_layers = len(pkv_0.key_cache) if is_dynamic else len(pkv_0)
    
#     batched_pkv = []
#     for layer_idx in range(num_layers):
#         layer_k, layer_v = [], []
#         for s in slots:
#             pkv = s.past_key_values
#             if hasattr(pkv, "key_cache"):
#                 k, v = pkv.key_cache[layer_idx], pkv.value_cache[layer_idx]
#             else:
#                 k, v = pkv[layer_idx]
                
#             pad_left = max_len - k.shape[2]
#             if pad_left > 0:
#                 k = torch.nn.functional.pad(k, (0, 0, pad_left, 0), value=0.0)
#                 v = torch.nn.functional.pad(v, (0, 0, pad_left, 0), value=0.0)
#             layer_k.append(k)
#             layer_v.append(v)
#         batched_pkv.append((torch.cat(layer_k, dim=0), torch.cat(layer_v, dim=0)))
#     return tuple(batched_pkv), max_len
# 
# def _unpad_and_split_kv_caches(batched_pkv, slots: list):
#     """Split batched KV cache (from model output) back into per-slot caches.
#     Called only when a slot finishes and the batch must be restructured.
#     """
#     is_dynamic = hasattr(batched_pkv, "key_cache")
#     num_layers = len(batched_pkv.key_cache) if is_dynamic else len(batched_pkv)
    
#     for i, s in enumerate(slots):
#         # s.input_ids already has the newly sampled token appended,
#         # so the valid KV length in batched_pkv is input_ids.shape[1] - 1
#         valid_kv_len = s.input_ids.shape[1] - 1
#         slot_pkv = []
#         for layer_idx in range(num_layers):
#             if is_dynamic:
#                 k = batched_pkv.key_cache[layer_idx][i:i+1, :, -valid_kv_len:, :]
#                 v = batched_pkv.value_cache[layer_idx][i:i+1, :, -valid_kv_len:, :]
#             else:
#                 k = batched_pkv[layer_idx][0][i:i+1, :, -valid_kv_len:, :]
#                 v = batched_pkv[layer_idx][1][i:i+1, :, -valid_kv_len:, :]
#             slot_pkv.append((k, v))
#         s.past_key_values = tuple(slot_pkv)

def _stack_and_pad_kv_caches(slots: list):
    """Left-pad and batch KV caches for a single batched forward pass.
    Returns (batched_pkv, max_kv_len).
    """
    if not slots:
        return None, 0
    max_len = max(s.input_ids.shape[1] - 1 for s in slots)
    
    pkv_0 = slots[0].past_key_values
    is_official_dynamic = (
        "DynamicCache" in str(type(pkv_0)) and 
        hasattr(pkv_0, "layers")
    )
    num_layers = len(pkv_0)

    # 初始化一个新的官方DynamicCache（保持类型，模型要求！）
    batched_cache = DynamicCache()
    for layer_idx in range(num_layers):
        layer_k, layer_v = [], []
        for s in slots:
            pkv = s.past_key_values
            layer_cache = pkv.layers[layer_idx]
            k = layer_cache.keys
            v = layer_cache.values

            # 左填充
            pad_left = max_len - k.shape[2]
            if pad_left > 0:
                k = torch.nn.functional.pad(k, (0, 0, pad_left, 0), value=0.0)
                v = torch.nn.functional.pad(v, (0, 0, pad_left, 0), value=0.0)
            layer_k.append(k)
            layer_v.append(v)
        
        # 拼接后存入DynamicCache
        batched_k = torch.cat(layer_k, dim=0)
        batched_v = torch.cat(layer_v, dim=0)
        batched_cache.update(batched_k, batched_v, layer_idx)

    # ✅ 返回DynamicCache对象，不是tuple！模型强制要求
    return batched_cache, max_len


def _unpad_and_split_kv_caches(batched_pkv: DynamicCache, slots: list):
    """Split batched KV cache back into per-slot DynamicCache objects."""
    num_layers = len(batched_pkv)
    
    for i, s in enumerate(slots):
        valid_kv_len = s.input_ids.shape[1] - 1
        # 为每个slot创建新的DynamicCache
        slot_cache = DynamicCache()
        
        for layer_idx in range(num_layers):
            layer_cache = batched_pkv.layers[layer_idx]
            # 截取当前样本的有效KV
            k = layer_cache.keys[i:i+1, :, -valid_kv_len:, :]
            v = layer_cache.values[i:i+1, :, -valid_kv_len:, :]
            # 存入slot的DynamicCache
            slot_cache.update(k, v, layer_idx)
        
        # ✅ 赋值DynamicCache，不是tuple！
        s.past_key_values = slot_cache


def _stack_and_pad_attention_masks(slots: list):
    """Left-pad attention masks to match the current padded sequence lengths."""
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
    """Instantiate a single shared InjectionState, PID controller, and StateMonitor
    for the entire slot pool (batch_size = max_concurrent_seqs).
    Each physical slot index i owns exclusive slices of all state tensors.
    """
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
        monitor_ema_beta   = 1.0     if mode == "Dynamic_Spherical_No_EMA"         else None
        use_raw_entropy    = mode in ("True_TAE", "TAE_Spherical")

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
    """Convert a finished slot into the result dict expected by run_full_experiment.
    Reads trajectory data from the global state at physical index slot_idx.
    """
    generated_ids = slot.input_ids[0, slot.input_len:]
    if tokenizer.pad_token_id is not None:
        mask = generated_ids != tokenizer.pad_token_id
        generated_ids = generated_ids[mask]

    gen_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    tokens   = [tokenizer.decode([t]).replace("\n", "\u21b5") for t in generated_ids]

    has_state = state is not None
    return {
        "text":               gen_text,
        "tokens":             tokens,
        "num_tokens":         len(tokens),
        "output_ids":         slot.input_ids[0].cpu().tolist(),
        "input_len":          slot.input_len,
        "prompt_idx":         slot.prompt_idx,
        "ema_trajectory":     state.ema_trajectory[slot_idx]          if has_state else [],
        "alpha_trajectory":   state.alpha_trajectory[slot_idx]        if has_state else [],
        "entropy_trajectory": state.entropy_trajectory[slot_idx]      if has_state else [],
        "history_hidden":     [],
        "intervention_start": state.intervention_start_step[slot_idx] if has_state else None,
        "intervention_end":   state.intervention_end_step[slot_idx]   if has_state else None,
        "convergence":        state.is_converged[slot_idx].item()     if has_state else False,
    }


def _safe_score_range_clean(scores: torch.Tensor, eos_id: int) -> torch.Tensor:
    """In-place NaN/Inf protection (mirrors InfNanProtectionProcessor)."""
    torch.nan_to_num_(scores, nan=-SAFE_SCORE_RANGE, posinf=SAFE_SCORE_RANGE,
                      neginf=-SAFE_SCORE_RANGE)
    max_scores, _ = scores.max(dim=-1)
    collapsed = max_scores <= (-SAFE_SCORE_RANGE + 1.0)
    if collapsed.any():
        scores[collapsed, :] = -SAFE_SCORE_RANGE
        scores[collapsed, eos_id] = SAFE_SCORE_RANGE
    return scores


def _sample_batch_tokens(
    logits_2d: torch.Tensor,
    do_sample: bool,
    temperature: float,
    top_p: float,
    top_k: int = 0,
    min_p: float = 0.0,
) -> torch.Tensor:
    """Sample next tokens for a [K, V] logit matrix. Returns [K, 1].

    Sampling pipeline (in order):
      1. Temperature scaling
      2. Top-K hard cap  (if top_k > 0)
      3. Min-P dynamic floor  (if min_p > 0.0)
         Discard tokens where P < min_p * P_max.
         Adapts to model confidence: strict when confident, lenient when uncertain.
      4. Top-P nucleus filter  (if top_p < 1.0)
      5. Softmax + multinomial draw
    """
    if do_sample:
        logits_scaled = logits_2d / max(temperature, 1e-6)

        # --- Stage 1: Top-K hard cap ---
        # Physically removes all tokens outside the top-k, preventing long-tail
        # noise from polluting the nucleus under any Top-P setting.
        if top_k > 0:
            top_k_effective = min(top_k, logits_scaled.size(-1))
            kth_vals = torch.topk(logits_scaled, top_k_effective, dim=-1)[0][..., -1, None]
            logits_scaled = logits_scaled.masked_fill(logits_scaled < kth_vals, -float("inf"))

        # --- Stage 2: Min-P dynamic floor ---
        # Threshold = min_p * P_max:  tight when the model is confident (high P_max),
        # relaxed when the model is uncertain (low P_max).
        # This breaks degeneration cascades by preventing low-quality tokens from
        # re-entering the pool after context contamination.
        if min_p > 0.0:
            probs_tmp = torch.softmax(logits_scaled, dim=-1)
            top_prob = probs_tmp.max(dim=-1, keepdim=True)[0]  # [K, 1]
            scaled_min_p = min_p * top_prob
            logits_scaled = logits_scaled.masked_fill(probs_tmp < scaled_min_p, -float("inf"))

        # --- Stage 3: Top-P nucleus filter ---
        sorted_logits, sorted_idx = torch.sort(logits_scaled, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_remove = cumulative_probs - torch.softmax(sorted_logits, dim=-1) > top_p
        sorted_logits[sorted_remove] = -float("inf")
        logits_final = torch.full_like(logits_scaled, -float("inf"))
        logits_final.scatter_(1, sorted_idx, sorted_logits)

        probs = torch.softmax(logits_final, dim=-1)
        next_tok = torch.multinomial(probs, num_samples=1)  # [K, 1]
    else:
        next_tok = logits_2d.argmax(dim=-1, keepdim=True)   # [K, 1]
    return next_tok



def run_continuous_batching_generation(
    model,
    tokenizer,
    prompts: list,
    mode: str,
    control_vectors: dict,
    max_concurrent_seqs: int = MAX_CONCURRENT_SEQS,
) -> "Generator[list[dict], None, None]":
    """
    Batched continuous-batching inference engine.

    All K active slots are forwarded in a SINGLE model() call per decode step,
    allowing the GPU to amortize weight loading across the full batch.  When a
    slot finishes (EOS or max_new_tokens), KV caches are split, the result is
    yielded, and the freed slot is immediately refilled with the next pending
    prompt (rare-restacking pattern).

    Isolation contract:
        A single global InjectionState of size `max_concurrent_seqs` is shared.
        Physical slot index `i` owns exclusive slices:
            state.alpha[i], ema_entropy[i], is_converged[i],
            pid.prev_error[i], pid.integral[i].
        The single registered hook reads alpha[active_batch_indices] which maps
        physical indices to positional batch rows \u2014 no cross-slot write is possible.

    Timing (same as model.generate):
        step t\u22121 \u2192 StateMonitor writes alpha[t\u22121]
        step t   \u2192 hook reads alpha[active_indices], applies per-row SLERP
                 \u2192 StateMonitor reads logits[t], writes alpha[t]

    Yields:
        list[dict] \u2014 one result dict per completed prompt (list of 1),
                     preserving the caller's batch bookkeeping convention.
    """
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
    if isinstance(eos_id, list):
        eos_id = eos_id[0]

    # ---- Global shared state + single hook ----
    state, pid, monitor = _build_global_components(
        mode, term_token_id, device, max_concurrent_seqs
    )

    hook_handle = None
    if control_vector is not None and mode in _HOOK_MODES:
        hook_fn, _ = create_steering_hook(
            state=state,
            control_vector=control_vector,
            mode=mode,
            continuous_alpha=CONTINUOUS_ALPHA,
            continuous_linear_alpha=CONTINUOUS_LINEAR_ALPHA,
            capture_hidden_states=False,
        )
        layer = model.model.layers[LAYER_ID]
        hook_handle = layer.register_forward_hook(hook_fn)

    # ---- Slot pool: fixed-size list, None = empty ----
    from collections import deque
    pending = deque(range(len(prompts)))
    slots: list[_Slot | None] = [None] * max_concurrent_seqs

    def _reset_slot_state(slot_idx: int):
        """Reset global state fields for a freshly occupied slot."""
        state.active_mask[slot_idx] = True
        state.is_converged[slot_idx] = False
        state.ema_entropy[slot_idx] = 0.0
        state.ema_trajectory[slot_idx] = []
        state.alpha_trajectory[slot_idx] = []
        state.entropy_trajectory[slot_idx] = []
        state.intervention_start_step[slot_idx] = None
        state.intervention_end_step[slot_idx] = None
        if mode == "Continuous":
            state.alpha[slot_idx] = CONTINUOUS_ALPHA
        elif mode == "Continuous_Linear":
            state.alpha[slot_idx] = CONTINUOUS_LINEAR_ALPHA
        else:
            state.alpha[slot_idx] = 0.0
        if pid is not None:
            if hasattr(pid, "integral"):
                pid.integral[slot_idx] = 0.0
            if hasattr(pid, "prev_error"):
                pid.prev_error[slot_idx] = 0.0
                
        if hasattr(state, "low_entropy_count"):
            state.low_entropy_count[slot_idx] = 0
            state.trigger_perturbation[slot_idx] = False
            state.cooldown_counter[slot_idx] = 0

    def _prefill_slot(slot_idx: int, prompt_idx: int) -> _Slot:
        """Tokenise + prefill one prompt into physical slot slot_idx."""
        p = prompts[prompt_idx]
        try:
            text = tokenizer.apply_chat_template(
                p, tokenize=False, add_generation_prompt=True,
                enable_thinking=ENABLE_THINKING,
            )
        except TypeError:
            text = tokenizer.apply_chat_template(
                p, tokenize=False, add_generation_prompt=True,
            )
        tokenizer.padding_side = "left"
        enc = tokenizer(text, return_tensors="pt").to(device)
        tokenizer.padding_side = "right"

        _reset_slot_state(slot_idx)
        # Tell the hook which physical index is active for this prefill forward.
        state.active_batch_indices = [slot_idx]

        with torch.no_grad():
            out = model(
                input_ids=enc.input_ids,
                attention_mask=enc.attention_mask,
                use_cache=True,
                return_dict=True,
            )

        first_logits_2d = _safe_score_range_clean(out.logits[:, -1, :], eos_id)  # [1, V]

        slot = _Slot(
            prompt_idx=prompt_idx,
            input_ids=enc.input_ids,
            attention_mask=enc.attention_mask,
            past_key_values=out.past_key_values,
            input_len=enc.input_ids.shape[1],
        )

        # Prime StateMonitor with prefill logits so alpha is ready at decode step 0.
        if monitor is not None:
            saved_mask = state.active_mask.clone()
            state.active_mask.fill_(False)
            state.active_mask[slot_idx] = True
            # Build full-width dummy buffers (monitor indexes via active_mask)
            vocab_size = model.config.vocab_size
            dummy_logits = torch.zeros(
                (max_concurrent_seqs, vocab_size), dtype=torch.float32, device=device
            )
            dummy_logits[slot_idx] = first_logits_2d[0]
            dummy_ids = torch.zeros(
                (max_concurrent_seqs, enc.input_ids.shape[1]), dtype=torch.long, device=device
            )
            dummy_ids[slot_idx] = enc.input_ids[0]
            monitor(dummy_ids, dummy_logits)
            state.active_mask = saved_mask

        # Sample the very first token from the prefill logits.
        first_tok = _sample_batch_tokens(first_logits_2d, DO_SAMPLE, TEMPERATURE, TOP_P, TOP_K, MIN_P)  # [1, 1]
        slot.input_ids = torch.cat([slot.input_ids, first_tok], dim=1)
        slot.attention_mask = torch.ones(
            1, slot.input_ids.shape[1], dtype=torch.long, device=device
        )
        slot.n_generated = 1

        first_tok_val = first_tok.view(-1)[0].item()
        if first_tok_val == eos_id or slot.n_generated >= AIME_MAX_TOKENS:
            slot.done = True

        return slot

    try:
        # ---- 1. Fill initial pool ----
        for i in range(min(max_concurrent_seqs, len(pending))):
            slot = _prefill_slot(i, pending.popleft())
            if slot.done:
                yield [_slot_to_result(slot, state, i, tokenizer)]
                slot.past_key_values = None
                slots[i] = None
            else:
                slots[i] = slot

        # ---- 2. Main batched decode loop ----
        # Build initial steady-state tensors (rebuilt only on rare slot changes).
        active_indices = [i for i, s in enumerate(slots) if s is not None]
        active_list   = [slots[i] for i in active_indices]

        if not active_list:
            return  # Edge: every prompt finished during prefill

        batched_pkv  = _stack_and_pad_kv_caches(active_list)[0]
        batched_mask = _stack_and_pad_attention_masks(active_list)

        _K = len(active_list)
        ones_col = torch.ones((_K, 1), dtype=batched_mask.dtype, device=device)

        # Pre-allocate monitor buffers (over-provisioned, never reallocated in hot-path).
        if monitor is not None:
            vocab_size = model.config.vocab_size
            dummy_logits_buf = torch.zeros(
                (max_concurrent_seqs, vocab_size), dtype=torch.float32, device=device
            )
            dummy_ids_buf = torch.zeros(
                (max_concurrent_seqs, 1), dtype=torch.long, device=device
            )
        else:
            dummy_logits_buf = dummy_ids_buf = None

        state.active_mask.fill_(False)
        for idx in active_indices:
            state.active_mask[idx] = True

        eos_tensor = torch.tensor(eos_id, dtype=torch.long, device=device)

        while any(s is not None for s in slots):
            # --- Hot path: single batched forward for all K active slots ---
            batched_last = torch.cat(
                [s.input_ids[:, -1:] for s in active_list], dim=0
            )  # [K, 1]

            state.active_batch_indices = active_indices

            with torch.no_grad():
                out = model(
                    input_ids=batched_last,
                    attention_mask=batched_mask,
                    past_key_values=batched_pkv,
                    use_cache=True,
                    return_dict=True,
                )

            batched_pkv = out.past_key_values          # HF appends new KV column in-place
            logits_K   = out.logits[:, -1, :]          # [K, V]
            logits_K   = _safe_score_range_clean(logits_K, eos_id)

            # Update StateMonitor once for all active slots.
            if monitor is not None:
                max_L = max(active_list[i].input_ids.shape[1] for i in range(len(active_list)))
                ids_buf = torch.zeros((max_concurrent_seqs, max_L), dtype=torch.long, device=device)
                
                for list_i, slot_i in enumerate(active_indices):
                    dummy_logits_buf[slot_i] = logits_K[list_i]
                    L = active_list[list_i].input_ids.shape[1]
                    ids_buf[slot_i, -L:] = active_list[list_i].input_ids[0, :]
                    
                monitor(ids_buf, dummy_logits_buf)

            # GPU-side EOS / max-len detection (single CPU sync).
            next_tokens = _sample_batch_tokens(logits_K, DO_SAMPLE, TEMPERATURE, TOP_P, TOP_K, MIN_P)  # [K, 1]
            next_flat   = next_tokens.squeeze(1)                         # [K]
            eos_hit     = next_flat.eq(eos_tensor)                       # [K] bool GPU
            n_gen_arr   = torch.tensor(
                [s.n_generated + 1 for s in active_list], dtype=torch.long, device=device
            )
            max_hit   = n_gen_arr.ge(AIME_MAX_TOKENS)
            done_mask = eos_hit | max_hit                                # [K] bool GPU
            has_finished = done_mask.any().item()                        # single sync

            done_list = done_mask.tolist() if has_finished else [False] * _K

            # Append sampled tokens to each slot's running sequence.
            for list_i, slot_i in enumerate(active_indices):
                s   = active_list[list_i]
                nxt = next_tokens[list_i:list_i + 1]    # [1, 1] view
                s.input_ids = torch.cat([s.input_ids, nxt], dim=1)
                s.attention_mask = torch.cat(
                    [s.attention_mask,
                     torch.ones(1, 1, dtype=torch.long, device=device)], dim=1
                )
                s.n_generated += 1
                if has_finished and done_list[list_i]:
                    s.done = True

            batched_mask = torch.cat([batched_mask, ones_col], dim=1)

            # --- Rare path: restructure when \u22651 slot finishes ---
            if has_finished:
                _unpad_and_split_kv_caches(batched_pkv, active_list)

                for list_i, slot_i in enumerate(active_indices):
                    s = active_list[list_i]
                    if s.done:
                        yield [_slot_to_result(s, state, slot_i, tokenizer)]
                        s.past_key_values = None
                        s.input_ids = None
                        s.attention_mask = None
                        state.active_mask[slot_i] = False

                        if pending:
                            new_slot = _prefill_slot(slot_i, pending.popleft())
                            if new_slot.done:
                                yield [_slot_to_result(new_slot, state, slot_i, tokenizer)]
                                new_slot.past_key_values = None
                                slots[slot_i] = None
                            else:
                                slots[slot_i] = new_slot
                        else:
                            slots[slot_i] = None

                # Rebuild steady-state tensors for next step.
                active_indices = [i for i, s in enumerate(slots) if s is not None]
                if not active_indices:
                    break

                active_list   = [slots[i] for i in active_indices]
                batched_pkv   = _stack_and_pad_kv_caches(active_list)[0]
                batched_mask  = _stack_and_pad_attention_masks(active_list)
                _K            = len(active_list)
                ones_col      = torch.ones((_K, 1), dtype=batched_mask.dtype, device=device)

                state.active_mask.fill_(False)
                for idx in active_indices:
                    state.active_mask[idx] = True

    finally:
        if hook_handle is not None:
            hook_handle.remove()


def _cleanup_slot(slot: _Slot):
    """Free GPU tensors held by a finished slot."""
    slot.past_key_values = None
    slot.input_ids = None
    slot.attention_mask = None


# ======================== Full Experiment Pipeline ========================

def run_full_experiment(
    model,
    tokenizer,
    dataset: list[dict],
    dataset_name: str = "AIME",
    modes: list[str] | None = None,
    control_vectors: dict | None = None,
    max_concurrent_seqs: int = MAX_CONCURRENT_SEQS,
    dataset_type: str = "aime",
    results_path: str = None,
):
    """
    Run the full AIME benchmark across all modes.

    Uses the Continuous Batching engine for ALL modes, achieving
    near-100% GPU utilization without cross-slot steering interference.

    Args:
        model: Loaded causal LM.
        tokenizer: Tokenizer.
        dataset: List of dataset problem dicts.
        dataset_name: Human-readable label for the dataset.
        modes: Experiment modes to run (defaults to config).
        control_vectors: Dict with 'purified' and/or 'raw' steering vectors.
        max_concurrent_seqs: Pool size for continuous batching (max simultaneously active sequences).
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

        print(f"  Using CONTINUOUS BATCHING (max_concurrent_seqs={max_concurrent_seqs}) for ALL modes...")
        
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
            max_concurrent_seqs=max_concurrent_seqs
        )

        for batch_results in gen_iterator:
            # Reconstruct the original dataset problems corresponding to these exact results.
            # Shorter generations finish earlier and thus yield order differs from prompt order.
            batch_dataset = [dataset[res["prompt_idx"]] for res in batch_results]
            
            # Submits to queue to be processed asynchronously
            results_queue.put((mode, batch_dataset, batch_results, mode_stats, mode_data))
            
            pbar.update(len(batch_results))
            
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
        max_concurrent_seqs=MAX_CONCURRENT_SEQS,
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
