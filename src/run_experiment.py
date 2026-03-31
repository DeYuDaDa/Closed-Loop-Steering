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

# ======================== Full Experiment Pipeline ========================
# Replaces run_batched_generation as the primary inference engine.
#
# Key invariant: each slot has its own InjectionState, PID, and hook handle,
# so alpha values are 100% isolated — no cross-slot contamination is possible.
# Finished slots are immediately refilled, eliminating the straggler problem.

from dataclasses import dataclass, field


@dataclass
class _Slot:
    """
    One active decoding slot in the continuous-batching pool.

    Lifecycle:
        created  → prefill  → decode loop  → done  → result emitted
    """
    prompt_idx: int                         # Index into the original prompts list
    input_ids: torch.Tensor                 # [1, seq_len]  current full sequence
    attention_mask: torch.Tensor            # [1, seq_len]
    past_key_values: object                 # KV cache (None before prefill)
    input_len: int                          # Length of the original prompt
    n_generated: int = 0                    # Tokens generated so far
    done: bool = False

    # Per-slot closed-loop state (batch_size=1 throughout)
    state: object = None                    # InjectionState
    pid: object = None                      # PIDController | TAEController | None
    monitor: object = None                  # StateMonitor | None
    hook_handle: object = None             # PyTorch forward-hook RemovableHandle | None


def _build_slot_components(
    mode: str,
    control_vector,
    control_vectors: dict,
    term_token_id,
    device: str,
):
    """
    Instantiate InjectionState, PID controller, and StateMonitor for one slot.
    Returns (state, pid, monitor).
    """
    state = InjectionState(batch_size=1, device=device)
    pid = None

    if mode == "Continuous":
        state.intervention_active.fill_(True)
        state.alpha.fill_(CONTINUOUS_ALPHA)
    elif mode == "Continuous_Linear":
        state.intervention_active.fill_(True)
        state.alpha.fill_(CONTINUOUS_LINEAR_ALPHA)
    elif mode in _DYNAMIC_MODES:
        if mode in ("True_TAE", "TAE_Spherical"):
            pid = TAEController(batch_size=1, device=device)
        else:
            pid = PIDController(batch_size=1, device=device)

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





def _slot_to_result(slot: _Slot, tokenizer) -> dict:
    """Convert a finished slot into the result dict expected by run_full_experiment."""
    generated_ids = slot.input_ids[0, slot.input_len:]

    # Strip trailing pad tokens
    if tokenizer.pad_token_id is not None:
        mask = generated_ids != tokenizer.pad_token_id
        generated_ids = generated_ids[mask]

    gen_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    tokens   = [tokenizer.decode([t]).replace("\n", "↵") for t in generated_ids]

    state = slot.state
    has_state = state is not None

    return {
        "text":               gen_text,
        "tokens":             tokens,
        "num_tokens":         len(tokens),
        "output_ids":         slot.input_ids[0].cpu().tolist(),
        "input_len":          slot.input_len,
        "prompt_idx":         slot.prompt_idx,
        "ema_trajectory":     state.ema_trajectory[0]           if has_state else [],
        "alpha_trajectory":   state.alpha_trajectory[0]         if has_state else [],
        "entropy_trajectory": state.entropy_trajectory[0]       if has_state else [],
        "history_hidden":     [],
        "intervention_start": state.intervention_start_step[0]  if has_state else None,
        "intervention_end":   state.intervention_end_step[0]    if has_state else None,
        "convergence":        state.is_converged[0].item()      if has_state else False,
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


def run_continuous_batching_generation(
    model,
    tokenizer,
    prompts: list,
    mode: str,
    control_vectors: dict,
    max_concurrent_seqs: int = MAX_CONCURRENT_SEQS,
) -> "Generator[list[dict], None, None]":
    """
    Continuous-batching inference engine.

    Maintains a pool of `max_concurrent_seqs` active decoding slots.
    When a slot finishes (EOS or max_new_tokens), its result is collected and the
    next waiting prompt immediately fills the freed slot — no straggler waiting.

    Each slot owns an isolated InjectionState + PID + hook handle (batch_size=1),
    ensuring zero cross-slot contamination of alpha values or EMA entropy state.

    Timing contract (identical to model.generate):
        step t-1 → StateMonitor writes alpha[t-1]
        step t   → hook reads alpha[t-1], applies SLERP, produces logits[t]
                 → StateMonitor reads logits[t], writes alpha[t]

    Args:
        model:               Loaded causal LM.
        tokenizer:           Tokenizer.
        prompts:             All formatted prompt message-lists for this experiment.
        mode:                Experiment mode string.
        control_vectors:     Dict with 'purified' / 'raw' steering tensors.
        max_concurrent_seqs: Size of the slot pool (analogous to batch_size).

    Yields:
        list[dict] — one result dict per prompt, yielded individually (list of 1)
                     so the caller's batch_start/batch_end bookkeeping still works.
    """
    device = model.device

    # Resolve control vector for this mode
    if mode in ("True_TAE", "Dynamic_Spherical_No_Manifold"):
        control_vector = control_vectors.get("raw", None)
    else:
        control_vector = control_vectors.get("purified", None)

    # Resolve </think> token for ThinkBrake
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

    # ---- One shared hook, proxy-dispatched to the current slot ----
    # Root cause of cross-slot bug: if each slot registers its own hook on the
    # SAME model layer, all K hooks fire on EVERY forward pass, effectively
    # applying K× the intended steering to each slot's hidden states.
    #
    # Fix: register exactly ONE hook whose state pointer is swapped to the
    # currently-decoding slot before each model() call.  All per-slot state
    # objects (InjectionState / PID / EMA) remain perfectly isolated.
    class _StateProxy:
        """Transparent proxy that forwards attribute access to the active slot's state."""
        def __init__(self):
            object.__setattr__(self, '_current', None)
        def _set(self, state):
            object.__setattr__(self, '_current', state)
        def __getattr__(self, name):
            return getattr(object.__getattribute__(self, '_current'), name)
        def __setattr__(self, name, value):
            setattr(object.__getattribute__(self, '_current'), name, value)

    state_proxy = _StateProxy()
    shared_hook_handle = None
    if control_vector is not None and mode in _HOOK_MODES:
        hook_fn, _ = create_steering_hook(
            state=state_proxy,
            control_vector=control_vector,
            mode=mode,
            continuous_alpha=CONTINUOUS_ALPHA,
            continuous_linear_alpha=CONTINUOUS_LINEAR_ALPHA,
            capture_hidden_states=False,
        )
        layer = model.model.layers[LAYER_ID]
        shared_hook_handle = layer.register_forward_hook(hook_fn)

    # ---- Queue of pending prompts ----
    from collections import deque
    pending = deque(range(len(prompts)))
    active_slots: list[_Slot] = []

    def _prefill_prompt(prompt_idx: int) -> _Slot:
        """Tokenize, prefill, return a ready-to-decode Slot.
        Note: NO per-slot hook registration — a single shared hook (state_proxy)
        handles all steering to eliminate cross-slot interference.
        """
        p = prompts[prompt_idx]
        try:
            text = tokenizer.apply_chat_template(
                p, tokenize=False, add_generation_prompt=True,
                enable_thinking=ENABLE_THINKING,
            )
        except TypeError:
            # Fallback for older tokenizers that don't support enable_thinking
            text = tokenizer.apply_chat_template(
                p, tokenize=False, add_generation_prompt=True,
            )
        tokenizer.padding_side = "left"
        enc = tokenizer(text, return_tensors="pt").to(device)
        tokenizer.padding_side = "right"

        input_len = enc.input_ids.shape[1]

        # Build components for this slot
        state, pid, monitor = _build_slot_components(
            mode, control_vector, control_vectors, term_token_id, device
        )

        slot = _Slot(
            prompt_idx=prompt_idx,
            input_ids=enc.input_ids,
            attention_mask=enc.attention_mask,
            past_key_values=None,
            input_len=input_len,
            state=state,
            pid=pid,
            monitor=monitor,
        )
        # No hook_handle per slot — shared hook is managed by the outer function.
        slot.hook_handle = None

        # Point shared hook at this slot's state for the prefill forward.
        # (Hook returns early for seq_len > 1, so this is safe but harmless.)
        state_proxy._set(state)

        # ---- Prefill: forward on the full prompt ----
        with torch.no_grad():
            out = model(
                input_ids=slot.input_ids,
                attention_mask=slot.attention_mask,
                use_cache=True,
                return_dict=True,
            )
        slot.past_key_values = out.past_key_values

        # Grab first-token logits from the prefill and call StateMonitor now,
        # so alpha is set before step 0's decode forward.
        first_logits = out.logits[:, -1:, :]   # [1, 1, V]
        first_logits_2d = first_logits[:, 0, :]  # [1, V]
        first_logits_2d = _safe_score_range_clean(first_logits_2d, eos_id)

        if slot.monitor is not None:
            # current_ids needed by monitor — pass full input_ids for context
            slot.monitor(slot.input_ids, first_logits_2d)

        # Determine first token by sampling from the prefill logits
        next_token = _sample_token(first_logits_2d, do_sample=DO_SAMPLE,
                                   temperature=TEMPERATURE, top_p=TOP_P)
        # Append first generated token to the running sequence
        slot.input_ids = torch.cat([slot.input_ids, next_token], dim=1)
        slot.attention_mask = torch.ones(
            1, slot.input_ids.shape[1], dtype=torch.long, device=device
        )
        slot.n_generated = 1

        if _is_done(next_token, eos_id, slot.n_generated):
            slot.done = True

        return slot

    def _sample_token(logits_2d: torch.Tensor, do_sample: bool,
                      temperature: float, top_p: float) -> torch.Tensor:
        """Sample next token from [1, V] logits. Returns [1, 1] tensor."""
        if do_sample:
            # Temperature scaling
            logits_scaled = logits_2d / max(temperature, 1e-6)
            # Top-p (nucleus) sampling
            sorted_logits, sorted_idx = torch.sort(logits_scaled, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            # Remove tokens beyond the top-p threshold
            sorted_remove = cumulative_probs - torch.softmax(sorted_logits, dim=-1) > top_p
            sorted_logits[sorted_remove] = -float("inf")
            # Scatter back
            logits_final = torch.full_like(logits_scaled, -float("inf"))
            logits_final.scatter_(1, sorted_idx, sorted_logits)
            probs = torch.softmax(logits_final, dim=-1)
            # multinomial on [1, V] with num_samples=1 → [1, 1]
            next_tok = torch.multinomial(probs, num_samples=1)
        else:
            # argmax on [1, V] with keepdim=True → [1, 1]
            next_tok = logits_2d.argmax(dim=-1, keepdim=True)
        return next_tok  # [1, 1]


    def _is_done(next_token: torch.Tensor, eos_id: int, n_generated: int) -> bool:
        tok_val = next_token.view(-1)[0].item()
        return (tok_val == eos_id) or (n_generated >= AIME_MAX_TOKENS)

    # ---- Fill initial slot pool ----
    n_fill = min(max_concurrent_seqs, len(pending))
    for _ in range(n_fill):
        slot = _prefill_prompt(pending.popleft())
        if slot.done:
            yield [_slot_to_result(slot, tokenizer)]
            _cleanup_slot(slot)
            # Immediately fill from pending if available
            if pending:
                slot = _prefill_prompt(pending.popleft())
                if not slot.done:
                    active_slots.append(slot)
                else:
                    yield [_slot_to_result(slot, tokenizer)]
                    _cleanup_slot(slot)
        else:
            active_slots.append(slot)

    # ---- Main decode loop ----
    try:
        while active_slots:
            next_active = []
            for slot in active_slots:
                # -- One decode step for this slot --
                # Point shared hook at THIS slot's state BEFORE the forward pass.
                # This ensures only slot.state.alpha is applied — no cross-contamination.
                state_proxy._set(slot.state)

                last_token = slot.input_ids[:, -1:]  # [1, 1]
                with torch.no_grad():
                    out = model(
                        input_ids=last_token,
                        attention_mask=slot.attention_mask,
                        past_key_values=slot.past_key_values,
                        use_cache=True,
                        return_dict=True,
                    )
                slot.past_key_values = out.past_key_values

                logits_2d = out.logits[:, -1, :]  # [1, V]
                logits_2d = _safe_score_range_clean(logits_2d, eos_id)

                # Update StateMonitor AFTER this forward so alpha is ready for NEXT step
                if slot.monitor is not None:
                    slot.monitor(slot.input_ids, logits_2d)

                # Sample next token
                next_token = _sample_token(logits_2d, do_sample=DO_SAMPLE,
                                           temperature=TEMPERATURE, top_p=TOP_P)
                slot.n_generated += 1

                # Append to running sequence
                slot.input_ids = torch.cat([slot.input_ids, next_token], dim=1)
                slot.attention_mask = torch.ones(
                    1, slot.input_ids.shape[1], dtype=torch.long, device=device
                )

                if _is_done(next_token, eos_id, slot.n_generated):
                    slot.done = True
                    # Emit result
                    yield [_slot_to_result(slot, tokenizer)]
                    _cleanup_slot(slot)

                    # Immediately fill with next pending prompt
                    if pending:
                        new_slot = _prefill_prompt(pending.popleft())
                        if new_slot.done:
                            yield [_slot_to_result(new_slot, tokenizer)]
                            _cleanup_slot(new_slot)
                        else:
                            next_active.append(new_slot)
                    # If no pending left, slot just disappears (pool shrinks)
                else:
                    next_active.append(slot)

            active_slots = next_active
    finally:
        # Guaranteed cleanup: remove the shared hook even if an exception occurred
        if shared_hook_handle is not None:
            shared_hook_handle.remove()


def _cleanup_slot(slot: _Slot):
    """Free GPU memory for a finished slot.
    No hook removal needed — slots no longer carry per-slot hooks.
    """
    # Free KV cache (can be very large for long sequences)
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
