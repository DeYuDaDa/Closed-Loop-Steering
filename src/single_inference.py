import torch
import gc
from config import (
    AIME_MAX_TOKENS,
    DO_SAMPLE,
    TEMPERATURE,
    TOP_P,
    TOP_K,
    MIN_P,
    ENABLE_THINKING,
    LAYER_ID,
    CONTINUOUS_ALPHA,
    CONTINUOUS_LINEAR_ALPHA,
)
from spherical_injector import create_steering_hook, spherical_rotate
from dataclasses import dataclass
from typing import List

@dataclass
class SuspendedTask:
    prompt_idx: int
    prompt_len: int
    input_ids: torch.Tensor
    task_state: dict
    is_done: bool = False

def create_batched_replay_hook(
    alpha_trajectories: List[List[float]],
    max_seq_len: int,
    device: torch.device,
    control_vector: torch.Tensor,
):
    """
    Creates a prefill replay hook that meticulously re-applies historical alpha trajectory
    compensations perfectly aligned with Left-Padded input_ids logic to flawlessly restore KV cache.
    """
    batch_size = len(alpha_trajectories)
    dtype = control_vector.dtype
    alpha_tensor = torch.zeros(batch_size, max_seq_len, device=device, dtype=dtype)
    
    for i, traj in enumerate(alpha_trajectories):
        gen_len = len(traj)
        if gen_len > 0:
            # Under left-padding, all valid tokens align to max_seq_len at the right edge
            alpha_tensor[i, max_seq_len - gen_len : max_seq_len] = torch.tensor(
                traj, device=device, dtype=dtype
            )
            
    alpha_mask = alpha_tensor.unsqueeze(-1)
    
    def replay_hook(module, args, output):
        hidden = output[0] if isinstance(output, tuple) else output
        seq_len = hidden.shape[1]
        
        # Guard: Replay Hook only actively alters during giant batched prefill
        if seq_len == 1:
            return output
            
        v_expanded = control_vector.expand(hidden.shape[0], hidden.shape[1], -1)
        is_active = (alpha_mask > 0).squeeze(-1)
        
        if is_active.any():
            h_new = spherical_rotate(hidden, v_expanded, alpha_mask)
            hidden = torch.where(is_active.unsqueeze(-1), h_new, hidden)
            
        if isinstance(output, tuple):
            return (hidden,) + output[1:]
        return hidden
        
    return replay_hook

def run_isolated_batch_inference(
    model,
    tokenizer,
    prompts: list,
    mode: str,
    control_vectors: dict,
    batch_size: int = 1,
):
    """
    Lockstep Chunked Batching (Stateless Re-Prefill Batching with Trajectory Replay).
    """
    import run_experiment  # dynamic import to avoid circular dependency
    import gc
    
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
        else:
            term_ids = tokenizer.encode("<channel|>", add_special_tokens=False)
            if term_ids:
                term_token_id = term_ids[-1]
    except Exception:
        pass

    eos_id = tokenizer.eos_token_id
    if isinstance(eos_id, list):
        eos_id = eos_id[0]

    active_queue = []
    
    # 1. Initialization: Create all SuspendedTasks initially
    for prompt_idx, p in enumerate(prompts):
        try:
            text = tokenizer.apply_chat_template(
                p, tokenize=False, add_generation_prompt=True,
                enable_thinking=ENABLE_THINKING,
            )
        except TypeError:
            text = tokenizer.apply_chat_template(
                p, tokenize=False, add_generation_prompt=True,
            )
            
        # Keep on CPU — only pull to GPU when actively batched
        enc = tokenizer(text, return_tensors="pt")
        input_len = enc.input_ids.shape[1]
        
        task = SuspendedTask(
            prompt_idx=prompt_idx,
            prompt_len=input_len,
            input_ids=enc.input_ids,
            task_state={
                "alpha_trajectory": [],
                "ema_trajectory": [],
                "entropy_trajectory": [],
                "intervention_start_step": None,
                "intervention_end_step": None,
                "margin": float("inf"),
                "is_converged": False,
                "intervention_active": False,
                "step_count": 0,
                "prev_error": 0.0,
                "integral": 0.0,
            }
        )
        active_queue.append(task)

    chunk_decode_steps = 2048

    # 2. Outer Orchestration Loop
    while len(active_queue) > 0:
        # Determine maximum sequence length in the current queue to set safe batch boundaries.
        # CRITICAL: use PEAK length (current + upcoming decode steps) to avoid OOM at end-of-chunk.
        max_L = max(t.input_ids.shape[1] for t in active_queue)
        peak_L = max_L + chunk_decode_steps  # Peak KV cache size after this chunk
        
        # Thresholds derived from: bs × peak_L × 0.144 MB/tok < 13 GB (31.47 - 16.4 model - 2.0 overhead)
        if peak_L < 4096:
            bs = 16
        elif peak_L < 8192:
            bs = 8
        elif peak_L < 16384:
            bs = 4
        else:
            bs = 2
            
        # If user passed a max batch_size for some reason
        bs = min(bs, batch_size) if batch_size > 1 else bs
            
        # Select batch
        batch_tasks = active_queue[:bs]
        active_queue = active_queue[bs:]
        actual_bs = len(batch_tasks)
        
        # Compute padding for the batch — move to GPU NOW for computation
        max_batch_seq_len = max(t.input_ids.shape[1] for t in batch_tasks)
        
        batched_input_ids = []
        batched_attention_mask = []
        
        for t in batch_tasks:
            seq_len = t.input_ids.shape[1]
            t_ids = t.input_ids.to(device)  # Pull from CPU → GPU here
            pad_len = max_batch_seq_len - seq_len
            
            if pad_len > 0:
                pad_tensor = torch.full((1, pad_len), tokenizer.pad_token_id, dtype=torch.long, device=device)
                b_id = torch.cat([pad_tensor, t_ids], dim=1)
                
                mask_pad = torch.zeros((1, pad_len), dtype=torch.long, device=device)
                mask_data = torch.ones((1, seq_len), dtype=torch.long, device=device)
                b_mask = torch.cat([mask_pad, mask_data], dim=1)
            else:
                b_id = t_ids
                b_mask = torch.ones((1, seq_len), dtype=torch.long, device=device)
                
            batched_input_ids.append(b_id)
            batched_attention_mask.append(b_mask)
            
        input_ids = torch.cat(batched_input_ids, dim=0)
        attention_mask = torch.cat(batched_attention_mask, dim=0)
        
        # 3. Instantiate cleanly Global Components
        state, pid, monitor = run_experiment._build_global_components(
            mode, term_token_id, device, batch_size=actual_bs
        )
        state.active_mask.fill_(True)
        state.active_batch_indices = list(range(actual_bs))
        
        # Restore Task State to Global InjectionState & PID
        for i, t in enumerate(batch_tasks):
            ts = t.task_state
            state.margin[i] = ts["margin"]
            state.is_converged[i] = ts["is_converged"]
            state.intervention_active[i] = ts["intervention_active"]
            state.step_count[i] = ts["step_count"]
            
            state.alpha_trajectory[i] = ts["alpha_trajectory"].copy()
            state.ema_trajectory[i] = ts["ema_trajectory"].copy()
            state.entropy_trajectory[i] = ts["entropy_trajectory"].copy()
            
            state.intervention_start_step[i] = ts["intervention_start_step"]
            state.intervention_end_step[i] = ts["intervention_end_step"]
            
            if len(ts["ema_trajectory"]) > 0:
                state.ema_entropy[i] = ts["ema_trajectory"][-1]
            else:
                state.ema_entropy[i] = 0.0
                
            if pid is not None:
                if hasattr(pid, "integral"):
                    pid.integral[i] = ts["integral"]
                if hasattr(pid, "prev_error"):
                    pid.prev_error[i] = ts["prev_error"]
                # For PID to smoothly continue, it should NOT be marked as first step if it has history
                if hasattr(pid, "is_first_step"):
                    pid.is_first_step[i] = (ts["step_count"] == 0)
                    
        # Construct and attach Batched Replay Hook for massive prefill
        replay_hook_handle = None
        hook_handle = None
        layer = model.model.layers[LAYER_ID]
        
        if control_vector is not None and mode in run_experiment._HOOK_MODES:
            alpha_trajs = [t.task_state["alpha_trajectory"] for t in batch_tasks]
            replay_hook = create_batched_replay_hook(
                alpha_trajs, max_batch_seq_len, device, control_vector
            )
            replay_hook_handle = layer.register_forward_hook(replay_hook)
            
        n_generated = 0
        done_mask = torch.zeros(actual_bs, dtype=torch.bool, device=device)
        past_key_values = None
        
        try:
            # --- 4. Massive Mathematical Prefill (KV Cache Restoration via Replay) ---
            # Bypass `model.forward` (which computes logits for the full massive sequence causing 7GB+ OOM)
            # Instead directly compute backbone hidden_states and past_key_values, then manually project the final token.
            with torch.no_grad():
                out = model.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=True,
                    return_dict=True,
                )
                last_hidden = out.last_hidden_state[:, -1:, :]  # [B, 1, hidden_dim]
                logits_1 = model.lm_head(last_hidden).squeeze(1)  # [B, vocab_size]
                logits_1 = run_experiment._safe_score_range_clean(logits_1, eos_id)
                past_key_values = out.past_key_values
            # Explicitly release the large prefill output (last_hidden_state is huge)
            del out, last_hidden
            if replay_hook_handle is not None:
                replay_hook_handle.remove()
                replay_hook_handle = None
                
            if control_vector is not None and mode in run_experiment._HOOK_MODES:
                hook_fn, _ = create_steering_hook(
                    state=state,
                    control_vector=control_vector,
                    mode=mode,
                    continuous_alpha=CONTINUOUS_ALPHA,
                    continuous_linear_alpha=CONTINUOUS_LINEAR_ALPHA,
                    capture_hidden_states=False,
                )
                hook_handle = layer.register_forward_hook(hook_fn)
            
            if monitor is not None:
                # Strip left-padding before passing to monitor.
                # The n-gram repetition detector uses token identity; padding zeros must not
                # be seen as repeated content or they silently trigger trigger_perturbation.
                # Use attention_mask to extract only real tokens per sequence.
                unpadded_for_monitor = [
                    input_ids[j, attention_mask[j].bool()].unsqueeze(0)
                    for j in range(actual_bs)
                ]
                # For monitor, all must be same length; use longest real token count
                max_real = max(u.shape[1] for u in unpadded_for_monitor)
                monitor_ids = torch.cat([
                    torch.cat([torch.full((1, max_real - u.shape[1]), tokenizer.pad_token_id,
                                         dtype=torch.long, device=device), u], dim=1)
                    for u in unpadded_for_monitor
                ], dim=0)
                monitor(monitor_ids, logits_1)
                
            next_tok = run_experiment._sample_batch_tokens(logits_1, DO_SAMPLE, TEMPERATURE, TOP_P, TOP_K, MIN_P)
            next_tok = torch.where(done_mask.unsqueeze(1), torch.full_like(next_tok, eos_id), next_tok)
            done_mask |= (next_tok.squeeze(1) == eos_id)

            input_ids = torch.cat([input_ids, next_tok], dim=1)
            attention_mask = torch.cat(
                [attention_mask, torch.ones(actual_bs, 1, dtype=torch.long, device=device)], dim=1
            )
            n_generated += 1
            
            # --- 5. Forward Autoregressive Decoding Loop ---
            while not done_mask.all() and n_generated < chunk_decode_steps and input_ids.shape[1] < AIME_MAX_TOKENS:
                with torch.no_grad():
                    out = model(
                        input_ids=input_ids[:, -1:],
                        attention_mask=attention_mask,
                        past_key_values=past_key_values,
                        use_cache=True,
                        return_dict=True,
                    )
                past_key_values = out.past_key_values
                logits_1 = run_experiment._safe_score_range_clean(out.logits[:, -1, :], eos_id)
                
                state.active_mask = ~done_mask
                if monitor is not None:
                    # Pass only real (non-padded) token history to avoid false n-gram hits on pad tokens.
                    unpadded_for_monitor = [
                        input_ids[j, attention_mask[j].bool()].unsqueeze(0)
                        for j in range(actual_bs)
                    ]
                    max_real = max(u.shape[1] for u in unpadded_for_monitor)
                    monitor_ids = torch.cat([
                        torch.cat([torch.full((1, max_real - u.shape[1]), tokenizer.pad_token_id,
                                             dtype=torch.long, device=device), u], dim=1)
                        for u in unpadded_for_monitor
                    ], dim=0)
                    monitor(monitor_ids, logits_1)
                
                next_tok = run_experiment._sample_batch_tokens(logits_1, DO_SAMPLE, TEMPERATURE, TOP_P, TOP_K, MIN_P)
                next_tok = torch.where(done_mask.unsqueeze(1), torch.full_like(next_tok, eos_id), next_tok)
                done_mask |= (next_tok.squeeze(1) == eos_id)

                input_ids = torch.cat([input_ids, next_tok], dim=1)
                attention_mask = torch.cat(
                    [attention_mask, torch.ones(actual_bs, 1, dtype=torch.long, device=device)], dim=1
                )
                n_generated += 1

        finally:
            if hook_handle is not None:
                hook_handle.remove()
            if replay_hook_handle is not None:
                replay_hook_handle.remove()

        # --- 6. Post-Decode Deconstruction and Suspension ---
        chunk_results = []
        for i, t in enumerate(batch_tasks):
            # Strip Left Padding to get pure sequence back!
            mask_i = attention_mask[i] == 1
            pure_input_ids = input_ids[i, mask_i].unsqueeze(0)
            
            is_eos_reached = done_mask[i].item()
            is_absolute_limit = pure_input_ids.shape[1] >= AIME_MAX_TOKENS
            
            if is_eos_reached or is_absolute_limit:
                # Finished, flush out result completely
                slot = run_experiment._Slot(
                    prompt_idx=t.prompt_idx,
                    input_ids=pure_input_ids,
                    attention_mask=None,
                    past_key_values=None,
                    input_len=t.prompt_len,
                    done=True,
                )
                chunk_results.append(run_experiment._slot_to_result(slot, state, i, tokenizer))
            else:
                # Suspend state and put back in queue
                ts = t.task_state
                ts["margin"] = state.margin[i]
                ts["is_converged"] = state.is_converged[i]
                ts["intervention_active"] = state.intervention_active[i]
                ts["step_count"] = state.step_count[i]
                
                ts["alpha_trajectory"] = state.alpha_trajectory[i].copy()
                ts["ema_trajectory"] = state.ema_trajectory[i].copy()
                ts["entropy_trajectory"] = state.entropy_trajectory[i].copy()
                
                ts["intervention_start_step"] = state.intervention_start_step[i]
                ts["intervention_end_step"] = state.intervention_end_step[i]
                    
                if pid is not None:
                    if hasattr(pid, "integral"):
                        ts["integral"] = pid.integral[i]
                    if hasattr(pid, "prev_error"):
                        ts["prev_error"] = pid.prev_error[i]
                
                t.input_ids = pure_input_ids.cpu()  # Park back to CPU until next chunk
                active_queue.append(t)
                
        try:
            del out, last_hidden, logits_1, next_tok, pure_input_ids
        except NameError:
            pass
        del state, pid, monitor, batched_input_ids, batched_attention_mask, input_ids, attention_mask, past_key_values, batch_tasks
        torch.cuda.empty_cache()
        gc.collect()

        if len(chunk_results) > 0:
            yield chunk_results

def run_single_inference(
    model,
    tokenizer,
    prompts: list,
    mode: str,
    control_vectors: dict,
):
    """
    Absolutely isolated sequential inference.
    Processes one prompt at a time, creating fresh component instances for each.
    Uses the identical manual decoding loop as the batched version to guarantee parity.
    """
    import run_experiment  # dynamic import to avoid circular dependency
    import gc
    
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
        else:
            term_ids = tokenizer.encode("<channel|>", add_special_tokens=False)
            if term_ids:
                term_token_id = term_ids[-1]
    except Exception:
        pass

    eos_id = tokenizer.eos_token_id
    if isinstance(eos_id, list):
        eos_id = eos_id[0]

    for prompt_idx, p in enumerate(prompts):
        # 1. Instantiate cleanly
        state, pid, monitor = run_experiment._build_global_components(
            mode, term_token_id, device, batch_size=1
        )
        state.active_mask[0] = True
        state.active_batch_indices = [0]
        
        hook_handle = None
        if control_vector is not None and mode in run_experiment._HOOK_MODES:
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
        
        input_ids = enc.input_ids
        attention_mask = enc.attention_mask
        input_len = input_ids.shape[1]
        n_generated = 0
        past_key_values = None
        
        try:
            # First Forward (Prefill)
            with torch.no_grad():
                out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=True,
                    return_dict=True,
                )
                
            logits_1 = run_experiment._safe_score_range_clean(out.logits[:, -1, :], eos_id)
            past_key_values = out.past_key_values
            
            if monitor is not None:
                monitor(input_ids, logits_1)
                
            next_tok = run_experiment._sample_batch_tokens(logits_1, DO_SAMPLE, TEMPERATURE, TOP_P, TOP_K, MIN_P)
            input_ids = torch.cat([input_ids, next_tok], dim=1)
            attention_mask = torch.cat(
                [attention_mask, torch.ones(1, 1, dtype=torch.long, device=device)], dim=1
            )
            n_generated += 1
            
            # Autoregressive Decode Loop
            while next_tok.item() != eos_id and n_generated < AIME_MAX_TOKENS:
                with torch.no_grad():
                    out = model(
                        input_ids=input_ids[:, -1:],
                        attention_mask=attention_mask,
                        past_key_values=past_key_values,
                        use_cache=True,
                        return_dict=True,
                    )
                past_key_values = out.past_key_values
                logits_1 = run_experiment._safe_score_range_clean(out.logits[:, -1, :], eos_id)
                
                if monitor is not None:
                    monitor(input_ids, logits_1)
                
                next_tok = run_experiment._sample_batch_tokens(logits_1, DO_SAMPLE, TEMPERATURE, TOP_P, TOP_K, MIN_P)
                input_ids = torch.cat([input_ids, next_tok], dim=1)
                attention_mask = torch.cat(
                    [attention_mask, torch.ones(1, 1, dtype=torch.long, device=device)], dim=1
                )
                n_generated += 1

        finally:
            if hook_handle is not None:
                hook_handle.remove()

        slot = run_experiment._Slot(
            prompt_idx=prompt_idx,
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=None,
            input_len=input_len,
            done=True,
        )
        
        result_dict = run_experiment._slot_to_result(slot, state, 0, tokenizer)
        
        # Absolute isolation per request - free GPU memory BEFORE yield pauses the function
        try:
            del out, logits_1, next_tok, text
        except NameError:
            pass
        del state, pid, monitor, enc, input_ids, attention_mask, past_key_values, slot

        torch.cuda.empty_cache()
        gc.collect()

        yield [result_dict]
