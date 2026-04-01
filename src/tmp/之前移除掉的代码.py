# 这是之前你清理程序入口时移除掉的大段函数，我不知道哪些还有用，所以我都给找回来了

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
        tokenizer.padding_side = 'left'
        inputs = tokenizer(
            formatted_prompts,
            padding=True,
            return_tensors="pt",
        ).to(model.device)
        
        # Reset to right padding
        tokenizer.padding_side = 'right'

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

from dataclasses import dataclass

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
                k = torch.nn.functional.pad(k, (0, 0, pad_left, 0), value=0.0)
                v = torch.nn.functional.pad(v, (0, 0, pad_left, 0), value=0.0)
            layer_k.append(k)
            layer_v.append(v)
        batched_pkv.append((torch.cat(layer_k, dim=0), torch.cat(layer_v, dim=0)))
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
        "prompt_idx":         slot.prompt_idx,
        "ema_trajectory":     state.ema_trajectory[slot_idx] if has_state else [],
        "alpha_trajectory":   state.alpha_trajectory[slot_idx] if has_state else [],
        "entropy_trajectory": state.entropy_trajectory[slot_idx] if has_state else [],
        "history_hidden":     [],
        "intervention_start": state.intervention_start_step[slot_idx] if has_state else None,
        "intervention_end":   state.intervention_end_step[slot_idx] if has_state else None,
        "convergence":        state.is_converged[slot_idx].item() if has_state else False,
    }

def _safe_score_range_clean(scores: torch.Tensor, eos_id: int) -> torch.Tensor:
    torch.nan_to_num_(scores, nan=-SAFE_SCORE_RANGE, posinf=SAFE_SCORE_RANGE, neginf=-SAFE_SCORE_RANGE)
    max_scores, _ = scores.max(dim=-1)
    collapsed = max_scores <= (-SAFE_SCORE_RANGE + 1.0)
    if collapsed.any():
        scores[collapsed, :] = -SAFE_SCORE_RANGE
        scores[collapsed, eos_id] = SAFE_SCORE_RANGE
    return scores

# NOTE: This first definition was the Strategy-C batched implementation.
# It is SUPERSEDED by the improved per-slot engine below (same function name,
# Python uses the later definition). Kept for reference ONLY — never executed.
def _run_batched_strategy_c_deprecated(
    model,
    tokenizer,
    prompts: list,
    mode: str,
    control_vectors: dict,
    max_concurrent_seqs: int = MAX_CONCURRENT_SEQS,
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
                continuous_alpha=CONTINUOUS_ALPHA,
                continuous_linear_alpha=CONTINUOUS_LINEAR_ALPHA,
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
                    state.alpha[idx] = CONTINUOUS_ALPHA
                elif mode == "Continuous_Linear":
                    state.alpha[idx] = CONTINUOUS_LINEAR_ALPHA
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
                out = model(input_ids=enc.input_ids, attention_mask=enc.attention_mask, use_cache=True, return_dict=True)
                
            first_logits = _safe_score_range_clean(out.logits[:, -1, :], eos_id)
            
            slot = _Slot(
                prompt_idx=prompt_idx,
                input_ids=enc.input_ids,
                attention_mask=enc.attention_mask,
                past_key_values=out.past_key_values,
                input_len=enc.input_ids.shape[1],
            )

            if monitor is not None:
                dummy_ids = torch.zeros((max_concurrent_seqs, enc.input_ids.shape[1]), dtype=torch.long, device=device)
                dummy_ids[slot_idx] = enc.input_ids[0]
                dummy_logits = torch.zeros((max_concurrent_seqs, first_logits.shape[1]), dtype=first_logits.dtype, device=device)
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
            if (next_tok.item() == eos_id) or (slot.n_generated >= AIME_MAX_TOKENS):
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

        # [P0 Fix 2]: Pre-allocate a persistent ones column and dummy buffers to avoid
        # per-step CUDA memory allocation.  Buffers are resized only when slot count changes
        # (rare), so the hot-path below does zero dynamic allocations.
        _K = len(active_list)
        ones_col = torch.ones((_K, 1), dtype=batched_mask.dtype, device=device)

        # [P0 Fix 1]: Pre-allocate dummy logits buffer (reused every step).
        # Over-provisioned to max_concurrent_seqs; no reallocation ever needed.
        # NOTE: dummy_ids is NOT allocated because StateMonitor never reads input_ids;
        #       we pass a zero-row placeholder tensor of shape [max_concurrent_seqs, 1].
        if monitor is not None:
            vocab_size = model.config.vocab_size
            dummy_logits_buf = torch.zeros((max_concurrent_seqs, vocab_size),
                                           dtype=torch.float32, device=device)
            # Stationary placeholder — monitor interface requires input_ids but ignores it.
            dummy_ids_placeholder = torch.zeros((max_concurrent_seqs, 1),
                                                dtype=torch.long, device=device)
        else:
            dummy_logits_buf = dummy_ids_placeholder = None

        # Pre-compute active_mask for monitor; updated only when slots change (rare).
        if state is not None:
            state.active_mask.fill_(False)
            for idx in active_indices:
                state.active_mask[idx] = True

        eos_id_tensor = torch.tensor(eos_id, dtype=torch.long, device=device)

        while any(s is not None for s in slots):
            # Extract last tokens for current step (O(K), no GPU alloc)
            batched_last_tokens = torch.cat([s.input_ids[:, -1:] for s in active_list], dim=0) # [K, 1]
            
            if state is not None:
                state.active_batch_indices = active_indices
                
            with torch.no_grad():
                out = model(
                    input_ids=batched_last_tokens,
                    attention_mask=batched_mask,
                    past_key_values=batched_pkv,
                    use_cache=True,
                    return_dict=True,
                )
                
            # [Strategy C]: In-place reception — HF appends the new KV column at C++ level
            batched_pkv = out.past_key_values
            
            logits_2d = _safe_score_range_clean(out.logits[:, -1, :], eos_id) # [K, V]
            
            if monitor is not None:
                # [P1 Fix]: Write only active-slot logits — no zero_() of the full buffer.
                # Inactive slots are filtered by state.active_mask, so stale values are harmless.
                for list_i, slot_i in enumerate(active_indices):
                    dummy_logits_buf[slot_i] = logits_2d[list_i]
                # input_ids placeholder is never read by StateMonitor — pass the stub.
                monitor(dummy_ids_placeholder, dummy_logits_buf)
                
            next_tokens = _sample_batch_tokens(logits_2d, DO_SAMPLE, TEMPERATURE, TOP_P) # [K, 1]

            # [P0 Fix 2]: GPU-side EOS detection — avoids K per-slot CUDA→CPU syncs.
            # next_tokens is [K, 1]; squeeze to [K] for comparison.
            next_flat = next_tokens.squeeze(1)                          # [K], stays on GPU
            eos_hit   = next_flat.eq(eos_id_tensor)                     # [K] bool, on GPU
            n_gen_arr = torch.tensor([s.n_generated + 1 for s in active_list],
                                     dtype=torch.long, device=device)   # [K]
            max_hit   = n_gen_arr.ge(AIME_MAX_TOKENS)                   # [K] bool, on GPU
            done_mask = eos_hit | max_hit                                # [K] bool, on GPU

            # Single sync: only pay CUDA→CPU cost once per step (not K times)
            has_finished = done_mask.any().item()

            # Update per-slot bookkeeping (no .item() calls here)
            done_list_cpu = done_mask.tolist() if has_finished else [False] * _K
            for list_i, slot_i in enumerate(active_indices):
                s = active_list[list_i]
                nxt = next_tokens[list_i:list_i+1]   # [1, 1] view, no copy
                s.input_ids = torch.cat([s.input_ids, nxt], dim=1)
                s.attention_mask = torch.cat(
                    [s.attention_mask, torch.ones(1, 1, dtype=torch.long, device=device)], dim=1
                )
                s.n_generated += 1
                if has_finished and done_list_cpu[list_i]:
                    s.done = True

            # Append new-token column to the shared mask (reuse pre-allocated ones_col)
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
                # Refresh pre-allocated hot-path tensors to match new active count
                _K = len(active_list)
                ones_col = torch.ones((_K, 1), dtype=batched_mask.dtype, device=device)
                # Sync active_mask to the new slot layout (no save/restore needed in hot-path)
                if state is not None:
                    state.active_mask.fill_(False)
                    for idx in active_indices:
                        state.active_mask[idx] = True
    finally:
        if hook_handle is not None:
            hook_handle.remove()

