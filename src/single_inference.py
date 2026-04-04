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
from spherical_injector import create_steering_hook

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
        
        yield [run_experiment._slot_to_result(slot, state, 0, tokenizer)]
        
        # Absolute isolation per request
        del state, pid, monitor, enc, input_ids, attention_mask, past_key_values, slot
        torch.cuda.empty_cache()
        gc.collect()
