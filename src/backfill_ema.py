import json
import argparse
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys

# Add src to path to import config
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config

from spherical_injector import spherical_rotate

def calculate_entropy_parallel(model, input_ids_tensor, input_len, control_vector=None, is_continuous=False):
    """
    Perform a single forward pass and calculate entropy for the generated part.
    If is_continuous is True, applies steering intervention during the forward pass.
    """
    handles = []
    if is_continuous and control_vector is not None:
        def parallel_steering_hook(module, args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            # Only steer the generated tokens to affect subsequent distributions
            # The first generated token (at input_len) remains unsteered to match
            # original behavior where steering starts after the prefill phase.
            if hidden.shape[1] > input_len:
                h_to_steer = hidden[:, input_len:-1, :] # Steer tokens that predict NEXT tokens
                v = control_vector.to(device=hidden.device, dtype=hidden.dtype)
                
                # Reshape v for broadcasting if needed
                if v.dim() == 1: v = v.unsqueeze(0).unsqueeze(0)
                elif v.dim() == 2: v = v.unsqueeze(0)
                
                h_steered = spherical_rotate(h_to_steer, v, config.CONTINUOUS_ALPHA)
                
                hidden_new = hidden.clone()
                hidden_new[:, input_len:-1, :] = h_steered
                
                if isinstance(output, tuple):
                    return (hidden_new,) + output[1:]
                return hidden_new
            return output

        # Register hook at target layer
        layer = model.model.layers[config.LAYER_ID]
        handle = layer.register_forward_hook(parallel_steering_hook)
        handles.append(handle)

    try:
        with torch.no_grad():
            outputs = model(input_ids_tensor)
            logits = outputs.logits  # [1, seq_len, vocab_size]
    finally:
        # Always remove hooks
        for h in handles:
            h.remove()
        
    # We want entropy for tokens generated AFTER input_len
    # logits[0, i] is the prediction for the token at input_ids[0, i+1]
    seq_len = input_ids_tensor.shape[1]
    gen_logits = logits[0, input_len-1 : seq_len-1, :] # [gen_len, vocab_size]
    
    # Compute entropy
    probs = F.softmax(gen_logits, dim=-1)
    log_probs = F.log_softmax(gen_logits, dim=-1)
    entropy = -torch.sum(probs * log_probs, dim=-1) # [gen_len]
    
    return entropy.cpu().tolist()

def load_control_vector(vector_dir, device, dtype):
    """Mirror of run_experiment's vector loading logic."""
    from vector_injector import VectorInjector
    try:
        injector = VectorInjector(vector_dir, device=device, model_dtype=dtype)
        if injector.activate("critic", coeff=1.0):
            v = injector.get_normalized_vector()
            # Normalize to unit vector
            v_flat = v.view(-1)
            v_normalized = v_flat / v_flat.norm()
            v_normalized = v_normalized.view(v.shape)
            injector.deactivate()
            return v_normalized
    except Exception as e:
        print(f"⚠️  Failed to load control vector: {e}")
    return None

def backfill_json(json_path, output_path, limit=None):
    print(f"Loading results from {json_path}...")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    print(f"Loading model and tokenizer from {config.MODEL_PATH}...")
    model_dtype = getattr(torch, config.DEFAULT_DTYPE)
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_PATH,
        torch_dtype=model_dtype,
        device_map=config.DEVICE_MAP,
    )
    
    control_vector = load_control_vector(
        config.VECTOR_DIR, 
        device=model.device, 
        dtype=model_dtype
    )
    
    beta = config.EMA_BETA
    
    for group_name in ["Baseline", "Continuous"]:
        if group_name not in data:
            print(f"Group {group_name} not found in JSON. Skipping.")
            continue
            
        problems = data[group_name].get("per_problem", [])
        if not problems:
            continue
            
        print(f"Processing {group_name} group ({len(problems)} problems)...")
        is_continuous = (group_name == "Continuous")
        
        count = 0
        for prob in tqdm(problems):
            if limit and count >= limit:
                break
                
            output_ids = prob.get("output_ids", [])
            input_len = prob.get("input_len", 0)
            
            if not output_ids or input_len == 0:
                continue
                
            # Perform parallel forward pass
            input_ids_tensor = torch.tensor([output_ids], device=model.device)
            try:
                entropy_traj = calculate_entropy_parallel(
                    model, 
                    input_ids_tensor, 
                    input_len, 
                    control_vector=control_vector,
                    is_continuous=is_continuous
                )
                
                # Calculate EMA
                ema_traj = []
                ema = 0.0
                for i, h in enumerate(entropy_traj):
                    if i == 0:
                        ema = h
                    else:
                        ema = beta * h + (1.0 - beta) * ema
                    ema_traj.append(ema)
                
                # Update problem record
                prob["entropy_trajectory"] = entropy_traj
                prob["ema_trajectory"] = ema_traj
                
            except Exception as e:
                print(f"\nError processing problem {prob.get('id')}: {e}")
                
            count += 1
            
        # Update group-level trajectories (using the first one as representative, if exists)
        if problems and "ema_trajectory" in problems[0]:
            data[group_name]["ema_trajectory"] = problems[0]["ema_trajectory"]
            data[group_name]["alpha_trajectory"] = [0.0] * len(problems[0]["ema_trajectory"])
            
    print(f"Saving updated results to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=config.JSON_INDENT, ensure_ascii=False)
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Back-fill entropy EMA trajectories")
    parser.add_argument("--json_path", type=str, required=True, help="Path to input experiment_results.json")
    parser.add_argument("--output_path", type=str, default=None, help="Path to save updated JSON")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of problems per group (for testing)")
    
    args = parser.parse_args()
    
    if args.output_path is None:
        base, ext = os.path.splitext(args.json_path)
        args.output_path = f"{base}_fixed{ext}"
        
    backfill_json(args.json_path, args.output_path, args.limit)
