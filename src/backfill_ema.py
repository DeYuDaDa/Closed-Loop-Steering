"""
backfill_ema.py
===============
Back-fill missing `entropy_trajectory` and `ema_trajectory` fields for ALL
experiment groups (including Continuous_Linear, Dynamic_Linear, and the four
ablation modes) by performing a single parallel forward pass on the already
generated token IDs.

Injection logic per mode:
  Baseline                       → no hook
  Continuous                     → SLERP with constant CONTINUOUS_ALPHA
  Continuous_Linear              → linear inject with constant CONTINUOUS_LINEAR_ALPHA
  Dynamic_Spherical              → SLERP with saved alpha_trajectory (purified vector)
  Dynamic_Spherical_No_Manifold  → SLERP with saved alpha_trajectory (raw vector)
  Dynamic_Linear                 → linear inject with saved alpha_trajectory
  Dynamic_Spherical_No_ThinkBrake→ SLERP with saved alpha_trajectory (purified vector)
  Dynamic_Spherical_No_EMA       → SLERP with saved alpha_trajectory; EMA beta=1.0

Usage:
  python backfill_ema.py --json_path ./results/.../experiment_results.json
"""

import json
import argparse
import math
import os
import sys
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from spherical_injector import spherical_rotate, linear_inject
from run_experiment import (
    load_control_vectors,
    _DYNAMIC_MODES,
    _HOOK_MODES,
    load_any_results,
    save_jsonl_results
)


# ---------------------------------------------------------------------------
# Helper: determine which injection flavour to use for each mode
# ---------------------------------------------------------------------------

def _get_mode_meta(mode: str) -> dict:
    """
    Returns per-mode metadata that controls how the parallel forward pass hook
    is constructed and how EMA is computed.

    Keys:
        use_hook   : bool   — whether to attach a steering hook at all
        use_linear : bool   — True → linear_inject; False → spherical_rotate
        use_raw    : bool   — True → use raw CAA vector (No_Manifold ablation)
        ema_beta   : float  — beta for EMA reconstruction (1.0 = instantaneous)
    """
    use_hook   = (mode in _HOOK_MODES)
    use_linear = (mode in ("Continuous_Linear", "Dynamic_Linear", "True_TAE"))
    use_raw    = (mode in ("Dynamic_Spherical_No_Manifold", "True_TAE"))
    ema_beta   = 1.0 if mode == "Dynamic_Spherical_No_EMA" else config.EMA_BETA

    return dict(use_hook=use_hook, use_linear=use_linear,
                use_raw=use_raw, ema_beta=ema_beta)


def _get_constant_alpha(mode: str) -> float | None:
    """Return fixed alpha for constant-injection modes, else None (use trajectory)."""
    if mode == "Continuous":
        return config.CONTINUOUS_ALPHA
    if mode == "Continuous_Linear":
        return config.CONTINUOUS_LINEAR_ALPHA
    return None


# ---------------------------------------------------------------------------
# Core: single parallel forward pass with optional hook
# ---------------------------------------------------------------------------

def calculate_entropy_parallel(
    model,
    input_ids_tensor: torch.Tensor,
    input_len: int,
    control_vector: torch.Tensor | None = None,
    alpha_trajectory: list[float] | None = None,
    use_linear: bool = False,
) -> list[float]:
    """
    Single forward pass over the full token sequence; returns per-token Shannon
    entropy for the generated part [input_len .. end).

    If control_vector and alpha_trajectory are provided, a replay hook is
    applied at config.LAYER_ID to simulate the original intervention.

    Args:
        model              : loaded AutoModelForCausalLM
        input_ids_tensor   : shape [1, full_seq_len]
        input_len          : number of prompt tokens
        control_vector     : unit-normalized [1, 1, d] tensor or None
        alpha_trajectory   : list of alpha values per generated token (may be
                             a constant list for Continuous modes)
        use_linear         : if True, use linear_inject instead of spherical_rotate
    """
    handles = []
    should_hook = (control_vector is not None
                   and alpha_trajectory is not None
                   and len(alpha_trajectory) > 0)

    if should_hook:
        seq_len = input_ids_tensor.shape[1]

        # Build alpha tensor over the full sequence (prompt part stays 0)
        alpha_full = torch.zeros(seq_len,
                                 device=input_ids_tensor.device,
                                 dtype=torch.float32)
        traj_len = min(len(alpha_trajectory), seq_len - input_len)
        if traj_len > 0:
            alpha_full[input_len:input_len + traj_len] = torch.tensor(
                alpha_trajectory[:traj_len], dtype=torch.float32,
                device=input_ids_tensor.device
            )
        # shape [1, seq_len, 1] for broadcasting over (batch, tokens, dim)
        alpha_full = alpha_full.unsqueeze(0).unsqueeze(-1)

        def parallel_hook(module, args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            if hidden.shape[1] <= input_len:
                return output  # prefill only, skip

            v = control_vector.to(device=hidden.device, dtype=hidden.dtype)
            if v.dim() == 1: v = v.unsqueeze(0).unsqueeze(0)
            elif v.dim() == 2: v = v.unsqueeze(0)

            # Mask out tokens where alpha == 0
            a = alpha_full.to(device=hidden.device, dtype=hidden.dtype)
            is_active = (a > 0).squeeze(-1)  # [1, seq_len]

            if not is_active.any():
                return output

            if use_linear:
                # Linear injection: h_new = h + α_slerp_to_linear * ||h|| * v
                # The trajectory already stores the SLERP alpha produced by PID.
                # We apply the same sin-calibration used at generation time.
                a_lin = torch.sin(a * (math.pi / 2.0))
                v_expanded = v.expand(hidden.shape[0], hidden.shape[1], -1)
                h_new = linear_inject(hidden, v_expanded, a_lin)
            else:
                v_expanded = v.expand(hidden.shape[0], hidden.shape[1], -1)
                h_new = spherical_rotate(hidden, v_expanded, a)

            hidden_new = torch.where(is_active.unsqueeze(-1), h_new, hidden)

            if isinstance(output, tuple):
                return (hidden_new,) + output[1:]
            return hidden_new

        base_model = model.model
        layer = base_model.language_model.layers[config.LAYER_ID] if hasattr(base_model, "language_model") else base_model.layers[config.LAYER_ID]
        handles.append(layer.register_forward_hook(parallel_hook))

    try:
        with torch.no_grad():
            outputs = model(input_ids_tensor)
            logits = outputs.logits   # [1, seq_len, vocab_size]
    finally:
        for h in handles:
            h.remove()

    # logits[0, i] predicts token at position i+1
    # Generated tokens start at input_len, so their prediction logits are at
    # indices [input_len-1 .. seq_len-2]
    seq_len = input_ids_tensor.shape[1]
    gen_logits = logits[0, input_len - 1: seq_len - 1, :]  # [gen_len, V]

    probs     = F.softmax(gen_logits, dim=-1)
    log_probs = F.log_softmax(gen_logits, dim=-1)
    entropy   = -torch.sum(probs * log_probs, dim=-1)  # [gen_len]
    return entropy.cpu().tolist()


# ---------------------------------------------------------------------------
# Main backfill logic
# ---------------------------------------------------------------------------

def backfill_json(json_path: str, output_path: str, limit: int | None = None):
    print(f"Loading results from {json_path}...")
    data = load_any_results(json_path)

    print(f"Loading model from {config.MODEL_PATH}...")
    model_dtype = getattr(torch, config.DEFAULT_DTYPE)
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_PATH,
        torch_dtype=model_dtype,
        device_map=config.DEVICE_MAP,
        attn_implementation=config.ATTN_IMPLEMENTATION,
    )

    print("\n🔬 Loading control vectors...")
    control_vectors = load_control_vectors(
        config.VECTOR_DIR,
        device=str(model.device),
        dtype=model_dtype,
    )
    purified_cv = control_vectors.get("purified", None)
    raw_cv      = control_vectors.get("raw", None)

    # Iterate over EVERY mode present in the JSON
    all_modes = list(data.keys())
    print(f"\n📋 Found groups: {all_modes}")

    for group_name in all_modes:
        group_data = data[group_name]
        problems   = group_data.get("per_problem", [])
        if not problems:
            print(f"\n⚠️  {group_name}: no per_problem entries, skipping.")
            continue

        meta    = _get_mode_meta(group_name)
        beta    = meta["ema_beta"]

        # Select the correct vector
        if not meta["use_hook"]:
            cv = None
        elif meta["use_raw"]:
            cv = raw_cv
            if cv is None:
                print(f"\n⚠️  {group_name}: raw vector not found "
                      f"(critic_raw.pt missing). Skipping hook.")
        else:
            cv = purified_cv

        print(f"\n{'='*60}")
        print(f"  Processing group: {group_name}  "
              f"(β={beta:.2f}, linear={meta['use_linear']}, raw={meta['use_raw']})")
        print(f"{'='*60}")

        const_alpha = _get_constant_alpha(group_name)

        count = 0
        for prob in tqdm(problems, desc=group_name):
            if limit is not None and count >= limit:
                break

            output_ids = prob.get("output_ids", [])
            input_len  = prob.get("input_len", 0)
            if not output_ids or input_len == 0:
                count += 1
                continue

            # Build alpha trajectory
            if const_alpha is not None:
                gen_len    = len(output_ids) - input_len
                alpha_traj = [const_alpha] * gen_len
            else:
                alpha_traj = prob.get("alpha_trajectory", [])

            # For Dynamic_Linear, the trajectory stores SLERP alphas —
            # the hook applies sin-calibration internally; pass them as-is.

            input_ids_tensor = torch.tensor([output_ids], device=model.device)
            try:
                entropy_traj = calculate_entropy_parallel(
                    model,
                    input_ids_tensor,
                    input_len,
                    control_vector=cv if meta["use_hook"] else None,
                    alpha_trajectory=alpha_traj if meta["use_hook"] else None,
                    use_linear=meta["use_linear"],
                )

                # Recompute EMA with the mode-specific beta
                ema_traj = []
                ema_val  = 0.0
                for step_i, h_val in enumerate(entropy_traj):
                    if step_i == 0:
                        ema_val = h_val
                    else:
                        ema_val = beta * h_val + (1.0 - beta) * ema_val
                    ema_traj.append(ema_val)

                prob["entropy_trajectory"] = entropy_traj
                prob["ema_trajectory"]     = ema_traj

            except Exception as e:
                print(f"\n  ⚠️  Error on problem {prob.get('id', count)}: {e}")

            count += 1

        # Update group-level representative trajectories
        filled = [p for p in problems if "ema_trajectory" in p]
        if filled:
            rep = filled[0]
            group_data["ema_trajectory"]   = rep["ema_trajectory"]
            group_data["alpha_trajectory"] = (
                prob.get("alpha_trajectory", [])
                if group_name != "Baseline"
                else [0.0] * len(rep["ema_trajectory"])
            )
            print(f"  ✅ {group_name}: back-filled {count} problems.")

    print(f"\n💾 Saving updated results to {output_path}...")
    if output_path.endswith(".jsonl"):
        save_jsonl_results(data, output_path)
    else:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=config.JSON_INDENT, ensure_ascii=False)
    print("Done!")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Back-fill entropy/EMA trajectories for all experiment groups"
    )
    parser.add_argument(
        "--json_path",  type=str, required=True,
        help="Path to input experiment_results.json"
    )
    parser.add_argument(
        "--output_path", type=str, default=None,
        help="Path to save updated JSON (default: <input>_fixed.json)"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit number of problems per group (for testing)"
    )
    parser.add_argument(
        "--modes", nargs="+", default=None,
        help="Only process specific modes (e.g. --modes Baseline Continuous_Linear)"
    )
    args = parser.parse_args()

    if args.output_path is None:
        base, ext = os.path.splitext(args.json_path)
        args.output_path = f"{base}_fixed{ext}"

    backfill_json(args.json_path, args.output_path, args.limit)
