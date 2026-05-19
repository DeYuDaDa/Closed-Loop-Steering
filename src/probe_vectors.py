"""
Vector Probe — What do critic_raw.pt and critic.pt point to?
=============================================================
Two experiments in one script:

Experiment A — Direct Hidden-State Decode
    Load the raw and purified critic vectors, scale them to a realistic
    hidden-state norm (~80-120 for layer-24 of Qwen3-8B), then feed them
    directly through the LM Head (+ optional RMS Norm) to get a token
    distribution.  Report top-20 tokens + probabilities.

Experiment B — Spherical Rotation Sweep
    Start from the same scaled vector (as the "current hidden state"),
    apply spherical_rotate with alpha ∈ [-0.6, -0.45, -0.3, -0.1, 0,
    0.1, 0.3, 0.45, 0.6] toward the *purified* unit critic vector,
    then decode each rotated state.  Report top-20 tokens + probabilities.

Usage:
    cd /path/to/src
    python probe_vectors.py

Outputs:
    ./probe_results/
        A_raw_vector_decode.txt
        A_purified_vector_decode.txt
        B_rotation_sweep.txt
"""

import os
import sys
import math
import torch
import json
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

# ── local imports (no modification of existing modules) ────────────────────────
from config import MODEL_PATH, LAYER_ID, VECTOR_DIR
LAYER_ID = 16
from spherical_injector import spherical_rotate
# ── constants ──────────────────────────────────────────────────────────────────
TARGET_NORM  = 100           # Realistic hidden-state norm at layer 24
TOP_K        = 20              # Number of tokens to report
OUT_DIR      = "./probe_results"

ALPHA_SWEEP  = [-0.6, -0.45, -0.3, -0.1, 0.0, 0.1, 0.3, 0.45, 0.6]

# ── helpers ────────────────────────────────────────────────────────────────────

def load_vector(path: str, device: str = "cpu") -> torch.Tensor:
    """Load a saved .pt vector and return as float32 CPU tensor."""
    v = torch.load(path, map_location="cpu").float()
    print(f"  Loaded  {Path(path).name}  shape={list(v.shape)}  norm={v.norm().item():.4f}")
    return v


def scale_to_norm(v: torch.Tensor, target_norm: float) -> torch.Tensor:
    """L2-normalize then scale to target_norm."""
    unit = v / v.norm().clamp_min(1e-8)
    return unit * target_norm


def decode_hidden_state(
    h: torch.Tensor,          # shape [hidden_dim]  (float32 CPU)
    model,
    tokenizer,
    top_k: int = TOP_K,
    label: str = "",
) -> list[tuple[str, float]]:
    """
    Push h through the model's LM head (with RMSNorm if present) and return
    top-k (token_str, probability) pairs.

    Qwen3 architecture:
        hidden → model.norm (RMSNorm) → lm_head (Linear, no bias)
    """
    device = next(model.parameters()).device
    dtype  = next(model.parameters()).dtype

    # Shape: [1, 1, hidden_dim]  (batch=1, seq=1)
    h_3d = h.to(device=device, dtype=dtype).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        # Apply the final layer norm that sits between transformer layers and lm_head
        norm_out = model.model.norm(h_3d)          # [1, 1, hidden_dim]
        logits   = model.lm_head(norm_out)         # [1, 1, vocab_size]
        logits   = logits[:, 0, :]                 # [1, vocab_size]

        probs    = torch.softmax(logits.float(), dim=-1)   # stable softmax

    topk_probs, topk_ids = probs[0].topk(top_k)
    results = []
    for prob, tok_id in zip(topk_probs.tolist(), topk_ids.tolist()):
        tok_str = tokenizer.decode([tok_id], skip_special_tokens=False)
        results.append((tok_str, prob, tok_id))

    return results


def fmt_token_table(results: list[tuple[str, float, int]], header: str) -> str:
    """Format top-k results into a pretty aligned table."""
    lines = []
    lines.append(header)
    lines.append("=" * len(header))
    lines.append(f"  {'Rank':<5} {'Prob':>8}  {'Token'}")
    lines.append("  " + "-" * 50)
    for rank, (tok, prob, tok_id) in enumerate(results, 1):
        # escape control characters for display
        tok_display = repr(tok).strip("'")
        lines.append(f"  {rank:<5} {prob:>8.4%}  {tok_display}  (id={tok_id})")
    lines.append("")
    return "\n".join(lines)


def save_text(path: str, content: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  ✅  Saved → {path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 70)
    print("  Vector Probe — Experiment A (Direct Decode) + B (Rotation Sweep)")
    print("=" * 70)

    # ── 1. Load vectors ────────────────────────────────────────────────────────
    raw_path      = os.path.join(VECTOR_DIR, "math_grounding_raw.pt")
    purified_path = os.path.join(VECTOR_DIR, "math_grounding.pt")

    print(f"\n[Step 1] Loading vectors from {VECTOR_DIR}")
    v_raw      = load_vector(raw_path)       # shape [d], arbitrary norm
    v_purified = load_vector(purified_path)  # shape [d], unit norm after extract_critic_vector.py

    hidden_dim = v_raw.shape[0]
    print(f"  Hidden dim: {hidden_dim}")

    # Scale both to a realistic hidden-state norm for layer 24
    h_raw      = scale_to_norm(v_raw,      TARGET_NORM)
    h_purified = scale_to_norm(v_purified, TARGET_NORM)
    print(f"\n  h_raw      norm after scaling: {h_raw.norm().item():.2f}")
    print(f"  h_purified norm after scaling: {h_purified.norm().item():.2f}")

    # ── 2. Load model ──────────────────────────────────────────────────────────
    print(f"\n[Step 2] Loading model: {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    print(f"  Model loaded on device: {next(model.parameters()).device}")

    # ══════════════════════════════════════════════════════════════════════════
    #  EXPERIMENT A — Direct Hidden-State Decode
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 70)
    print("  EXPERIMENT A — Direct Hidden-State Decode")
    print("─" * 70)

    a_output_lines = []
    a_output_lines.append("EXPERIMENT A — Direct Hidden-State Decode")
    a_output_lines.append(f"Layer: {LAYER_ID}  |  Hidden dim: {hidden_dim}  |  Target norm: {TARGET_NORM}")
    a_output_lines.append(f"Model: {MODEL_PATH}\n")

    for vec_name, h_vec, v_ori_norm in [
        ("critic_raw.pt  (raw CAA vector)",      h_raw,      v_raw.norm().item()),
        ("critic.pt      (PCA-purified, unit)",  h_purified, v_purified.norm().item()),
    ]:
        print(f"\n  Decoding: {vec_name}  (original norm={v_ori_norm:.4f})")
        header = f"▶ {vec_name}"
        header += f"\n  original norm = {v_ori_norm:.4f}  →  scaled to {TARGET_NORM}"
        results = decode_hidden_state(h_vec, model, tokenizer, top_k=TOP_K, label=vec_name)
        table = fmt_token_table(results, header)
        a_output_lines.append(table)

        # Also print to console
        for rank, (tok, prob, tok_id) in enumerate(results, 1):
            print(f"    {rank:>2}. {prob:.4%}  {repr(tok):<25} (id={tok_id})")

    a_text = "\n".join(a_output_lines)
    save_text(os.path.join(OUT_DIR, "A_vector_decode.txt"), a_text)

    # ══════════════════════════════════════════════════════════════════════════
    #  EXPERIMENT B — Spherical Rotation Sweep
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 70)
    print("  EXPERIMENT B — Spherical Rotation Sweep")
    print("─" * 70)
    print(f"  Control vector: critic.pt (unit norm)")
    print(f"  Base hidden state: h_raw (critic_raw scaled to norm={TARGET_NORM})")
    print(f"  Alpha sweep: {ALPHA_SWEEP}")

    b_output_lines = []
    b_output_lines.append("EXPERIMENT B — Spherical Rotation Sweep")
    b_output_lines.append(f"Base hidden state : critic_raw scaled to norm={TARGET_NORM}")
    b_output_lines.append(f"Control vector    : critic.pt (unit norm, PCA-purified)")
    b_output_lines.append(f"Alpha sweep       : {ALPHA_SWEEP}")
    b_output_lines.append(f"Layer: {LAYER_ID}  |  Model: {MODEL_PATH}\n")

    device = next(model.parameters()).device
    dtype  = next(model.parameters()).dtype

    # Control vector in [1, 1, d] on model device — unit norm
    v_ctrl = v_purified.to(device=device, dtype=torch.float32)
    v_ctrl = v_ctrl / v_ctrl.norm().clamp_min(1e-8)
    v_ctrl_3d = v_ctrl.unsqueeze(0).unsqueeze(0)  # [1, 1, d]

    # We sweep from TWO base states for completeness:
    #   h_raw_base      (the raw CAA vector scaled to norm 100)
    #   h_purified_base (the purified unit vector scaled to norm 100)
    base_states = [
        ("h_raw      (critic_raw, scaled)",      h_raw),
        ("h_purified (critic.pt,  scaled)",      h_purified),
    ]

    for base_label, h_base in base_states:
        b_output_lines.append(f"\n{'═' * 60}")
        b_output_lines.append(f"  Base state: {base_label}")
        b_output_lines.append(f"{'═' * 60}\n")
        print(f"\n  ── Base: {base_label}")

        for alpha in ALPHA_SWEEP:
            # h_3d: [1, 1, d] float32
            h_3d = h_base.to(device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

            # Apply spherical rotation
            h_rot = spherical_rotate(h_3d, v_ctrl_3d, alpha=alpha)  # [1, 1, d]

            # Norm check
            orig_norm  = h_3d.norm().item()
            rot_norm   = h_rot.norm().item()

            # Decode
            h_rot_flat = h_rot[0, 0, :].detach().cpu().float()
            results    = decode_hidden_state(h_rot_flat, model, tokenizer, top_k=TOP_K)

            # Angle between original and rotated (sanity check)
            cos_sim = torch.nn.functional.cosine_similarity(
                h_3d.view(1, -1).float(),
                h_rot.view(1, -1).float(),
                dim=-1
            ).item()
            angle_deg = math.degrees(math.acos(max(-1.0, min(1.0, cos_sim))))

            header = (
                f"  α = {alpha:+.2f}  |  norm: {orig_norm:.2f} → {rot_norm:.2f}  "
                f"|  angle shift: {angle_deg:.2f}°"
            )
            b_output_lines.append(header)
            b_output_lines.append("  " + "-" * 58)

            print(f"\n    α={alpha:+.2f}  norm {orig_norm:.1f}→{rot_norm:.1f}  Δangle={angle_deg:.1f}°  top-5:")
            for rank, (tok, prob, tok_id) in enumerate(results, 1):
                tok_display = repr(tok)
                line = f"    {rank:>2}. {prob:.4%}  {tok_display:<30} (id={tok_id})"
                b_output_lines.append(line)
                if rank <= 5:
                    print(f"      {rank}. {prob:.4%}  {tok_display}")

            b_output_lines.append("")

    b_text = "\n".join(b_output_lines)
    save_text(os.path.join(OUT_DIR, "B_rotation_sweep.txt"), b_text)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  PROBE COMPLETE")
    print("=" * 70)
    print(f"  Results saved to: {os.path.abspath(OUT_DIR)}/")
    print(f"    A_vector_decode.txt   — direct hidden-state decode")
    print(f"    B_rotation_sweep.txt  — spherical rotation sweep")


if __name__ == "__main__":
    main()
