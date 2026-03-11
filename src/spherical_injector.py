"""
Module 4: Spherical Steering Engine — Norm-Preserving Rotation
================================================================
The mathematical core of the closed-loop system.

Given hidden state h and control vector v, rotate h toward v by angle α
in the h-v hyperplane while strictly preserving ||h||.

Reference:
  - Spherical Steering: Geometry-Aware Activation Rotation for Language Models

Mathematical formulation (Gram-Schmidt + rotation):
    1. ĥ = h / ||h||₂
    2. u = v - (v · ĥ)ĥ          (remove component parallel to h)
    3. û = u / ||u||₂             (normalize to get orthonormal basis)
    4. ĥ_rotated = cos(α)·ĥ + sin(α)·û
    5. h_new = ||h||₂ · ĥ_rotated  (restore original norm)
"""

import torch
import torch
import math


def spherical_rotate(
    h: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor | float,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Perform norm-preserving spherical rotation of h toward v by angle α.

    This is a pure function with no side effects — suitable for use
    inside a forward hook.
    Supports batched processing natively.

    Args:
        h: Hidden state tensor, shape [batch, 1, d].
        v: Normalized control vector, shape broadcastable to h.
        alpha: Rotation angle in radians, shape [batch] or float.
        eps: Small constant for numerical stability.

    Returns:
        h_new: Rotated hidden state with ||h_new|| == ||h||.
    """
    # 1. Compute and save the original norm
    h_norm = torch.norm(h, dim=-1, keepdim=True)  # [batch, 1, 1]

    # 2. Get unit direction of h
    h_hat = h / (h_norm + eps)  # [batch, 1, d]

    # 3. Gram-Schmidt: find component of v orthogonal to h_hat
    v_dot_h = torch.sum(v * h_hat, dim=-1, keepdim=True)  # [batch, 1, 1]
    u = v - v_dot_h * h_hat  # [batch, 1, d]

    # 4. Normalize u to get orthonormal basis vector û
    u_norm = torch.norm(u, dim=-1, keepdim=True)  # [batch, 1, 1]
    u_hat = u / (u_norm + eps)  # [batch, 1, d]

    # 5. Handle degenerate case
    is_degenerate = (u_norm.squeeze(-1) < eps)  # [batch, 1]

    # 6. Rotate in the h-v plane
    if isinstance(alpha, torch.Tensor):
        # Dynamically append dimensions so it broadcasts with h
        alpha_b = alpha
        while alpha_b.dim() < h.dim():
            alpha_b = alpha_b.unsqueeze(-1)
        cos_a = torch.cos(alpha_b)
        sin_a = torch.sin(alpha_b)
    else:
        # Scalar handling (e.g. Continuous mode fixed alpha)
        cos_a = math.cos(alpha)
        sin_a = math.sin(alpha)

    h_hat_rotated = cos_a * h_hat + sin_a * u_hat  # [batch, 1, d]

    # 7. Restore original norm
    h_new = h_norm * h_hat_rotated  # [batch, 1, d]

    # 8. Handle degenerate cases by reverting to original h
    if is_degenerate.any():
        mask = is_degenerate.unsqueeze(-1).expand_as(h)
        h_new = torch.where(mask, h, h_new)

    # Note: If alpha == 0, the equation cleanly yields:
    # cos(0)*h_hat + sin(0)*u_hat = h_hat.
    # h_norm * h_hat = h.
    # So we don't strictly need a special mask for alpha == 0,
    # though we could add one for micro-optimization of exact precision match.

    return h_new


def create_steering_hook(
    state,
    control_vector: torch.Tensor,
    mode: str = "Dynamic_Spherical",
    continuous_alpha: float = 0.15,
    capture_hidden_states: bool = False,
):
    """
    Factory function that creates a forward hook for spherical steering.

    The hook reads α from the shared InjectionState and applies
    spherical rotation to the hidden state at the target layer.

    Args:
        state: InjectionState object (shared with StateMonitor).
        control_vector: Normalized control vector v, shape [1, 1, d].
        mode: Experiment mode — "Baseline", "Continuous", or "Dynamic_Spherical".
        continuous_alpha: Fixed rotation angle for Continuous mode.

    Returns:
        hook: A callable compatible with register_forward_hook().
        history: List that accumulates hidden states for post-hoc analysis.
    """
    history_hidden = []

    def steering_hook(module, args, output):
        hidden = output[0] if isinstance(output, tuple) else output
        seq_len = hidden.shape[1]

        # Record the last token's hidden state for analysis
        if capture_hidden_states:
            history_hidden.append(hidden[:, -1, :].detach().cpu())

        # Only intervene during autoregressive decoding (seq_len == 1),
        # never during the prefill phase (seq_len > 1)
        if seq_len != 1:
            return output

        if mode == "Baseline":
            # No intervention
            return output

        elif mode == "Continuous":
            # Fixed-strength spherical rotation at every step
            alpha = continuous_alpha

        elif mode == "Dynamic_Spherical":
            # Read α from PID controller via shared state
            alpha = state.alpha
        else:
            return output

        # For scalar alpha (Continuous mode)
        if isinstance(alpha, (float, int)) and alpha <= 0:
            return output
        # For tensor alpha (Dynamic_Spherical mode)
        elif isinstance(alpha, torch.Tensor) and (alpha <= 0).all():
            return output

        # Align control vector to device/dtype of hidden state
        v = control_vector.to(device=hidden.device, dtype=hidden.dtype)

        # Ensure shapes: hidden is [batch, 1, dim], v should be [1, 1, dim]
        if v.dim() == 1:
            v = v.unsqueeze(0).unsqueeze(0)
        elif v.dim() == 2:
            v = v.unsqueeze(0)

        # Extract the last token's hidden state
        h = hidden[:, -1:, :]  # [batch, 1, dim]

        # Perform spherical rotation
        h_new = spherical_rotate(h, v, alpha)

        # Write back
        hidden = hidden.clone()
        hidden[:, -1:, :] = h_new

        if isinstance(output, tuple):
            return (hidden,) + output[1:]
        return hidden

    return steering_hook, history_hidden
