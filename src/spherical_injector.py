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
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Perform SLERP (Spherical Linear Interpolation) to steer h toward v.
    Strictly aligned with the official Spherical Steering paper logic.

    Args:
        h: Hidden state tensor, shape [batch, 1, d].
        v: Normalized target control vector, shape broadcastable to h.
        alpha: Steering strength percentage [0.0, 1.0]. 
               0.0 = no steering, 1.0 = strictly aligned with v.
        eps: Small constant for numerical stability.

    Returns:
        h_new: Rotated hidden state with ||h_new|| == ||h||.
    """
    orig_dtype = h.dtype
    h = h.float()
    v = v.float()

    # 1. Compute and save the original norm
    h_norm = torch.norm(h, dim=-1, keepdim=True).clamp_min(eps)
    h_hat = h / h_norm

    # 2. Compute current angle (theta) between h_hat and v
    cos_theta = torch.sum(h_hat * v, dim=-1, keepdim=True).clamp(-1.0 + eps, 1.0 - eps)
    theta = torch.acos(cos_theta)

    # 3. Handle alpha broadcasting and clamping [0, 1]
    if isinstance(alpha, torch.Tensor):
        alpha_b = alpha.float()
        while alpha_b.dim() < h.dim():
            alpha_b = alpha_b.unsqueeze(-1)
    else:
        alpha_b = torch.tensor(alpha, dtype=torch.float32, device=h.device)
    
    # 物理意义修正：alpha 现在是 0~1 的插值比例
    t = torch.clamp(alpha_b, 0.0, 1.0)

    # 4. Compute new angle (shrink the gap by proportion t)
    # If t=0.3, theta_new is 70% of original theta (moved 30% closer)
    theta_new = (1.0 - t) * theta

    # 5. Spherical interpolation (SLERP orthogonal basis)
    sin_theta = torch.sin(theta).clamp_min(eps)
    u = (h_hat - cos_theta * v) / sin_theta  # Orthogonal component

    # 6. Construct new vector using target v and orthogonal u
    h_hat_rotated = torch.cos(theta_new) * v + torch.sin(theta_new) * u

    # 7. Restore original norm
    h_new = h_norm * h_hat_rotated

    # 8. Handle edge cases (if h and v are already perfectly aligned)
    is_aligned = (theta < eps)
    if is_aligned.any():
        mask = is_aligned.expand_as(h)
        h_new = torch.where(mask, h, h_new)

    return h_new.to(orig_dtype)

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
