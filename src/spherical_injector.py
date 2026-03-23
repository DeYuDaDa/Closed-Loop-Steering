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
import math
from config import MATH_EPSILON, CONTINUOUS_ALPHA, CONTINUOUS_LINEAR_ALPHA, CAPTURE_HIDDEN_STATES


def spherical_rotate(
    h: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor | float,
    eps: float = MATH_EPSILON,
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
    
    # Prevent explosion if highly collinear
    is_collinear = sin_theta < 1e-5
    u = torch.where(
        is_collinear.expand_as(h_hat), 
        torch.zeros_like(h_hat), 
        (h_hat - cos_theta * v) / sin_theta
    )

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


def linear_inject(
    h: torch.Tensor,
    v: torch.Tensor,
    alpha_linear: float | torch.Tensor,
    eps: float = MATH_EPSILON,
) -> torch.Tensor:
    """
    Norm-scaled linear addition: h_new = h + α_linear * ||h|| * v_unit.

    This is the "Continuous_Linear" control group (w/o SLERP).
    The coefficient α_linear is pre-calibrated via Equal Orthogonal Projection
    so that the projection onto v matches SLERP exactly, making the comparison fair.
    Crucially, linear addition CANNOT preserve ||h||, increasing the norm by
    approximately sqrt(1 + α_linear^2) which causes the expected state shock.

    Args:
        h: Hidden state tensor [batch, 1, d].
        v: Unit-normalized control vector, shape broadcastable to h.
        alpha_linear: Pre-calibrated linear coefficient (scalar or [batch] tensor).
        eps: Numerical stability epsilon.

    Returns:
        h_new: Norm-perturbed hidden state.
    """
    orig_dtype = h.dtype
    h_f = h.float()
    v_f = v.float()

    # Ensure v is a unit vector (defensive re-normalization)
    v_norm = torch.norm(v_f, dim=-1, keepdim=True).clamp_min(eps)
    v_unit = v_f / v_norm

    # Compute ||h|| per token position: shape [batch, 1, 1]
    h_norm = torch.norm(h_f, dim=-1, keepdim=True).clamp_min(eps)  # [batch, 1, 1]

    # Broadcast alpha_linear to [batch, 1, 1] if it's a tensor
    if isinstance(alpha_linear, torch.Tensor):
        a = alpha_linear.float()
        while a.dim() < h_f.dim():
            a = a.unsqueeze(-1)
    else:
        a = torch.tensor(alpha_linear, dtype=torch.float32, device=h_f.device)

    # h_new = h + (α_linear * ||h||) * v_unit
    h_new = h_f + a * h_norm * v_unit

    return h_new.to(orig_dtype)

def create_steering_hook(
    state,
    control_vector: torch.Tensor,
    mode: str = "Dynamic_Spherical",
    continuous_alpha: float = CONTINUOUS_ALPHA,
    continuous_linear_alpha: float = CONTINUOUS_LINEAR_ALPHA,
    capture_hidden_states: bool = CAPTURE_HIDDEN_STATES,
):
    """
    Factory function that creates a forward hook for spherical or linear steering.

    The hook reads α from the shared InjectionState and applies
    spherical rotation (Continuous / Dynamic_Spherical) or norm-scaled
    linear addition (Continuous_Linear) to the hidden state at the target layer.

    Args:
        state: InjectionState object (shared with StateMonitor).
        control_vector: Normalized control vector v, shape [1, 1, d].
        mode: Experiment mode — "Baseline", "Continuous", "Continuous_Linear",
              or "Dynamic_Spherical".
        continuous_alpha: Fixed SLERP rotation angle for Continuous mode.
        continuous_linear_alpha: Fixed linear coefficient for Continuous_Linear mode.

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

        elif mode == "Continuous_Linear":
            # Fixed-strength norm-scaled linear addition at every step (no SLERP)
            alpha = continuous_linear_alpha

        elif mode == "Dynamic_Spherical":
            # Read α from PID controller via shared state
            alpha = state.alpha
        else:
            return output

        # For scalar alpha (Continuous / Continuous_Linear mode)
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

        # Perform injection (linear for Continuous_Linear, spherical for others)
        if mode == "Continuous_Linear":
            h_new = linear_inject(h, v, alpha)
        else:
            h_new = spherical_rotate(h, v, alpha)

        # Write back
        hidden = hidden.clone()
        hidden[:, -1:, :] = h_new

        if isinstance(output, tuple):
            return (hidden,) + output[1:]
        return hidden

    return steering_hook, history_hidden
