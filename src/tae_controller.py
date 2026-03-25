"""
Module: TAE Controller — Token-Entropy-based Adaptive (Open-loop) Controller
============================================================================
Implements the TAE baseline from the EMNLP 2025 paper.

TAE (Token-level Adaptive Entropy) is an open-loop controller that maps the
*instantaneous* raw entropy H_t of the current token directly to an intervention
strength alpha via a linear scaling:

    alpha_t = Clamp(k * H_t, 0, alpha_max)

where k = alpha_max / H_ref is a gain constant calibrated so that a token
with entropy H_ref saturates at alpha_max.

This is fundamentally different from the closed-loop PD controller:
- No EMA smoothing   → alpha jitters at token-level frequency
- No ThinkBrake      → alpha never hard-cuts off at convergence
- No error feedback  → no set-point tracking; pure open-loop

The class shares the same .step() API as PIDController for drop-in swapping.
"""

import torch
from config import ALPHA_MAX


# Reference entropy used for gain calibration.
# Value 3.0 nats corresponds to roughly uniform distribution over ~20 candidates,
# a typical high-confusion token in chain-of-thought reasoning.
TAE_ENTROPY_REFERENCE = 3.0


class TAEController:
    """
    Open-loop TAE (Token-level Adaptive Entropy) controller.

    At each decoding step, maps raw (instantaneous) token entropy H_t
    linearly to an intervention strength alpha_t.

    Supports batched multi-sequence processing; API is compatible with
    PIDController so run_experiment.py can swap them without restructuring.

    Input:  raw_entropy [batch_size]   (instantaneous Shannon entropy per token)
    Output: alpha       [batch_size]   (intervention strength, clamped to [0, alpha_max])
    """

    def __init__(
        self,
        batch_size: int,
        device: str = "cuda",
        alpha_max: float = ALPHA_MAX,
        entropy_reference: float = TAE_ENTROPY_REFERENCE,
    ):
        """
        Args:
            batch_size:        Number of parallel sequences.
            device:            Torch device string.
            alpha_max:         Upper clamp for alpha output.
            entropy_reference: Entropy value that maps to alpha_max (gain calibration).
        """
        self.batch_size = batch_size
        self.device = device
        self.alpha_max = alpha_max
        # k = alpha_max / H_ref
        self.k_weight = alpha_max / max(entropy_reference, 1e-6)

    def step(
        self,
        entropy: torch.Tensor,
        active_mask: torch.Tensor,
        is_converged: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute open-loop TAE alpha for the current decoding step.

        Args:
            entropy:      *Instantaneous* raw Shannon entropy, shape [batch_size].
                          (Note: for drop-in compatibility, we accept whatever tensor
                           is passed — for TAE it should be H_t, NOT the EMA.)
            active_mask:  Boolean [batch_size]: True if sequence is still generating.
            is_converged: Ignored by TAE (no ThinkBrake). Kept for API compatibility.

        Returns:
            alpha: Intervention strength [batch_size], clamped to [0, alpha_max].
        """
        # Open-loop linear mapping: alpha = k * H_t
        alpha = self.k_weight * entropy

        # Clamp to valid range
        alpha = torch.clamp(alpha, min=0.0, max=self.alpha_max)

        # Zero out finished sequences
        alpha = torch.where(active_mask, alpha, torch.zeros_like(alpha))

        return alpha

    def reset(self):
        """No internal state to reset (open-loop controller)."""
        pass
