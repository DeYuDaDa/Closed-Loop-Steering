"""
Module 3: PID Dynamic Controller
==================================
Discrete PID controller that maps TECA → rotation angle α.

The controller only produces positive α when TECA exceeds the setpoint,
meaning the model is "confused" and needs intervention.
When TECA is below setpoint, α = 0 (no intervention needed).

Mathematical formulation:
    e_t = TECA_t - SetPoint
    P_t = Kp * e_t
    I_t = I_{t-1} + Ki * e_t   (with anti-windup clamping)
    D_t = Kd * (e_t - e_{t-1})
    α_t = Clamp(P_t + I_t + D_t, 0, α_max)

Anti-Windup:
    Conditional integration is used to prevent the integral term from
    accumulating indefinitely when the output is already saturated.
    This avoids the runaway feedback loop where high TECA → high integral
    → α stays at max → model never converges → TECA stays high.
"""

import torch
from config import (
    PID_KP, PID_KI, PID_KD, ALPHA_MAX, ENTROPY_THRESHOLD,
    EAST_ENABLED, EAST_LAMBDA_SCALE, EAST_HIGH_ENTROPY_THETA,
    EAST_H_MIN, EAST_H_MAX,
)


class PIDController:
    """
    Discrete PID controller for closed-loop steering, with anti-windup.
    Supports batched multi-sequence processing.

    Input:  Entropy_t [batch_size]
    Output: α_t       [batch_size]
    """

    def __init__(
        self,
        batch_size: int,
        device: str = "cuda",
        setpoint: float = ENTROPY_THRESHOLD,
        kp: float = PID_KP,
        ki: float = PID_KI,
        kd: float = PID_KD,
        alpha_max: float = ALPHA_MAX,
        kp_min: float = 0.5,
        kp_max: float = 2.5,
        e_mid: float = 0.05,
        lambda_val: float = 40.0,
        use_dynamic_gain: bool = True,
        use_soft_clip: bool = True,
        # EAST parameters
        east_enabled: bool = EAST_ENABLED,
        east_lambda: float = EAST_LAMBDA_SCALE,
        east_theta: float = EAST_HIGH_ENTROPY_THETA,
        east_h_min: float = EAST_H_MIN,
        east_h_max: float = EAST_H_MAX,
    ):
        self.batch_size = batch_size
        self.device = device
        self.setpoint = setpoint
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.alpha_max = alpha_max
        
        # Adaptive PD parameters
        self.kp_min = kp_min
        self.kp_max = kp_max
        self.e_mid = e_mid
        self.lambda_val = lambda_val
        self.use_dynamic_gain = use_dynamic_gain
        self.use_soft_clip = use_soft_clip

        # EAST: Entropy-Scaled Steering (方案一, arXiv:2406.00244)
        # When EMA_t > theta_high (model confused / high-entropy), alpha is suppressed toward 0.
        # This prevents Steering from "surface hijacking" the flat probability landscape.
        self.east_enabled = east_enabled
        self.east_lambda   = east_lambda
        self.east_theta    = east_theta
        self.east_h_min    = east_h_min
        self.east_h_max    = east_h_max

        # Internal state (Batched)
        self.prev_error: torch.Tensor = torch.zeros(self.batch_size, device=self.device)
        self.integral: torch.Tensor = torch.zeros(self.batch_size, device=self.device)
        self.is_first_step: torch.Tensor = torch.ones(self.batch_size, dtype=torch.bool, device=self.device)

    def step(self, entropy: torch.Tensor, active_mask: torch.Tensor, is_converged: torch.Tensor) -> torch.Tensor:
        """
        Compute one PID step with adaptive PD and soft clipping for the batch.

        Args:
            entropy: Current EMA entropy tensor [batch_size].
            active_mask: Boolean tensor [batch_size] defining which sequences are generating.
            is_converged: Boolean tensor [batch_size] from ThinkBrake convergence latch.

        Returns:
            alpha: Rotation angle tensor [batch_size], bounded by [0, alpha_max].
        """
        # 1. Base error: e_t = max(0, EMA_t - SetPoint)
        # Clamped at 0 to avoid negative intervention when entropy is below threshold.
        error = torch.clamp(entropy - self.setpoint, min=0.0)

        # Handle first step pseudo-spike in derivative by anchoring prev_error to current error
        self.prev_error = torch.where(
            self.is_first_step & active_mask,
            error,
            self.prev_error
        )
        # Clear the first step flag for active sequences
        self.is_first_step = torch.where(
            active_mask,
            torch.zeros_like(self.is_first_step),
            self.is_first_step
        )

        # 2. Adaptive Proportional Gain (K_p)
        if self.use_dynamic_gain and self.lambda_val > 0.0:
            # K_p(e_t) = Kp_min + (Kp_max - Kp_min) / (1 + exp(-lambda * (e_t - e_mid)))
            dynamic_kp = self.kp_min + (self.kp_max - self.kp_min) / (1.0 + torch.exp(-self.lambda_val * (error - self.e_mid)))
            P = dynamic_kp * error
        else:
            P = self.kp * error

        # Derivative term
        D = self.kd * (error - self.prev_error)

        # Update previous error only for active sequences
        self.prev_error = torch.where(
            active_mask,
            error,
            self.prev_error
        )

        # Integral term (Kept for backwards compatibility, mostly 0 in PD)
        # Anti-Windup via Conditional Integration:
        raw_output = P + self.integral + D
        
        # Saturated conditions
        output_at_upper = (raw_output >= self.alpha_max) & (error > 0)
        output_at_lower = (raw_output <= 0.0) & (error < 0)
        saturated = output_at_upper | output_at_lower
        
        # Only update integral if NOT saturated AND sequence is active AND NOT converged
        update_mask = (~saturated) & active_mask & (~is_converged)
        
        new_integral_val = self.integral + self.ki * error
        # Clamp integral
        new_integral_val = torch.clamp(new_integral_val, min=-self.alpha_max, max=self.alpha_max)
        
        self.integral = torch.where(
            update_mask,
            new_integral_val,
            self.integral
        )

        # 3. Raw Output Calculation: u_t = max(0, P_t + I_t + D_t)
        u_t = torch.clamp(P + self.integral + D, min=0.0)

        # 4. Tanh Soft Clipping Protection
        if self.use_soft_clip:
            alpha = self.alpha_max * torch.tanh(u_t / self.alpha_max)
        else:
            # Fall back to hard clamping
            alpha = torch.clamp(u_t, min=0.0, max=self.alpha_max)

        # 5. EAST: Entropy-Scaled Steering (方案一)
        # Suppresses alpha when EMA entropy is in the mid-to-high range (0.35–0.65),
        # preventing the steering vector from "surface hijacking" the flat probability
        # landscape and forcing degenerate surface tokens (Let/Wait/me).
        #
        # α'_t = α_t · sigmoid_decay · (1 - H_normalized)
        #   sigmoid_decay  = σ(-λ · (EMA_t - θ_high))  → 1 when EMA_t << θ_high,  0 when EMA_t >> θ_high
        #   H_normalized   = clip((H_t - H_min)/(H_max - H_min), 0, 1)
        if self.east_enabled:
            # Component 1: sigmoid decay — hard gate around theta_high
            sigmoid_decay = torch.sigmoid(-self.east_lambda * (entropy - self.east_theta))
            # Component 2: normalized entropy weight — smooth attenuation
            h_range = max(self.east_h_max - self.east_h_min, 1e-6)
            h_norm = ((entropy - self.east_h_min) / h_range).clamp(0.0, 1.0)
            east_scale = sigmoid_decay * (1.0 - h_norm)
            alpha = alpha * east_scale

        # 6. ThinkBrake Hard Cutoff: if converged, force alpha to 0
        alpha = torch.where(
            is_converged,
            torch.zeros_like(alpha),
            alpha
        )

        # For inactive sequences, ensure alpha is exactly 0
        alpha = torch.where(active_mask, alpha, torch.zeros_like(alpha))

        return alpha

    def reset(self):
        """Reset controller state for a new generation episode."""
        self.prev_error.zero_()
        self.integral.zero_()
        self.is_first_step.fill_(True)
