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
from config import PID_KP, PID_KI, PID_KD, ALPHA_MAX, TECA_THRESHOLD


class PIDController:
    """
    Discrete PID controller for closed-loop steering, with anti-windup.
    Supports batched multi-sequence processing.

    Input:  TECA_t [batch_size]
    Output: α_t    [batch_size]
    """

    def __init__(
        self,
        batch_size: int,
        device: str = "cuda",
        setpoint: float = TECA_THRESHOLD,
        kp: float = PID_KP,
        ki: float = PID_KI,
        kd: float = PID_KD,
        alpha_max: float = ALPHA_MAX,
    ):
        self.batch_size = batch_size
        self.device = device
        self.setpoint = setpoint
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.alpha_max = alpha_max

        # Internal state (Batched)
        self.prev_error: torch.Tensor = torch.zeros(self.batch_size, device=self.device)
        self.integral: torch.Tensor = torch.zeros(self.batch_size, device=self.device)

    def step(self, teca: torch.Tensor, active_mask: torch.Tensor) -> torch.Tensor:
        """
        Compute one PID step with anti-windup protection for the whole batch.

        Args:
            teca: Current TECA tensor [batch_size].
            active_mask: Boolean tensor [batch_size] defining which sequences are generating.

        Returns:
            alpha: Rotation angle tensor [batch_size], clamped to [0, alpha_max].
        """
        # Error: positive when TECA exceeds setpoint (model is confused)
        error = teca - self.setpoint

        # Proportional term
        P = self.kp * error

        # Derivative term (based on previous error before integral update)
        D = self.kd * (error - self.prev_error)

        # Anti-Windup via Conditional Integration:
        raw_output = P + self.integral + D
        
        # Saturated conditions
        output_at_upper = (raw_output >= self.alpha_max) & (error > 0)
        output_at_lower = (raw_output <= 0.0) & (error < 0)
        saturated = output_at_upper | output_at_lower
        
        # Only update integral if NOT saturated AND sequence is active
        update_mask = (~saturated) & active_mask
        
        new_integral_val = self.integral + self.ki * error
        # Clamp integral
        new_integral_val = torch.clamp(new_integral_val, min=-self.alpha_max, max=self.alpha_max)
        
        self.integral = torch.where(
            update_mask,
            new_integral_val,
            self.integral
        )

        # Update previous error only for active sequences
        self.prev_error = torch.where(
            active_mask,
            error,
            self.prev_error
        )

        # Final output
        alpha = P + self.integral + D

        # Clamp to [0, alpha_max]
        alpha = torch.clamp(alpha, min=0.0, max=self.alpha_max)

        # For inactive sequences, ensure alpha is exactly 0
        alpha = torch.where(active_mask, alpha, torch.zeros_like(alpha))

        return alpha

    def reset(self):
        """Reset controller state for a new generation episode."""
        self.prev_error.zero_()
        self.integral.zero_()
