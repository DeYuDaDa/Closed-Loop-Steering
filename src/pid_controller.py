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

from config import PID_KP, PID_KI, PID_KD, ALPHA_MAX, TECA_THRESHOLD


class PIDController:
    """
    Discrete PID controller for closed-loop steering, with anti-windup.

    Input:  TECA_t (current Token Entropy Cumulative Average)
    Output: α_t   (rotation angle for spherical steering)
    """

    def __init__(
        self,
        setpoint: float = TECA_THRESHOLD,
        kp: float = PID_KP,
        ki: float = PID_KI,
        kd: float = PID_KD,
        alpha_max: float = ALPHA_MAX,
    ):
        self.setpoint = setpoint
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.alpha_max = alpha_max

        # Internal state
        self.prev_error: float = 0.0
        self.integral: float = 0.0

    def step(self, teca: float) -> float:
        """
        Compute one PID step with anti-windup protection.

        Args:
            teca: Current TECA value (process variable).

        Returns:
            alpha: Rotation angle, clamped to [0, alpha_max].
        """
        # Error: positive when TECA exceeds setpoint (model is confused)
        error = teca - self.setpoint

        # Proportional term
        P = self.kp * error

        # Derivative term (based on previous error before integral update)
        D = self.kd * (error - self.prev_error)

        # Anti-Windup via Conditional Integration:
        # Check if the pre-update output is already saturated.
        # If saturated in the same direction as the current error,
        # skip the integral accumulation to prevent windup.
        raw_output = P + self.integral + D
        output_at_upper = raw_output >= self.alpha_max and error > 0
        output_at_lower = raw_output <= 0.0 and error < 0
        if not (output_at_upper or output_at_lower):
            self.integral += self.ki * error
            # Hard clamp on integral as a secondary safety net
            self.integral = max(-self.alpha_max, min(self.integral, self.alpha_max))

        # Update previous error
        self.prev_error = error

        # Final output
        alpha = P + self.integral + D

        # Clamp to [0, alpha_max]:
        #   - α = 0    means no intervention (TECA is fine)
        #   - α > 0    means rotate hidden state toward control vector
        #   - α_max    is safety ceiling to prevent gibberish
        alpha = max(0.0, min(alpha, self.alpha_max))

        return alpha

    def reset(self):
        """Reset controller state for a new generation episode."""
        self.prev_error = 0.0
        self.integral = 0.0
