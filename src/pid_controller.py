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
    I_t = I_{t-1} + Ki * e_t
    D_t = Kd * (e_t - e_{t-1})
    α_t = Clamp(P_t + I_t + D_t, 0, α_max)
"""

from config import PID_KP, PID_KI, PID_KD, ALPHA_MAX, TECA_THRESHOLD


class PIDController:
    """
    Discrete PID controller for closed-loop steering.

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
        Compute one PID step.

        Args:
            teca: Current TECA value (process variable).

        Returns:
            alpha: Rotation angle, clamped to [0, alpha_max].
        """
        # Error: positive when TECA exceeds setpoint (model is confused)
        error = teca - self.setpoint

        # Proportional term
        P = self.kp * error

        # Integral term (accumulate)
        self.integral += self.ki * error

        # Derivative term
        D = self.kd * (error - self.prev_error)

        # Update previous error
        self.prev_error = error

        # Raw output
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
