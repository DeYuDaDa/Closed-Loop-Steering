"""
Module 1: State Monitor — TECA & ThinkBrake
=============================================
Real-time LogitsProcessor that computes:
  - TECA (Token Entropy Cumulative Average): detects model confusion/exploration
  - ThinkBrake (Logit Margin): detects logical convergence

References:
  - CER: Cumulative Entropy Regulation (TECA)
  - ThinkBrake: Mitigating Overthinking in Tool Reasoning (Margin)
"""

import torch
import torch.nn.functional as F
from transformers import LogitsProcessor

from config import (
    TECA_TEMPERATURE,
    TECA_EPSILON,
    TECA_THRESHOLD,
    CONVERGENCE_MARGIN_TAU,
)


class InjectionState:
    """
    Shared mutable state object passed between LogitsProcessor and Forward Hook.
    Acts as the communication bus for the closed-loop system.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        # --- TECA state ---
        self.entropy_sum: float = 0.0
        self.step_count: int = 0
        self.teca: float = 0.0

        # --- ThinkBrake state ---
        self.margin: float = float("inf")

        # --- PID output (written by PIDController, read by Hook) ---
        self.alpha: float = 0.0

        # --- Trajectory logging (for visualization) ---
        self.teca_trajectory: list[float] = []
        self.alpha_trajectory: list[float] = []
        self.entropy_trajectory: list[float] = []

        # --- Flags ---
        self.intervention_active: bool = False
        self.converged: bool = False

        # --- Intervention window tracking ---
        self.intervention_start_step: int | None = None
        self.intervention_end_step: int | None = None


class StateMonitor(LogitsProcessor):
    """
    LogitsProcessor that computes TECA and ThinkBrake Margin at every
    generation step, updating the shared InjectionState object.
    """

    def __init__(
        self,
        state: InjectionState,
        pid_controller=None,
        term_token_id: int | None = None,
        temperature: float = TECA_TEMPERATURE,
        epsilon: float = TECA_EPSILON,
        teca_threshold: float = TECA_THRESHOLD,
        margin_tau: float = CONVERGENCE_MARGIN_TAU,
    ):
        """
        Args:
            state: Shared InjectionState object.
            pid_controller: Optional PIDController instance. If provided,
                            PID.step(teca) is called automatically.
            term_token_id: Token ID for </think> (for ThinkBrake margin).
            temperature: Softmax temperature for entropy calculation.
            epsilon: Small constant for numerical stability.
            teca_threshold: TECA threshold that triggers intervention.
            margin_tau: Margin threshold for convergence detection.
        """
        self.state = state
        self.pid = pid_controller
        self.term_token_id = term_token_id
        self.temperature = temperature
        self.epsilon = epsilon
        self.teca_threshold = teca_threshold
        self.margin_tau = margin_tau

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        Called at every generation step by HuggingFace generate().
        `scores` is the raw logits tensor of shape [batch, vocab_size].
        """
        # --- 1. Compute current token entropy H_t ---
        # Apply temperature scaling and softmax
        probs = F.softmax(scores / self.temperature, dim=-1)  # [batch, V]
        # Shannon entropy: H = -sum(p * log(p))
        log_probs = torch.log(probs + self.epsilon)
        H_t = -torch.sum(probs * log_probs, dim=-1)  # [batch]
        H_t_val = H_t[0].item()  # Take first batch element

        # --- 2. Update TECA (cumulative average entropy) ---
        self.state.step_count += 1
        self.state.entropy_sum += H_t_val
        self.state.teca = self.state.entropy_sum / self.state.step_count

        # Log trajectories
        self.state.entropy_trajectory.append(H_t_val)
        self.state.teca_trajectory.append(self.state.teca)

        # --- 3. Compute ThinkBrake Margin M_t (if term token is set) ---
        if self.term_token_id is not None:
            log_probs_full = F.log_softmax(scores, dim=-1)  # [batch, V]
            max_log_prob = log_probs_full[0].max().item()
            term_log_prob = log_probs_full[0, self.term_token_id].item()
            self.state.margin = max_log_prob - term_log_prob

            # Check convergence
            if self.state.margin <= self.margin_tau:
                self.state.converged = True

        # --- 4. Drive PID controller if TECA breaches threshold ---
        if self.pid is not None:
            alpha = self.pid.step(self.state.teca)
            self.state.alpha = alpha

            # Track intervention window:
            # - Record start only on the FIRST time alpha > 0
            # - Record/update end whenever alpha = 0 (or at the end of generation)
            if alpha > 0:
                if not self.state.intervention_active:
                    self.state.intervention_active = True
                    self.state.intervention_start_step = self.state.step_count
                # If active, keep updating the "latest" known end step to the current step
                # so that if generation stops while intervening, we have a valid end.
                self.state.intervention_end_step = self.state.step_count
            else:
                if self.state.intervention_active:
                    self.state.intervention_active = False
                    self.state.intervention_end_step = self.state.step_count

        # Log alpha trajectory
        self.state.alpha_trajectory.append(self.state.alpha)

        # Scores pass through unmodified — we only observe, never mutate logits
        return scores
