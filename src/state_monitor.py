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
    Handles batched computation.
    """

    def __init__(self, batch_size: int, device: str = "cuda"):
        self.batch_size = batch_size
        self.device = device
        self.reset()

    def reset(self):
        # --- TECA state (Batched) ---
        self.entropy_sum: torch.Tensor = torch.zeros(self.batch_size, device=self.device)
        self.step_count: torch.Tensor = torch.zeros(self.batch_size, device=self.device)
        self.teca: torch.Tensor = torch.zeros(self.batch_size, device=self.device)

        # --- ThinkBrake state (Batched) ---
        self.margin: torch.Tensor = torch.full((self.batch_size,), float("inf"), device=self.device)

        # --- PID output (written by PIDController, read by Hook) ---
        self.alpha: torch.Tensor = torch.zeros(self.batch_size, device=self.device)

        # --- Trajectory logging (Batched) ---
        # List of lists, outer list is per-sequence in the batch
        self.teca_trajectory: list[list[float]] = [[] for _ in range(self.batch_size)]
        self.alpha_trajectory: list[list[float]] = [[] for _ in range(self.batch_size)]
        self.entropy_trajectory: list[list[float]] = [[] for _ in range(self.batch_size)]

        # --- Flags (Batched) ---
        self.intervention_active: torch.Tensor = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)
        self.converged: torch.Tensor = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)
        
        # Track whether a sequence is still generating (True) or has hit EOS (False)
        self.active_mask: torch.Tensor = torch.ones(self.batch_size, dtype=torch.bool, device=self.device)

        # --- Intervention window tracking (Batched) ---
        self.intervention_start_step: list[int | None] = [None] * self.batch_size
        self.intervention_end_step: list[int | None] = [None] * self.batch_size


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
        batch_size = scores.shape[0]

        # Safety check: if batch size changed (e.g., due to dropping finished sequences in some frameworks),
        # but HuggingFace `generate` with `padding` keeps batch size constant, we rely on padding tokens.
        # We assume batch size remains constant as initialized in InjectionState.
        
        # --- 1. Compute current token entropy H_t for the whole batch ---
        # Apply temperature scaling and softmax
        probs = F.softmax(scores / self.temperature, dim=-1)  # [batch, V]
        # Shannon entropy: H = -sum(p * log(p))
        log_probs = torch.log(probs + self.epsilon)
        H_t = -torch.sum(probs * log_probs, dim=-1)  # [batch]

        # Determine which sequences just generated EOS token
        # input_ids shape is [batch, current_length]
        # In HF generate, if a sequence hits EOS, it typically pads further generations.
        # We can dynamically detect termination if term_token_id is emitted, or rely on active_mask
        # being updated externally by run_experiment if needed. But HF generate doesn't tell LogitsProcessor
        # directly if a sequence finished. 
        # A simple heuristic: if a sequence is already converged or we see pad tokens, we could mask. 
        # For our purposes, we'll keep updating state for all sequences that are still active.
        
        # Update step_count only for active sequences
        self.state.step_count = self.state.step_count + self.state.active_mask.to(torch.float32)

        # Ensure we don't divide by zero
        safe_steps = torch.clamp(self.state.step_count, min=1.0)

        # Update entropy_sum only for active sequences
        self.state.entropy_sum = torch.where(
            self.state.active_mask,
            self.state.entropy_sum + H_t,
            self.state.entropy_sum
        )

        # Compute TECA
        self.state.teca = self.state.entropy_sum / safe_steps

        # --- 2. Compute ThinkBrake Margin M_t (if term token is set) ---
        if self.term_token_id is not None:
            log_probs_full = F.log_softmax(scores, dim=-1)  # [batch, V]
            # max_log_prob shape: [batch]
            max_log_prob, _ = log_probs_full.max(dim=-1)
            # term_log_prob shape: [batch]
            term_log_prob = log_probs_full[:, self.term_token_id]
            
            current_margin = max_log_prob - term_log_prob
            
            # Update margin where active
            self.state.margin = torch.where(
                self.state.active_mask,
                current_margin,
                self.state.margin
            )

            # Check convergence
            newly_converged = (self.state.margin <= self.margin_tau) & self.state.active_mask
            self.state.converged = self.state.converged | newly_converged

        # --- 3. Drive PID controller if TECA breaches threshold ---
        if self.pid is not None:
            # PID controller step now returns a tensor of alphas [batch]
            alpha = self.pid.step(self.state.teca, self.state.active_mask)
            self.state.alpha = alpha

            # Track intervention window (Vectorized):
            # Start: alpha > 0 and not previously active
            just_started = (alpha > 0) & (~self.state.intervention_active) & self.state.active_mask
            self.state.intervention_active = self.state.intervention_active | just_started
            
            # End: alpha == 0 and was previously active
            just_ended = (alpha <= 0) & self.state.intervention_active & self.state.active_mask
            self.state.intervention_active = self.state.intervention_active & (~just_ended)

            # Record step counts into lists
            step_counts_list = self.state.step_count.to(torch.int32).tolist()
            for i in range(batch_size):
                if just_started[i].item():
                    self.state.intervention_start_step[i] = step_counts_list[i]
                
                # If intervention is currently active, continuously update the end step
                # so if it cuts off, we have the latest step
                if self.state.intervention_active[i].item():
                    self.state.intervention_end_step[i] = step_counts_list[i]
                elif just_ended[i].item():
                    self.state.intervention_end_step[i] = step_counts_list[i]

        # --- 4. Log trajectories ---
        H_t_list = H_t.tolist()
        teca_list = self.state.teca.tolist()
        alpha_list = self.state.alpha.tolist()
        active_list = self.state.active_mask.tolist()

        for i in range(batch_size):
            if active_list[i]:
                self.state.entropy_trajectory[i].append(H_t_list[i])
                self.state.teca_trajectory[i].append(teca_list[i])
                self.state.alpha_trajectory[i].append(alpha_list[i])

        # Scores pass through unmodified — we only observe, never mutate logits
        return scores
