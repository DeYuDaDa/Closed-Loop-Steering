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
    ENTROPY_THRESHOLD,
    EMA_BETA,
    CONVERGENCE_MARGIN_TAU,
    COLLAPSE_ENTROPY_MIN,
    COLLAPSE_COUNT_THRESHOLD,
    N_GRAM_K,
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
        # --- Entropy State (EMA based) ---
        self.ema_entropy: torch.Tensor = torch.zeros(self.batch_size, device=self.device)
        self.step_count: torch.Tensor = torch.zeros(self.batch_size, device=self.device)

        # --- ThinkBrake state (Batched) ---
        self.margin: torch.Tensor = torch.full((self.batch_size,), float("inf"), device=self.device)

        # --- PID output (written by PIDController, read by Hook) ---
        self.alpha: torch.Tensor = torch.zeros(self.batch_size, device=self.device)

        # --- Trajectory logging (Batched) ---
        # List of lists, outer list is per-sequence in the batch
        self.ema_trajectory: list[list[float]] = [[] for _ in range(self.batch_size)]
        self.alpha_trajectory: list[list[float]] = [[] for _ in range(self.batch_size)]
        self.entropy_trajectory: list[list[float]] = [[] for _ in range(self.batch_size)]

        # --- Flags (Batched) ---
        self.intervention_active: torch.Tensor = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)
        self.is_converged: torch.Tensor = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)
        
        # Track whether a sequence is still generating (True) or has hit EOS (False)
        self.active_mask: torch.Tensor = torch.ones(self.batch_size, dtype=torch.bool, device=self.device)

        # --- Intervention window tracking (Batched) ---
        self.intervention_start_step: list[int | None] = [None] * self.batch_size
        self.intervention_end_step: list[int | None] = [None] * self.batch_size

        # --- Anti-Collapse Watchdog (Batched) ---
        self.low_entropy_count: torch.Tensor = torch.zeros(self.batch_size, dtype=torch.int32, device=self.device)
        self.trigger_perturbation: torch.Tensor = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)
        self.cooldown_counter: torch.Tensor = torch.zeros(self.batch_size, dtype=torch.int32, device=self.device)


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
        entropy_threshold: float = ENTROPY_THRESHOLD,
        ema_beta: float = EMA_BETA,
        margin_tau: float = CONVERGENCE_MARGIN_TAU,
        use_raw_entropy: bool = False,
    ):
        """
        Args:
            state: Shared InjectionState object.
            pid_controller: Optional controller instance (PIDController or TAEController).
                            If provided, controller.step(entropy) is called automatically.
            term_token_id: Token ID for </think> (for ThinkBrake margin).
            temperature: Softmax temperature for entropy calculation.
            epsilon: Small constant for numerical stability.
            teca_threshold: TECA threshold that triggers intervention.
            margin_tau: Margin threshold for convergence detection.
            use_raw_entropy: If True, pass raw instantaneous H_t to the controller
                             instead of the EMA-smoothed entropy. Used by TAE modes.
        """
        self.state = state
        self.pid = pid_controller
        self.term_token_id = term_token_id
        self.temperature = temperature
        self.epsilon = epsilon
        self.entropy_threshold = entropy_threshold
        self.ema_beta = ema_beta
        self.margin_tau = margin_tau
        self.use_raw_entropy = use_raw_entropy


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
        # probs = F.softmax(scores / self.temperature, dim=-1)  # [batch, V]
        # # Shannon entropy: H = -sum(p * log(p))
        # log_probs = torch.log(probs + self.epsilon)
        # H_t = -torch.sum(probs * log_probs, dim=-1)  # [batch]
        # --- 1. Compute current token entropy H_t for the whole batch ---
        # 必须使用 T=1.0 的原始 Logits 来真实反映模型的迷茫度
        probs = F.softmax(scores, dim=-1)         # [batch, V]
        log_probs = F.log_softmax(scores, dim=-1) # 使用原生 API 保证数值稳定
        
        # Shannon entropy: H = -sum(p * log(p))
        H_t = -torch.sum(probs * log_probs, dim=-1)  # [batch]


        # Determine which sequences just generated EOS token
        # input_ids shape is [batch, current_length]
        # In HF generate, if a sequence hits EOS, it typically pads further generations.
        # We can dynamically detect termination if term_token_id is emitted, or rely on active_mask
        # being updated externally by run_experiment if needed. But HF generate doesn't tell LogitsProcessor
        # directly if a sequence finished. 
        # A simple heuristic: if a sequence is already converged or we see pad tokens, we could mask. 
        # For our purposes, we'll keep updating state for all sequences that are still active.
        
        # --- 1. Compute EMA Entropy ---
        is_first_step = (self.state.step_count == 0)
        self.state.ema_entropy = torch.where(
            is_first_step,
            H_t,
            self.ema_beta * H_t + (1.0 - self.ema_beta) * self.state.ema_entropy
        )
        
        # Update step_count only for active sequences
        self.state.step_count = self.state.step_count + self.state.active_mask.to(torch.float32)

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

            # Check convergence (ThinkBrake Latch)
            just_converged = (self.state.margin <= self.margin_tau) & self.state.active_mask
            self.state.is_converged = self.state.is_converged | just_converged

        # --- 3. Drive controller (PID or TAE) if present ---
        if self.pid is not None:
            # TAE uses raw H_t; PID uses EMA-smoothed entropy
            controller_input = (
                H_t  # raw instantaneous entropy
                if self.use_raw_entropy
                else self.state.ema_entropy
            )
            alpha = self.pid.step(controller_input, self.state.active_mask, self.state.is_converged)
            
            # --- 3.1 Anti-Collapse Watchdog (Repetition + Low Entropy Detection) ---
            # 1. Low Entropy Count
            is_low_entropy = (H_t < COLLAPSE_ENTROPY_MIN) & self.state.active_mask
            self.state.low_entropy_count = torch.where(
                is_low_entropy,
                self.state.low_entropy_count + 1,
                torch.zeros_like(self.state.low_entropy_count)
            )
            
            # 2. Local Repetition Detection (N-Gram cycle check over last max_period tokens)
            is_repeating = torch.zeros(batch_size, dtype=torch.bool, device=self.state.device)
            seq_len = input_ids.shape[1]
            max_period = N_GRAM_K * 2
            for p in range(1, max_period + 1):
                if seq_len >= 2 * p:
                    match = (input_ids[:, -p:] == input_ids[:, -2*p:-p]).all(dim=-1)
                    is_repeating = is_repeating | match

            # 3. Trigger condition
            should_trigger = (self.state.low_entropy_count > COLLAPSE_COUNT_THRESHOLD) & is_repeating & self.state.active_mask
            
            # We only trigger if NOT currently on cooldown
            self.state.trigger_perturbation = should_trigger & (self.state.cooldown_counter == 0)
            
            # 4. Handle Cooldown overrides
            in_cooldown = (self.state.cooldown_counter > 0) & self.state.active_mask
            self.state.cooldown_counter = torch.where(
                in_cooldown,
                self.state.cooldown_counter - 1,
                torch.zeros_like(self.state.cooldown_counter)
            )
            
            # If triggered or in cooldown, PID alpha is forcefully suppressed to 0, 
            # allowing the model (or perturbation) to navigate freely without PID drag.
            suppress_mask = self.state.trigger_perturbation | in_cooldown
            alpha = torch.where(suppress_mask, torch.zeros_like(alpha), alpha)
            
            self.state.alpha = alpha

            # Track intervention window (Vectorized):
            # Start: alpha > 0 and not previously active
            just_started = (alpha > 0) & (~self.state.intervention_active) & self.state.active_mask
            self.state.intervention_active = self.state.intervention_active | just_started
            
            # End: alpha == 0 and was previously active
            just_ended = (alpha <= 0) & self.state.intervention_active & self.state.active_mask
            self.state.intervention_active = self.state.intervention_active & (~just_ended)

            # --- 4. ThinkBrake Hard Cutoff for active_mask ---
            # If converged, we stop intervention conceptually by masking (or we could just let PID handle it)
            # In the user request, it says: self.state.active_mask = (self.state.ema_entropy > ENTROPY_THRESHOLD) & (~self.state.is_converged)
            # Actually, active_mask in HF usually means "sequence is still generating". 
            # The user request might mean a mask for the INTERVENTION.
            # Let's adjust active_mask for intervention trigger:
            intervention_trigger_mask = (self.state.ema_entropy > self.entropy_threshold) & (~self.state.is_converged)
            # We already passed is_converged to PID, so PID output will be 0 if converged.

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

        # --- 5. Log trajectories ---
        H_t_list = H_t.tolist()
        ema_list = self.state.ema_entropy.tolist()
        alpha_list = self.state.alpha.tolist()
        active_list = self.state.active_mask.tolist()
        converged_list = self.state.is_converged.tolist()

        for i in range(batch_size):
            if active_list[i]:
                self.state.entropy_trajectory[i].append(H_t_list[i])
                # Only record if not converged (or record 0 as requested)
                if not converged_list[i]:
                    self.state.ema_trajectory[i].append(ema_list[i])
                else:
                    self.state.ema_trajectory[i].append(0.0)
                self.state.alpha_trajectory[i].append(alpha_list[i])

        # Scores pass through unmodified — we only observe, never mutate logits
        return scores
