"""
Module 5: DTR Evaluator — Deep-Thinking Ratio Calculator
==========================================================
Post-hoc evaluation tool that measures how deeply the model "thinks"
at each generated token, using JSD-based convergence depth analysis.

Enhanced with:
  - Local DTR (for intervention window analysis)
  - PPL (perplexity) calculation
  - Repetition Rate (n-gram overlap)

Reference:
  - Think Deep, Not Just Long: Measuring LLM Reasoning Effort
    via Deep-Thinking Tokens (Algorithm 1)
"""

import torch
import torch.nn.functional as F
import math
import numpy as np
from collections import Counter

from config import DTR_G, DTR_RHO, REPETITION_NGRAM


class DTRCalculator:
    """
    JSD-based Deep-Thinking Ratio calculator.

    For each token t in a sequence:
      1. Get hidden states h_{t,l} at all layers l
      2. Un-embed to get p_{t,l} = Softmax(W_U · h_{t,l})
      3. Compute JSD between final-layer and each layer
      4. Find convergence depth c_t (first layer where min JSD ≤ g)
      5. Token is "deep-thinking" if c_t ≥ ⌈ρL⌉
    """

    def __init__(self, model, g: float = DTR_G, rho: float = DTR_RHO):
        """
        Args:
            model: HuggingFace causal LM with lm_head.
            g: Convergence threshold (default 0.5).
            rho: Deep-thinking layer fraction (default 0.85).
        """
        self.model = model
        self.L = model.config.num_hidden_layers
        self.threshold_g = g
        self.deep_thinking_threshold = math.ceil(rho * self.L)

    @torch.no_grad()
    def _compute_jsd_vectorized(self, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """
        Compute Jensen-Shannon Divergence between two distributions.

        Args:
            p, q: Probability tensors of shape [..., V].

        Returns:
            JSD values of shape [...].
        """
        m = 0.5 * (p + q)
        eps = 1e-9
        h_m = -torch.sum(m * torch.log(m + eps), dim=-1)
        h_p = -torch.sum(p * torch.log(p + eps), dim=-1)
        h_q = -torch.sum(q * torch.log(q + eps), dim=-1)
        return h_m - 0.5 * h_p - 0.5 * h_q

    @torch.no_grad()
    def calculate(self, input_ids: torch.Tensor):
        """
        Compute DTR and per-token convergence depth for the full sequence.

        Args:
            input_ids: Token IDs, shape [batch, seq_len].

        Returns:
            dtr_scores: List of DTR scores (float) per batch element.
            c_t: Convergence depth matrix, shape [batch, seq_len].
        """
        outputs = self.model(input_ids, output_hidden_states=True)
        hidden_states = outputs.hidden_states[1:]  # Skip embedding layer

        # Final layer distribution
        final_h = hidden_states[-1]
        final_logits = self.model.lm_head(final_h)
        p_t_L = F.softmax(final_logits, dim=-1)

        batch_size, seq_len, _ = p_t_L.shape
        device = p_t_L.device

        min_jsd_so_far = torch.full((batch_size, seq_len), float("inf"), device=device)
        c_t = torch.full((batch_size, seq_len), self.L, dtype=torch.long, device=device)

        for l in range(self.L):
            logits_t_l = self.model.lm_head(hidden_states[l])
            p_t_l = F.softmax(logits_t_l, dim=-1)

            jsd_l = self._compute_jsd_vectorized(p_t_L, p_t_l)
            min_jsd_so_far = torch.minimum(min_jsd_so_far, jsd_l)

            # JSD first drops below threshold
            mask = (min_jsd_so_far <= self.threshold_g) & (c_t == self.L)
            c_t[mask] = l + 1

        is_deep_thinking = c_t >= self.deep_thinking_threshold
        dtr_scores = is_deep_thinking.float().mean(dim=-1)

        return dtr_scores.cpu().tolist(), c_t.cpu()

    @torch.no_grad()
    def calculate_local_dtr(
        self,
        input_ids: torch.Tensor,
        window_start: int,
        window_end: int,
    ) -> float:
        """
        Compute Local DTR within a specific intervention window.

        Args:
            input_ids: Full token IDs, shape [batch, seq_len].
            window_start: Start index of the intervention window.
            window_end: End index of the intervention window.

        Returns:
            local_dtr: DTR score within the window (0.0 to 1.0).
        """
        _, c_t = self.calculate(input_ids)
        # Extract the window slice
        window_c_t = c_t[0, window_start:window_end]  # [window_len]
        is_deep = window_c_t >= self.deep_thinking_threshold
        local_dtr = is_deep.float().mean().item()
        return local_dtr


def calculate_ppl(model, tokenizer, text: str) -> float:
    """
    Calculate perplexity (PPL) of a given text.

    PPL = exp(average cross-entropy loss per token).

    Args:
        model: HuggingFace causal LM.
        tokenizer: Corresponding tokenizer.
        text: Text string to evaluate.

    Returns:
        ppl: Perplexity value.
    """
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    input_ids = inputs.input_ids

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss  # Average cross-entropy per token

    ppl = math.exp(loss.item())
    return ppl


def calculate_repetition_rate(text: str, n: int = REPETITION_NGRAM) -> float:
    """
    Calculate n-gram repetition rate.

    Repetition Rate = 1 - (unique n-grams / total n-grams)

    A value of 0.0 means no repetition; 1.0 means all n-grams are repeated.

    Args:
        text: Text string to analyze.
        n: N-gram size (default: 4).

    Returns:
        rep_rate: Repetition rate (0.0 to 1.0).
    """
    words = text.split()
    if len(words) < n:
        return 0.0

    ngrams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    total = len(ngrams)
    unique = len(set(ngrams))

    if total == 0:
        return 0.0

    rep_rate = 1.0 - (unique / total)
    return rep_rate