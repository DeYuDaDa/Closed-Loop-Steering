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
from spherical_injector import spherical_rotate


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
    def calculate(
        self,
        input_ids: torch.Tensor,
        control_vector: torch.Tensor | None = None,
        alpha_trajectory: list[float] | None = None,
        input_len: int = 0,
        layer_id: int = 16,
    ):
        """
        Compute DTR and per-token convergence depth for the full sequence.
        
        If control_vector and alpha_trajectory are provided, a replay hook is temporarily
        attached to perfectly simulate the runtime intervention.
        
        Args:
            input_ids: Token IDs, shape [batch, seq_len].
            control_vector: The normalized steering vector [1, 1, dim].
            alpha_trajectory: List of alpha values applied at each generation step.
            input_len: Number of prompt tokens (these get alpha=0).
            layer_id: The layer to attach the replay hook.

        Returns:
            dtr_scores: List of mean DTR scores (float) per batch element.
            c_t_lists: List of sequence logic convergence depths per token (for generated part).
        """
        # --- Memory Offloading Strategy ---
        # Instead of output_hidden_states=True which keeps ALL 32 layers of 
        # hidden states (seq_len*4096 dim) in VRAM at once, we:
        # 1. Ask for hidden states.
        # 2. Immediately move them to CPU to release GPU memory.
        # 3. Process layer by layer, moving only required states back to GPU.

        # Ensure we don't leak memory during the massive forward pass
        torch.cuda.empty_cache()
        
        # --- Intervention Replay Hook ---
        hook_handle = None
        if control_vector is not None and alpha_trajectory is not None:
            seq_len = input_ids.shape[1]
            # Construct alpha mask for the entire sequence
            alpha_tensor = torch.zeros(seq_len, device=input_ids.device, dtype=control_vector.dtype)
            
            # The generated tokens correspond to indices input_len to the end
            # We match the alpha_trajectory length
            traj_len = min(len(alpha_trajectory), seq_len - input_len)
            if traj_len > 0:
                alpha_tensor[input_len:input_len + traj_len] = torch.tensor(
                    alpha_trajectory[:traj_len], device=input_ids.device, dtype=control_vector.dtype
                )
            
            # Shape for broadcasting: [1, seq_len, 1]
            alpha_tensor = alpha_tensor.unsqueeze(0).unsqueeze(-1)
            
            def replay_hook(module, args, output):
                hidden = output[0] if isinstance(output, tuple) else output
                
                # Align control vector shape
                v = control_vector.to(device=hidden.device, dtype=hidden.dtype)
                if v.dim() == 1:
                    v = v.unsqueeze(0).unsqueeze(0)
                elif v.dim() == 2:
                    v = v.unsqueeze(0)
                
                # Expand to match sequence length
                v_expanded = v.expand(hidden.shape[0], hidden.shape[1], -1)
                
                # Only apply rotation where alpha > 0
                alpha_mask = alpha_tensor.to(device=hidden.device)
                is_active = (alpha_mask > 0).squeeze(-1) # [batch, seq_len]
                
                if is_active.any():
                    # Rotate all active hidden states in parallel!
                    h_new = spherical_rotate(hidden, v_expanded, alpha_mask)
                    # Use torch.where to smoothly merge rotated tokens and clean tokens
                    hidden = torch.where(is_active.unsqueeze(-1), h_new, hidden)
                
                if isinstance(output, tuple):
                    return (hidden,) + output[1:]
                return hidden
                
            layer = self.model.model.layers[layer_id]
            hook_handle = layer.register_forward_hook(replay_hook)

        try:
            outputs = self.model(input_ids, output_hidden_states=True)
        finally:
            if hook_handle is not None:
                hook_handle.remove()
        # Type note: Model output hidden states is a tuple of (L+1) tensors
        # (embeddings + L layers). We want layer 1 to L.
        # Move immediately to CPU to save VRAM
        hidden_states_cpu = [h.detach().cpu() for h in outputs.hidden_states[1:]] 
        # Fast release the massive forward pass graph
        del outputs 
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        batch_size, seq_len, dim = hidden_states_cpu[0].shape
        device = self.model.device
        
        # Only evaluate DTR on generated tokens!
        # The prompt part is invariant and not part of the model's generation choices.
        gen_len = seq_len - input_len
        if gen_len <= 0:
            return [0.0] * batch_size, [[] for _ in range(batch_size)]
        
        c_t_cpu = torch.full((batch_size, gen_len), self.L, dtype=torch.long, device="cpu")

        # Invert the loop to process in sequence chunks!
        # This prevents the LM head vocabulary matrix (V=152064) from blowing up VRAM.
        chunk_size = 256
        for start_idx in range(input_len, seq_len, chunk_size):
            end_idx = min(start_idx + chunk_size, seq_len)
            chunk_len = end_idx - start_idx
            
            # Get final layer for this chunk
            final_h_chunk = hidden_states_cpu[-1][:, start_idx:end_idx, :].to(device)
            final_logits = self.model.lm_head(final_h_chunk)
            p_t_L_chunk = F.softmax(final_logits, dim=-1)
            del final_logits, final_h_chunk
            
            min_jsd_chunk = torch.full((batch_size, chunk_len), float("inf"), device=device)
            c_t_chunk = torch.full((batch_size, chunk_len), self.L, dtype=torch.long, device=device)
            
            for l in range(self.L):
                # Move only the current layer back to GPU
                h_l_chunk = hidden_states_cpu[l][:, start_idx:end_idx, :].to(device)
                logits_l_chunk = self.model.lm_head(h_l_chunk)
                p_t_l_chunk = F.softmax(logits_l_chunk, dim=-1)
                del logits_l_chunk, h_l_chunk  # Free intermediate GPU memory

                jsd_l = self._compute_jsd_vectorized(p_t_L_chunk, p_t_l_chunk)
                del p_t_l_chunk
                
                min_jsd_chunk = torch.minimum(min_jsd_chunk, jsd_l)

                # JSD first drops below threshold
                mask = (min_jsd_chunk <= self.threshold_g) & (c_t_chunk == self.L)
                c_t_chunk[mask] = l + 1
            
            # Store chunk back to CPU tensor
            c_t_cpu[:, start_idx - input_len : end_idx - input_len] = c_t_chunk.cpu()
            
            del p_t_L_chunk, min_jsd_chunk, c_t_chunk
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        is_deep_thinking = c_t_cpu >= self.deep_thinking_threshold
        dtr_scores = is_deep_thinking.float().mean(dim=-1).tolist()
        
        c_t_lists = c_t_cpu.tolist()

        # Aggressively release the massive CPU list of 32 layer tensors (1GB+)
        del hidden_states_cpu
        del is_deep_thinking
        del c_t_cpu
        import gc
        gc.collect()

        return dtr_scores, c_t_lists

    @torch.no_grad()
    def calculate_local_dtr(
        self,
        input_ids: torch.Tensor,
        window_start: int,
        window_end: int,
        control_vector: torch.Tensor | None = None,
        alpha_trajectory: list[float] | None = None,
        input_len: int = 0,
        layer_id: int = 16,
    ) -> float:
        """
        Compute Local DTR within a specific intervention window.

        Args:
            input_ids: Full token IDs, shape [batch, seq_len].
            window_start: Start index of the intervention window.
            window_end: End index of the intervention window.
            control_vector: The normalized steering vector [1, 1, dim].
            alpha_trajectory: List of alphas applied.
            input_len: Number of prompt tokens.
            layer_id: The layer to attach the replay hook.

        Returns:
            local_dtr: DTR score within the window (0.0 to 1.0).
        """
        _, c_t_lists = self.calculate(
            input_ids, 
            control_vector=control_vector, 
            alpha_trajectory=alpha_trajectory,
            input_len=input_len,
            layer_id=layer_id
        )
        # Extract the window slice from the CPU list
        # Note: the returned list is mapped to generated tokens only!
        gen_w_start = max(0, window_start - input_len)
        gen_w_end = min(len(c_t_lists[0]), window_end - input_len)
        
        if gen_w_start >= gen_w_end:
            return 0.0

        window_c_t = c_t_lists[0][gen_w_start:gen_w_end]
        is_deep = [1 for c in window_c_t if c >= self.deep_thinking_threshold]
        local_dtr = len(is_deep) / len(window_c_t)
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