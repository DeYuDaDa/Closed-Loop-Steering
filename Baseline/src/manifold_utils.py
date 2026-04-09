"""
Module 2: Manifold Projection via PCA
=======================================
Offline tool that purifies CAA-extracted control vectors by projecting
them onto the principal component subspace of normal inference activations.

Reference:
  - Manifold Steering: Mitigating Overthinking via Manifold Steering

Mathematical formulation:
    Given raw vector v_raw ∈ R^d and activation matrix A ∈ R^{N×d}:
    1. PCA on centered A → top-k eigenvectors U_k = [u_1, ..., u_k]
    2. v_manifold = Σ_{i=1}^{k} (v_raw · u_i) u_i
"""

import torch
import numpy as np
from sklearn.decomposition import PCA
from typing import Optional

from config import PCA_N_COMPONENTS, TEMPERATURE, TOP_P


class ManifoldProjector:
    """
    Projects control vectors onto the principal component subspace
    of the model's normal inference hidden state activations.
    """

    def __init__(self, n_components: int = PCA_N_COMPONENTS):
        self.n_components = n_components
        self.pca: Optional[PCA] = None
        self.components: Optional[np.ndarray] = None  # shape: [k, d]

    def fit(self, activations: np.ndarray):
        """
        Fit PCA on collected hidden state activations.

        Args:
            activations: Numpy array of shape [N, d] where N is the number
                         of collected activation vectors and d is hidden dim.
        """
        self.pca = PCA(n_components=self.n_components)
        self.pca.fit(activations)
        self.components = self.pca.components_  # shape: [k, d]
        print(f"[ManifoldProjector] Fitted PCA with {self.n_components} components.")
        print(f"  Explained variance ratio: {self.pca.explained_variance_ratio_.sum():.4f}")

    def purify_vector(self, v_raw: torch.Tensor) -> torch.Tensor:
        """
        Project v_raw onto the PCA subspace.

        Args:
            v_raw: Raw CAA vector, shape [d] or [1, d] or [1, 1, d].

        Returns:
            v_purified: Projected vector with same shape as input.
        """
        if self.components is None:
            raise RuntimeError("PCA not fitted yet. Call .fit() first.")

        original_shape = v_raw.shape
        device = v_raw.device
        dtype = v_raw.dtype

        # Flatten to [d]
        v = v_raw.detach().cpu().float().numpy().flatten()

        # Projection: v_manifold = Σ (v · u_i) * u_i
        U = self.components  # [k, d]
        coefficients = U @ v  # [k] — dot products (v · u_i)
        v_manifold = coefficients @ U  # [d] — sum of projections

        # Convert back to torch and restore shape
        v_purified = torch.tensor(v_manifold, dtype=dtype, device=device)
        v_purified = v_purified.view(original_shape)

        return v_purified

    def save_components(self, path: str):
        """Save PCA components to disk."""
        if self.components is None:
            raise RuntimeError("PCA not fitted yet.")
        np.save(path, self.components)
        print(f"[ManifoldProjector] Components saved to {path}")

    def load_components(self, path: str):
        """Load PCA components from disk."""
        self.components = np.load(path)
        self.n_components = self.components.shape[0]
        print(f"[ManifoldProjector] Loaded {self.n_components} components from {path}")


def collect_activations_from_model(
    model,
    tokenizer,
    prompts: list[str],
    layer_id: int,
    max_tokens: int = 128,
) -> np.ndarray:
    """
    Utility: run model on a set of prompts and collect hidden state
    activations at a specific layer. Used to fit the ManifoldProjector.

    Args:
        model: HuggingFace causal LM.
        tokenizer: Corresponding tokenizer.
        prompts: List of prompt strings.
        layer_id: Which transformer layer to hook.
        max_tokens: Max new tokens to generate per prompt.

    Returns:
        activations: Numpy array of shape [total_tokens, hidden_dim].
    """
    all_activations = []

    def collect_hook(module, args, output):
        hidden = output[0] if isinstance(output, tuple) else output
        # Collect all token positions (not just last)
        all_activations.append(hidden.detach().cpu().float())
        return output

    layer = model.model.layers[layer_id]
    handle = layer.register_forward_hook(collect_hook)

    with torch.no_grad():
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                pad_token_id=tokenizer.eos_token_id,
            )

    handle.remove()

    # Concatenate all collected activations: list of [batch, seq, dim] → [N, dim]
    stacked = torch.cat(all_activations, dim=0)  # may be [B, seq, dim]
    if stacked.dim() == 3:
        stacked = stacked.view(-1, stacked.shape[-1])  # [N, dim]

    return stacked.numpy()


def purify_and_save(
    raw_vector_path: str,
    activations: np.ndarray,
    output_path: str,
    n_components: int = PCA_N_COMPONENTS,
):
    """
    End-to-end convenience: load raw vector → fit PCA → purify → save.

    Args:
        raw_vector_path: Path to raw CAA vector (.pt file).
        activations: Numpy array of hidden state activations [N, d].
        output_path: Where to save the purified vector (.pt file).
        n_components: Number of PCA components.
    """
    # Load raw vector
    v_raw = torch.load(raw_vector_path, map_location="cpu")
    print(f"[purify_and_save] Loaded raw vector: shape={v_raw.shape}")

    # Fit PCA and project
    projector = ManifoldProjector(n_components=n_components)
    projector.fit(activations)
    v_purified = projector.purify_vector(v_raw)

    # Normalize the purified vector
    v_purified_flat = v_purified.view(-1)
    v_purified_flat = v_purified_flat / v_purified_flat.norm()
    v_purified = v_purified_flat.view(v_raw.shape)

    # Save
    torch.save(v_purified, output_path)
    print(f"[purify_and_save] Purified vector saved to {output_path}")
    print(f"  Raw norm: {v_raw.view(-1).norm().item():.4f}")
    print(f"  Purified norm: {v_purified.view(-1).norm().item():.4f}")
