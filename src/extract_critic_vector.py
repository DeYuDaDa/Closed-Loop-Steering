"""
Extract Critic Vector — Standalone CAA Extraction Script
==========================================================
Offline tool that extracts a Contrastive Activation Addition (CAA)
control vector from paired critic data, then purifies it via
Manifold Projection (PCA).

Usage (run on GPU server):
    cd /path/to/src
    python extract_critic_vector.py

Inputs:
    - critic_data.json   : list of {prompt, pos_completion, neg_completion}
    - config.py          : MODEL_PATH, LAYER_ID, VECTOR_DIR, PCA_N_COMPONENTS

Outputs (saved to VECTOR_DIR):
    - critic_raw.pt          : raw CAA vector  (shape [d])
    - critic.pt              : PCA-purified vector (shape [d])
    - pca_components.npy     : saved PCA principal components
"""

import os
import json
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from config import MODEL_PATH, LAYER_ID, VECTOR_DIR, PCA_N_COMPONENTS
from manifold_utils import ManifoldProjector


# ------------------------------------------------------------------ #
#  Activation collection                                              #
# ------------------------------------------------------------------ #

def collect_last_token_activation(
    model,
    tokenizer,
    text: str,
    layer_id: int,
) -> torch.Tensor:
    """
    Run a single forward pass and return the hidden state of the
    **last token** at the specified layer.

    Args:
        model: HuggingFace CausalLM.
        tokenizer: Corresponding tokenizer.
        text: Full input text (prompt + completion).
        layer_id: Transformer layer to hook.

    Returns:
        Tensor of shape [hidden_dim].
    """
    activation_store = {}

    def hook_fn(module, args, output):
        hidden = output[0] if isinstance(output, tuple) else output
        # Take the last token's hidden state
        activation_store["last"] = hidden[:, -1, :].detach().cpu().float()
        return output

    layer = model.model.layers[layer_id]
    handle = layer.register_forward_hook(hook_fn)

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        model(**inputs)

    handle.remove()
    return activation_store["last"].squeeze(0)  # [hidden_dim]


def extract_caa_vector(
    model,
    tokenizer,
    data: list[dict],
    layer_id: int,
) -> tuple[torch.Tensor, np.ndarray]:
    """
    Extract the Contrastive Activation Addition (CAA) vector.

    For each data pair:
        pos_activation = hidden_state(prompt + pos_completion)[-1]
        neg_activation = hidden_state(prompt + neg_completion)[-1]

    CAA vector = mean(pos_activations) - mean(neg_activations)

    Also returns all collected activations (pos + neg) for PCA fitting.

    Args:
        model: HuggingFace CausalLM.
        tokenizer: Corresponding tokenizer.
        data: List of dicts with keys: prompt, pos_completion, neg_completion.
        layer_id: Target transformer layer.

    Returns:
        v_raw: Raw CAA vector, shape [hidden_dim].
        all_activations: Numpy array of all activations [2*N, hidden_dim].
    """
    pos_activations = []
    neg_activations = []

    print(f"\n{'='*60}")
    print(f"  Extracting CAA vector from {len(data)} pairs at layer {layer_id}")
    print(f"{'='*60}\n")

    for i, item in enumerate(tqdm(data, desc="Collecting activations")):
        prompt = item["prompt"]
        pos_text = prompt + " " + item["pos_completion"]
        neg_text = prompt + " " + item["neg_completion"]

        pos_act = collect_last_token_activation(model, tokenizer, pos_text, layer_id)
        neg_act = collect_last_token_activation(model, tokenizer, neg_text, layer_id)

        pos_activations.append(pos_act)
        neg_activations.append(neg_act)

    # Stack into tensors
    pos_stack = torch.stack(pos_activations)  # [N, d]
    neg_stack = torch.stack(neg_activations)  # [N, d]

    # CAA: mean difference
    v_raw = pos_stack.mean(dim=0) - neg_stack.mean(dim=0)  # [d]

    # Combine all activations for PCA (both pos & neg represent normal inference)
    all_acts = torch.cat([pos_stack, neg_stack], dim=0).numpy()  # [2N, d]

    print(f"\n  ✅ Raw CAA vector extracted:")
    print(f"    shape = {list(v_raw.shape)}")
    print(f"    norm  = {v_raw.norm().item():.4f}")
    print(f"  Activations collected: {all_acts.shape[0]} vectors of dim {all_acts.shape[1]}")

    return v_raw, all_acts


# ------------------------------------------------------------------ #
#  Main script                                                        #
# ------------------------------------------------------------------ #

def main():
    # ---- Load data ----
    data_path = os.path.join(os.path.dirname(__file__), "critic_data.json")
    print(f"Loading critic data from {data_path}...")
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"  Loaded {len(data)} prompt-completion pairs.")

    # ---- Load model ----
    print(f"\nLoading model from {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    print(f"  Model loaded. Device: {model.device}")

    # ---- Extract raw CAA vector ----
    v_raw, all_activations = extract_caa_vector(
        model, tokenizer, data, LAYER_ID
    )

    # ---- Save raw vector ----
    os.makedirs(VECTOR_DIR, exist_ok=True)
    raw_path = os.path.join(VECTOR_DIR, "critic_raw.pt")
    torch.save(v_raw, raw_path)
    print(f"\n  💾 Raw vector saved to {raw_path}")

    # ---- PCA purification via ManifoldProjector ----
    print(f"\n{'='*60}")
    print(f"  Manifold Projection (PCA purification)")
    print(f"{'='*60}")

    projector = ManifoldProjector(n_components=PCA_N_COMPONENTS)
    projector.fit(all_activations)

    # Save PCA components for future use
    pca_path = os.path.join(VECTOR_DIR, "pca_components.npy")
    projector.save_components(pca_path)

    # Purify the vector
    v_purified = projector.purify_vector(v_raw)

    # Record pre-normalization norm for coeff calibration
    purified_pre_norm = v_purified.view(-1).norm().item()

    # Normalize the purified vector
    v_purified_flat = v_purified.view(-1)
    v_purified_flat = v_purified_flat / v_purified_flat.norm()
    v_purified = v_purified_flat.view(v_raw.shape)

    # Save purified vector
    purified_path = os.path.join(VECTOR_DIR, "critic.pt")
    torch.save(v_purified, purified_path)
    print(f"\n  💾 Purified vector saved to {purified_path}")

    # ---- Save norm metadata for coeff calibration ----
    norm_metadata = {
        "raw_caa_norm": v_raw.norm().item(),
        "purified_norm_before_normalize": purified_pre_norm,
        "purified_norm_after_normalize": v_purified.view(-1).norm().item(),
        "mean_activation_norm": float(
            np.linalg.norm(all_activations, axis=1).mean()
        ),
        "std_activation_norm": float(
            np.linalg.norm(all_activations, axis=1).std()
        ),
        "layer_id": LAYER_ID,
        "pca_n_components": PCA_N_COMPONENTS,
        "pca_explained_variance": float(
            projector.pca.explained_variance_ratio_.sum()
        ),
        "note": (
            "The purified vector (critic.pt) is L2-normalized to unit norm. "
            "Use raw_caa_norm or purified_norm_before_normalize to calibrate "
            "the coeff parameter in VectorInjector. A reasonable starting "
            "coeff is purified_norm_before_normalize / mean_activation_norm."
        ),
    }
    metadata_path = os.path.join(VECTOR_DIR, "norm_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(norm_metadata, f, indent=2, ensure_ascii=False)
    print(f"  💾 Norm metadata saved to {metadata_path}")

    # ---- Summary ----
    print(f"\n{'='*60}")
    print(f"  EXTRACTION COMPLETE — Summary")
    print(f"{'='*60}")
    print(f"  Data pairs used:        {len(data)}")
    print(f"  Target layer:           {LAYER_ID}")
    print(f"  Hidden dim:             {v_raw.shape[0]}")
    print(f"  PCA components:         {PCA_N_COMPONENTS}")
    print(f"  Raw vector norm:        {v_raw.norm().item():.4f}")
    print(f"  Purified norm (pre-L2): {purified_pre_norm:.4f}")
    print(f"  Purified vector norm:   {v_purified.view(-1).norm().item():.4f}")
    print(f"  Mean activation norm:   {norm_metadata['mean_activation_norm']:.4f}")
    print(f"  Explained variance:     {projector.pca.explained_variance_ratio_.sum():.4f}")
    print(f"\n  Output files:")
    print(f"    {raw_path}")
    print(f"    {purified_path}")
    print(f"    {pca_path}")
    print(f"    {metadata_path}")
    print(f"\n  🎉 Ready for use in run_experiment.py!")


if __name__ == "__main__":
    main()
