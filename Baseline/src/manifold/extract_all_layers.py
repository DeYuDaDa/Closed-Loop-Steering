import os
import json
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.config import MODEL_PATH, VECTOR_DIR, PCA_N_COMPONENTS
from src.manifold_utils import ManifoldProjector

def extract_all_layers_caa(model, tokenizer, data):
    """
    Extract CAA vectors for all layers in one go using output_hidden_states=True.
    """
    num_layers = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size
    
    # Accumulators for each layer
    all_pos_activations = [[] for _ in range(num_layers + 1)] # including embedding if needed, but layers are 1 to num_layers
    all_neg_activations = [[] for _ in range(num_layers + 1)]

    print(f"Collecting activations for {len(data)} pairs across {num_layers} layers...")
    
    for item in tqdm(data, desc="Processing pairs"):
        prompt = item["prompt"]
        pos_text = prompt + " " + item["pos_completion"]
        neg_text = prompt + " " + item["neg_completion"]
        
        # Process positive
        inputs = tokenizer(pos_text, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states # tuple of (batch, seq, dim)
            for l, h in enumerate(hidden_states):
                all_pos_activations[l].append(h[0, -1, :].detach().cpu().float())
        
        # Process negative
        inputs = tokenizer(neg_text, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            for l, h in enumerate(hidden_states):
                all_neg_activations[l].append(h[0, -1, :].detach().cpu().float())

    # Calculate CAA vectors for each layer
    caa_vectors = []
    layer_wise_all_acts = []
    
    for l in range(num_layers + 1):
        pos_stack = torch.stack(all_pos_activations[l])
        neg_stack = torch.stack(all_neg_activations[l])
        v_raw = pos_stack.mean(dim=0) - neg_stack.mean(dim=0)
        caa_vectors.append(v_raw)
        layer_wise_all_acts.append(torch.cat([pos_stack, neg_stack], dim=0).numpy())
        
    return caa_vectors, layer_wise_all_acts

def main():
    data_path = os.path.join(project_root, "src/critic_data.json")
    if not os.path.exists(data_path):
        # Fallback if not in src
        data_path = os.path.join(project_root, "critic_data.json")
        
    print(f"Loading critic data from {data_path}...")
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Loading model from {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, 
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    )
    model.eval()

    caa_vectors, layer_wise_all_acts = extract_all_layers_caa(model, tokenizer, data)

    # Save directory
    layer_dir = os.path.join(VECTOR_DIR, "layer_wise")
    os.makedirs(layer_dir, exist_ok=True)

    print("Purifying and saving vectors for all layers...")
    for l in range(len(caa_vectors)):
        # Skip layer 0 (embedding) if preferred, but following "all layers"
        v_raw = caa_vectors[l]
        acts = layer_wise_all_acts[l]
        
        projector = ManifoldProjector(n_components=PCA_N_COMPONENTS)
        projector.fit(acts)
        
        v_purified = projector.purify_vector(v_raw)
        
        # Normalize
        v_purified = v_purified / v_purified.norm()
        
        save_path = os.path.join(layer_dir, f"layer_{l}_purified.pt")
        torch.save(v_purified, save_path)
        
        if l % 5 == 0:
            print(f"  Layer {l} saved.")

    print(f"Done! Vectors saved to {layer_dir}")

if __name__ == "__main__":
    main()
