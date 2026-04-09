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

from src.config import MODEL_PATH, LAYER_ID, VECTOR_DIR

def collect_last_token_activation(model, tokenizer, text, layer_id):
    activation_store = {}
    def hook_fn(module, args, output):
        hidden = output[0] if isinstance(output, tuple) else output
        activation_store["last"] = hidden[:, -1, :].detach().cpu().float()
        return output

    layer = model.model.layers[layer_id]
    handle = layer.register_forward_hook(hook_fn)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        model(**inputs)
    handle.remove()
    return activation_store["last"].squeeze(0)

def main():
    data_path = os.path.join(project_root, "src/critic_data.json")
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Loading model from {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()

    pos_activations = []
    neg_activations = []

    for item in tqdm(data, desc="Extracting Prototypes"):
        prompt = item["prompt"]
        pos_text = prompt + " " + item["pos_completion"]
        neg_text = prompt + " " + item["neg_completion"]
        pos_activations.append(collect_last_token_activation(model, tokenizer, pos_text, LAYER_ID))
        neg_activations.append(collect_last_token_activation(model, tokenizer, neg_text, LAYER_ID))

    mu_T = torch.stack(pos_activations).mean(dim=0)
    mu_H = torch.stack(neg_activations).mean(dim=0)

    # Normalize to unit vectors for spherical steering
    mu_T = mu_T / mu_T.norm()
    mu_H = mu_H / mu_H.norm()

    save_path = os.path.join(VECTOR_DIR, "prototypes.pt")
    torch.save({"mu_T": mu_T, "mu_H": mu_H}, save_path)
    print(f"Prototypes saved to {save_path}")

if __name__ == "__main__":
    main()
