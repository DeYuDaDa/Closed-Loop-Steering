import argparse
import os
import re
import json
import random
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import trange, tqdm
import sys

# Add project root and SEAL directory to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
seal_dir = os.path.join(project_root, "src/SEAL-main/SEAL-main")
sys.path.append(project_root)
sys.path.append(seal_dir)

from util.loaders.aime_loader import load_aime_dataset
from util.loaders.math500_loader import load_math500_dataset
from get_math_results import main as eval_main

def trim_output(output):
    for prefix in ["Answer the following question", "Question:", "Comment:"]:
        if prefix in output:
            output = output.split(prefix)[0]
    return output

def main(args):
    random.seed(42)
    torch.manual_seed(42)

    print(f"Loading data for {args.dataset}...")
    if args.dataset == "MATH500":
        data_path = os.path.join(project_root, "src/util/dataset/MATH500.jsonl")
        test_data = load_math500_dataset(data_path)
    elif args.dataset == "AIME":
        data_path = os.path.join(project_root, f"src/util/dataset/{args.aime_file}")
        test_data = load_aime_dataset(data_path)
    else:
        raise ValueError(f"Dataset {args.dataset} not supported")

    if args.start:
        test_data = test_data[args.start:]
    if args.max_examples:
        test_data = test_data[:args.max_examples]

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model from {args.model_name_or_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, 
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    )
    model.eval()

    num_layers = model.config.num_hidden_layers
    
    # Load all layer vectors
    print(f"Loading layer-wise vectors from {args.vectors_dir}...")
    steering_vectors = {}
    for l in range(num_layers + 1):
        v_path = os.path.join(args.vectors_dir, f"layer_{l}_purified.pt")
        if os.path.exists(v_path):
            steering_vectors[l] = torch.load(v_path).to(model.device).to(torch.bfloat16)
        else:
            print(f"  Warning: Layer {l} vector not found.")

    # Intervention hook
    def get_manifold_hook(layer_idx):
        def hook(module, input, output):
            if layer_idx not in steering_vectors:
                return output
            
            h = output[0] if isinstance(output, tuple) else output
            v = steering_vectors[layer_idx]
            
            # Formula: h' = h - alpha * v * (v^T h)
            # h: [B, S, D], v: [D]
            dot_products = torch.matmul(h, v) # [B, S]
            
            # Use outer product for efficiency if possible or just multiply
            # projection = dot_products.unsqueeze(-1) * v # [B, S, D]
            # Equivalent but explicit
            projection = torch.outer(dot_products.reshape(-1), v).reshape(h.shape)
            
            h_new = h - args.alpha * projection
            
            return (h_new,) if isinstance(output, tuple) else h_new
        return hook

    # Register hooks for all layers
    handles = []
    print(f"Registering intervention hooks on all {num_layers} layers (alpha={args.alpha})...")
    for l in range(num_layers):
        # Apply to the end of each transformer block
        layer = model.model.layers[l]
        handles.append(layer.register_forward_hook(get_manifold_hook(l + 1))) # Paper uses residuals after layer

    prefix = "Answer the following questions. You should think step-by-step and put your final answer within \\boxed{}.\n"
    prompts = []
    for example in test_data:
        prompt = prefix + "Question: " + example["problem"].strip() + "\nAnswer: <think>\n"
        prompts.append(prompt)

    results = []
    for i in trange(0, len(prompts), args.batch_size, desc="Generating"):
        batch = prompts[i:i+args.batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            output = model.generate(
                **inputs, 
                max_new_tokens=args.max_tokens, 
                do_sample=False, 
                use_cache=True
            )
        
        prompt_len = inputs["input_ids"].shape[1]
        for out in output:
            gen_text = tokenizer.decode(out[prompt_len:], skip_special_tokens=True)
            results.append(trim_output(gen_text))

    for h in handles:
        h.remove()

    predictions = []
    for example, gen_text, prompt in zip(test_data, results, prompts):
        predictions.append({
            "prompt": prompt,
            "problem": example["problem"],
            "answer": example["answer"],
            "model_generation": [gen_text],
        })

    save_path = os.path.join(args.save_dir, "predictions.jsonl")
    with open(save_path, "w") as f:
        for p in predictions:
            f.write(json.dumps(p) + "\n")
    
    print(f"Predictions saved to {save_path}")
    eval_main(save_path, save=True, output_dir=args.save_dir, dataset=args.dataset)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--vectors_dir", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="AIME")
    parser.add_argument("--aime_file", type=str, default="aime2024.jsonl")
    parser.add_argument("--save_dir", type=str, default="results/manifold")
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--start", type=int, default=0)
    args = parser.parse_args()
    main(args)
