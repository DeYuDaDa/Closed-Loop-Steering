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

from util.loaders.aime_loader import load_aime_dataset, build_aime_prompt
from util.loaders.math500_loader import load_math500_dataset, build_math500_prompt
from get_math_results import main as eval_main
from spherical_steering import get_spherical_intervention
from baukit import TraceDict

def trim_output(output):
    for prefix in ["Answer the following question", "Question:", "Comment:"]:
        if prefix in output:
            output = output.split(prefix)[0]
    return output

def main(args):
    random.seed(42)
    torch.manual_seed(42)

    print(f"Loading data for {args.dataset}...")
    test_data = []
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
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"Loading model from {args.model_name_or_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, 
        torch_dtype=torch.float16, 
        device_map="auto",
        trust_remote_code=True
    )
    
    # Load prototypes
    print(f"Loading prototypes from {args.prototypes_path}...")
    prototypes = torch.load(args.prototypes_path)
    mu_T = prototypes["mu_T"].to(model.device)
    mu_H = prototypes["mu_H"].to(model.device)

    # Setup intervention
    layer_name = f"model.layers.{args.layer}"
    steering_stats = {'total': 0, 'steered': 0}
    hook_fn = get_spherical_intervention(
        mu_T, mu_H, 
        kappa=args.kappa, 
        alpha=args.alpha, 
        beta=args.beta, 
        stats=steering_stats
    )

    prefix = "Answer the following questions. You should think step-by-step and put your final answer within \\boxed{}.\n"
    prompts = []
    for example in test_data:
        prompt = prefix + "Question: " + example["problem"].strip() + "\nAnswer: <think>\n"
        prompts.append(prompt)

    results = []
    for i in trange(0, len(prompts), args.batch_size, desc="Generating"):
        batch_prompts = prompts[i : i + args.batch_size]
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(model.device)
        
        # We need to apply the hook during generation. 
        # Spherical Steering usually applies to each generated token.
        with torch.no_grad():
            with TraceDict(model, [layer_name], edit_output=hook_fn) as ret:
                output_ids = model.generate(
                    **inputs, 
                    max_new_tokens=args.max_tokens,
                    do_sample=False,
                    use_cache=True
                )
        
        prompt_len = inputs["input_ids"].shape[1]
        for j, out_id in enumerate(output_ids):
            gen_text = tokenizer.decode(out_id[prompt_len:], skip_special_tokens=True)
            results.append(trim_output(gen_text))

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
    print(f"Steering stats: {steering_stats}")
    
    # Run evaluation
    eval_main(save_path, save=True, output_dir=args.save_dir, dataset=args.dataset)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--prototypes_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="AIME")
    parser.add_argument("--aime_file", type=str, default="aime2024.jsonl")
    parser.add_argument("--save_dir", type=str, default="results/spherical")
    parser.add_argument("--layer", type=int, default=24)
    parser.add_argument("--kappa", type=float, default=20.0)
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--beta", type=float, default=-0.05)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--start", type=int, default=0)
    args = parser.parse_args()
    
    main(args)
