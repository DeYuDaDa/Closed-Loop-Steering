import argparse
import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import trange
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "src/SEAL-main/SEAL-main"))

from util.loaders.aime_loader import load_aime_dataset
from util.loaders.math500_loader import load_math500_dataset
from get_math_results import main as eval_main

def trim_output(output):
    for prefix in ["Answer the following question", "Question:", "Comment:"]:
        if prefix in output:
            output = output.split(prefix)[0]
    return output

def main(args):
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
        test_data = test_data[args.max_examples]

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model from {args.model_name_or_path}...")
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()

    print(f"Loading steering vector from {args.vector_path}...")
    v_md = torch.load(args.vector_path).to(model.device).to(torch.bfloat16)
    
    layer_idx = args.layer
    multiplier = args.multiplier

    def caa_hook(module, args_in, output):
        # output is (hidden_states, ...)
        h = output[0] if isinstance(output, tuple) else output
        # Multiply and add
        # guide says only after prompt, but in generate() it's handled by position_ids or just applying to the last token during decoding.
        # During prefill, h has shape [batch, seq_len, dim]. During decoding, [batch, 1, dim].
        h[:, -1, :] += multiplier * v_md
        return output

    handle = model.model.layers[layer_idx].register_forward_hook(caa_hook)

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
            output = model.generate(**inputs, max_new_tokens=args.max_tokens, do_sample=False, use_cache=True)
        
        prompt_len = inputs["input_ids"].shape[1]
        for out in output:
            gen_text = tokenizer.decode(out[prompt_len:], skip_special_tokens=True)
            results.append(trim_output(gen_text))

    handle.remove()

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
    
    eval_main(save_path, save=True, output_dir=args.save_dir, dataset=args.dataset)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--vector_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="AIME")
    parser.add_argument("--aime_file", type=str, default="aime2024.jsonl")
    parser.add_argument("--save_dir", type=str, default="results/caa")
    parser.add_argument("--layer", type=int, default=24)
    parser.add_argument("--multiplier", type=float, default=1.0)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--start", type=int, default=0)
    args = parser.parse_args()
    main(args)
