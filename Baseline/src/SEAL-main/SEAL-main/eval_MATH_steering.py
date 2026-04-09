import argparse
import os
import re
import json
import random
import torch
import evaluate
from transformers import  AutoTokenizer
from modeling_utils.modeling_qwen2 import Qwen2ForCausalLM
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from collections import Counter
from datasets import load_dataset
from peft import PeftModel, PeftConfig

import sys
import os
import gc
from tqdm import trange

from get_math_results import main as eval_main
os.environ["TOKENIZERS_PARALLELISM"] = "false"

exact_match = evaluate.load("exact_match")


def trim_output(output):
    instruction_prefix = "Answer the following question"
    question_prefix = 'Question:'
    comment_prefix = 'Comment:'  # for some reason, Llama 13B likes to generate these comments indefinitely

    for prefix in [instruction_prefix, question_prefix, comment_prefix]:
        if prefix in output:
            output = output.split(prefix)[0]

    return output


def extract_box(pred_str):
    ans = pred_str.split("boxed")[-1]
    if len(ans) == 0:
        return ""
    elif ans[0] == "{":
        stack = 1
        a = ""
        for c in ans[1:]:
            if c == "{":
                stack += 1
                a += c
            elif c == "}":
                stack -= 1
                if stack == 0:
                    break
                a += c
            else:
                a += c
    else:
        a = ans.split("$")[0].strip()

    return a


def extract_last_number(pred_str):
    o = re.sub(r"(\d),(\d)", r"\1\2", pred_str)
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", o)
    if numbers:
        ans = numbers[-1]
    else:
        ans = None
    return ans


import sys
import os

# Add the project root to sys.path to import from src
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

from util.loaders.aime_loader import load_aime_dataset, build_aime_prompt
from util.loaders.math500_loader import load_math500_dataset, build_math500_prompt

def main(args):
    random.seed(42)

    print("Loading data...")
    test_data = []
    if args.dataset == "MATH500":
        # Use local loader for MATH500
        data_path = os.path.join(project_root, "util/dataset/MATH500.jsonl")
        data = load_math500_dataset(data_path)
        for example in data:
            test_data.append({
                "question": example["problem"],
                "answer": example["answer"],
                "gt": example["answer"],
            })
    elif args.dataset == "AIME":
        # Support AIME dataset
        # Default to aime2024.jsonl if not specified, or we could support multiple
        data_path = os.path.join(project_root, "util/dataset/aime2024.jsonl")
        data = load_aime_dataset(data_path)
        for example in data:
            test_data.append({
                "question": example["problem"],
                "answer": example["answer"],
                "gt": example["answer"],
            })
    elif args.dataset == "GSM":
        data_path = "data/gsm/test.jsonl"
        with open(data_path) as fin:
            for line in fin:
                example = json.loads(line)
                answer = example["answer"].split("####")[1].strip()
                answer =  re.sub(r"(\d),(\d)", r"\1\2", answer)
                test_data.append({
                    "question": example["question"],
                    "answer":example["answer"].split("####")[0].strip(),
                    "gt": answer
                })
    else:
        raise ValueError(f"Dataset {args.dataset} not supported")
    if args.start:
        test_data = test_data[args.start:]
    if args.max_examples and len(test_data) > args.max_examples:
        test_data = test_data[:args.max_examples]

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path if args.tokenizer_name_or_path else args.model_name_or_path)

     # set padding side to left for batch generation
    tokenizer.padding_side = "left"

    # set pad token to eos token if pad token is not set (as is the case for llama models)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    prefix="Answer the following questions. You should think step-by-step and put your final answer within \\boxed{}.\n"
    prompts = []
    for i, example in enumerate(test_data):
        prompt =  prefix+"Question: " + example["question"].strip()+"\nAnswer: "
        if args.use_chat_format:
            if  "deepseek" in args.model_name_or_path:
                messages = [{"role": "user", "content": prefix + "Question: " + example["question"].strip()}]
            else:
                messages = [{"role": "system", "content": prefix}, {"role": "user", "content": "Question: " + example["question"].strip()}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            if args.remove_bos and tokenizer.bos_token is not None and prompt.startswith(tokenizer.bos_token):
                prompt = prompt[len(tokenizer.bos_token):]
        prompts.append(prompt)
    with open(os.path.join(args.save_dir, "example_prompt.txt"), 'w') as fout:
        fout.write(prompts[0])


    if "qwen" in args.model_name_or_path.lower():
        model = Qwen2ForCausalLM.from_pretrained(args.model_name_or_path, device_map="auto")
    else:
        raise ValueError("Model not supported")
    
    if args.steering:
        steer_vec = torch.load(args.steering_vector, weights_only=True)
        steer_vec = steer_vec.to(model.device)
        model.set_steering_flag(steering_flag=True, steering_layer=args.steering_layer, steer_vec=steer_vec,  steer_coef=args.steering_coef, tokenizer=tokenizer)

    outputs = []
    for i in trange(0, len(prompts), args.batch_size):
        if args.steering:
            model.start_new_round()
        batch = prompts[i:i+args.batch_size]
        tokenized_batch = tokenizer(batch, return_tensors="pt", padding=True)
        tokenized_batch = {k: v.to(model.device) for k, v in tokenized_batch.items()}
        with torch.no_grad():
            output = model.generate(**tokenized_batch, do_sample=False, max_new_tokens=args.max_tokens,use_cache=True)
        prompt_len = tokenized_batch["input_ids"].shape[1]
        output = [tokenizer.decode(o[prompt_len:], skip_special_tokens=True) for o in output]
        outputs.extend(output)

    outputs = [[trim_output(o)] for o in outputs]


    predictions = [{
        "prompt": prompt,
        "problem": example["question"],
        "answer": example["gt"],
        "solution":  example["answer"],
        "model_generation": output,
    } for example, output, prompt in zip(test_data, outputs, prompts)]

    with open(os.path.join(args.save_dir, "predictions.jsonl"), "w") as fout:
        for prediction in predictions:
            fout.write(json.dumps(prediction) + "\n")
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--start",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="results/gsm"
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--tokenizer_name_or_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--use_chat_format",
        action="store_true",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="MATH",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--remove_bos",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--steering",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--steering_vector",
        type=str,
        default=None
    )
    parser.add_argument(
        "--steering_layer",
        type=int,
        default=-1
    )
    parser.add_argument(
        "--steering_coef",
        type=float,
        default=0.0
    )


    args = parser.parse_args()

    if args.steering:
        vector_name_split = args.steering_vector.split("/")[-3:]
        vector_name_split[-1] = vector_name_split[-1].split(".")[0]
        name = "_".join(vector_name_split)
        args.save_dir = os.path.join(args.save_dir, name, f"coef_{args.steering_coef}")
    else:
        args.save_dir = os.path.join(args.save_dir, "base")
    
    if args.remove_bos:
        args.save_dir = args.save_dir + "_remove_bos"

    if args.max_examples or args.start:
        start = 0 if args.start is None else args.start
        end = start + args.max_examples if args.max_examples is not None else -1
        args.save_dir = os.path.join(args.save_dir, f"{start}_{end}")
        
    print(args.save_dir)
    main(args)
    eval_main(os.path.join(args.save_dir, "predictions.jsonl"), save=True, k=None, output_dir=args.save_dir, dataset=args.dataset)


        
