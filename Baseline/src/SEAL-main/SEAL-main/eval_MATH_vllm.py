
import argparse
import os
import re
import json
import random
import torch
import evaluate
from transformers import AutoModelForCausalLM, AutoTokenizer, OPTForCausalLM, GPTNeoXForCausalLM
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from collections import Counter
from datasets import load_dataset
from functools import partial


import sys
import os
import gc

from get_math_results import main as eval_main
os.environ["TOKENIZERS_PARALLELISM"] = "false"

exact_match = evaluate.load("exact_match")


def logit_adjustment(token_ids, logits, adjust_ids, values, max_len=-1):
    if max_len <= 0 or len(token_ids) <= max_len:
        logits[adjust_ids.to(logits.device)] += values
    return logits



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





def main(args):
    random.seed(42)

    print("Loading data...")
    test_data = []
    if args.dataset == "MATH500":
        data = load_dataset("HuggingFaceH4/MATH-500", split="test")
        for example in data:
            gt = extract_box(example["solution"])
            test_data.append({
                "question": example["problem"],
                "answer": example["solution"],
                "gt":gt,
            })
    elif args.dataset == "MATH_train":
        data_path = "data/MATH/train.jsonl"
        with open(data_path) as fin:
            for line in fin:
                example = json.loads(line)
                gt = extract_box(example["solution"])
                test_data.append({
                    "question": example["problem"],
                    "answer": example["solution"],
                    "gt":gt,
                })
    elif args.dataset in ["GSM", "GSM_train"]:
        if args.dataset == "GSM_train":
            data_path = "data/gsm/train.jsonl"
        else:
            data_path = "data/gsm/test.jsonl"
        with open(data_path) as fin:
            for line in fin:
                example = json.loads(line)
                answer = example["answer"].split("####")[1].strip()
                answer =  re.sub(r"(\d),(\d)", r"\1\2", answer)
                test_data.append({
                    "question": example["question"],
                    "answer": example["answer"].split("####")[0].strip(),
                    "gt": answer
                })
    else:
        raise ValueError("Dataset not supported")
    if args.max_examples and len(test_data) > args.max_examples:
        test_data = test_data[:args.max_examples]

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path if args.tokenizer_name_or_path else args.model_name_or_path)

     # set padding side to left for batch generation
    tokenizer.padding_side = "left"

    # set pad token to eos token if pad token is not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    prefix="Answer the following questions. You should think step-by-step and put your final answer within \\boxed{}.\n"
    prompts = []
    for i, example in enumerate(test_data):
        prompt =  prefix+"Question: " + example["question"].strip()+"\nAnswer: "
        if args.use_chat_format:
            if "gemma" in args.model_name_or_path or "deepseek" in args.model_name_or_path:
                messages = [{"role": "user", "content": prefix + "Question: " + example["question"].strip()}]
            else:
                messages = [{"role": "system", "content": prefix}, {"role": "user", "content": "Question: " + example["question"].strip()}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            if args.remove_bos and tokenizer.bos_token is not None and prompt.startswith(tokenizer.bos_token):
                prompt = prompt[len(tokenizer.bos_token):]
        prompts.append(prompt)
    with open(os.path.join(args.save_dir, "example_prompt.txt"), 'w') as fout:
        fout.write(prompts[0])

    model = LLM(model=args.model_name_or_path, tokenizer=args.tokenizer_name_or_path if args.tokenizer_name_or_path else args.model_name_or_path, swap_space=16, gpu_memory_utilization=0.98, enable_lora=args.peft is not None, tensor_parallel_size=torch.cuda.device_count(), max_lora_rank=128, max_model_len=args.max_tokens+2000)

    if not args.logit_adjustment:

        sampling_params = SamplingParams(n=1,
                                        temperature=0,
                                        max_tokens=args.max_tokens)
    else:
        vocab = tokenizer.get_vocab()
        logit_adjustment_tokens = torch.LongTensor([vocab[token] for token in vocab.keys() if any([x in token for x in args.logit_adjustment_tokens])]).to("cuda")
        logit_adjustment_process = partial(logit_adjustment, adjust_ids=logit_adjustment_tokens, values=args.logit_adjustment_value, max_len=args.logit_adjustment_max_len)
        sampling_params = SamplingParams(n=1,
                                        temperature=0,
                                        max_tokens=args.max_tokens,
                                        logits_processors=[logit_adjustment_process]
                                        )
    if args.peft is not None:
        outputs = model.generate(prompts=prompts, sampling_params=sampling_params, lora_request=LoRARequest("lora_path", 1, lora_path=args.peft))
    else:
        outputs = model.generate(prompts=prompts, sampling_params=sampling_params)

    result = []
    for output in outputs:
        attempts = []
        for ith_output in output.outputs:
            attempts.append(ith_output.text)
        result.append(attempts)
    

    outputs = [[trim_output(o) for o in output] for output in result]

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
        "--start",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--max_examples",
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
        "--peft",
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
        "--remove_bos",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--logit_adjustment",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--logit_adjustment_tokens",
        type=str,
        nargs="*",
        default=[]
    )
    parser.add_argument(
        "--logit_adjustment_value",
        type=float,
        default=0.0
    )
    parser.add_argument(
        "--logit_adjustment_max_len",
        type=int,
        default=-1
    )


    args = parser.parse_args()

    if args.logit_adjustment:
        name = "_".join(args.logit_adjustment_tokens)+f"_value_{args.logit_adjustment_value}"
        if args.logit_adjustment_max_len>0:
            name += f"_first{args.logit_adjustment_max_len}"
        args.save_dir = os.path.join(args.save_dir, "logit-adjustment", name)

    main(args)
    eval_main(os.path.join(args.save_dir, "predictions.jsonl"), save=True, k=None, output_dir=args.save_dir)
    



        
