import json
import os 
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import argparse


def generate_math_data(data_dir, data_path):
    correct, incorrect = [], []
    with open(data_path) as f:
        data = f.readlines()
        data = [json.loads(line) for line in data]
    with open(f"{data_dir}/math_eval.jsonl") as f:
        eval = f.readlines()
        eval = [json.loads(line) for line in eval]
    
    data = data[:len(eval)]
    for d, e in zip(data, eval):
        local_correct, local_incorrect = [], []
        prompt = e["prompt"]
        assert d["problem"] == e["problem"]
        for o, c in zip(e["model_generation"], e["all_eval"]):
            if c:
                local_correct.append({"prompt":prompt, "response":o, "level":d["level"], "gt":e["answer"]})
            else:
                local_incorrect.append({"prompt":prompt, "response":o, "level":d["level"], "gt":e["answer"]})
        correct.extend(local_correct)
        incorrect.extend(local_incorrect)
    return correct, incorrect
    




def generate_index(text, tokenizer, split_id, think_only=True):

    check_words=["verify", "make sure", "hold on", "think again", "'s correct", "'s incorrect", "Let me check", "seems right"]
    check_prefix = ["Wait"]
    swicth_words = ["think differenly", "another way", "another approach", "another method", "another solution", "another strategy", "another technique"]
    switch_prefix = ["Alternatively"]
    
    tokens = tokenizer.encode(text)
    if think_only:
        think_begin_id = tokenizer.encode("<think>", add_special_tokens=False)[0]
        think_end_id = tokenizer.encode("</think>", add_special_tokens=False)[0]
        if think_begin_id not in tokens:
            return [], [], []
    
        start = tokens.index(think_begin_id)+1
        if think_end_id not in tokens[start:]:
            end=len(tokens)
        else:
            end = tokens.index(think_end_id, start)
        think_tokens = tokens[start:end]
    else:
        think_tokens = tokens
        start = 0

    index = [i for i, t in enumerate(think_tokens) if t in split_id] + [len(think_tokens)]
    step_index = []
    check_index=[]
    switch_index=[]

    for i in range(len(index)-1):
        step_index.append(index[i]+start)
        step = think_tokens[index[i]+1:index[i+1]]
        step = tokenizer.decode(step).strip(" ").strip("\n")
        if any([step.lower().startswith(p.lower()) for p in check_prefix]) or any([w.lower() in step.lower() for w in check_words]):
                check_index.append(i)
        elif any([step.lower().startswith(p.lower()) for p in switch_prefix]) or any([w.lower() in step.lower() for w in swicth_words]):
            switch_index.append(i)
    return step_index, check_index, switch_index

def generate(model_path, data, save_dir):
    think_only = "deepseek" in model_path.lower()
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = "left"
    # set pad token to eos token if pad token is not set (as is the case for llama models)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    vocab = tokenizer.get_vocab()
    split_id = [vocab[token] for token in vocab.keys() if "ĊĊ" in token]

    prompts = [d["prompt"]+d["response"] for d in data]

    layer_num = model.config.num_hidden_layers+1
    hidden_dict=[{} for _ in range(layer_num)]

    for k, p in tqdm(enumerate(prompts), total=len(prompts)):
        tokenized_batch = tokenizer([p], return_tensors="pt", padding=True)
        tokenized_batch = {k: v.to(model.device) for k, v in tokenized_batch.items()}
        with torch.no_grad():
            output = model(**tokenized_batch, output_hidden_states=True)
            hidden_states = output.hidden_states
            hidden_states = [h.detach().cpu() for h in hidden_states]
        layer_num = len(hidden_states)
        step_index, check_index, switch_index = generate_index(p, tokenizer, split_id, think_only=think_only)
        step_index = torch.LongTensor(step_index)
        check_index = torch.LongTensor(check_index)
        switch_index = torch.LongTensor(switch_index)
        for i in range(layer_num):
            h = hidden_states[i][0]
            step_h = h[step_index]
            hidden_dict[i][k] = {"step":step_h, "check_index": check_index, "switch_index": switch_index}
        del hidden_states
    os.makedirs(save_dir, exist_ok=True)
    torch.save(hidden_dict, f"{save_dir}/hidden.pt")
    json.dump(prompts, open(f"{save_dir}/prompts.json", "w"))






if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--type", type=str, default="correct", choices=["correct", "incorrect"])
    parser.add_argument("--start", type=int, default=-1)
    parser.add_argument("--sample", type=int, default=-1)
    args = parser.parse_args()
    correct, incorrect = generate_math_data(data_dir=args.data_dir, data_path=args.data_path)
    if args.type == "correct":
        data = correct
    else:
        data = incorrect
    save_dir = f"{args.data_dir}/hidden_{args.type}"
    if args.start != -1:
        data = data[args.start:]
        if args.sample != -1:
            data = data[:args.sample]
            save_dir = f"{save_dir}_{args.start}_{args.start+args.sample}"
        else:
            save_dir = f"{save_dir}_{args.start}_-1"
    print(save_dir)
    generate(args.model_path, data, save_dir)