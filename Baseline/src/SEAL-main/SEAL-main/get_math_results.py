import os
import json
from tqdm import tqdm, trange
from eval_math_rule.evaluation.grader import math_equal
from eval_math_rule.evaluation.parser import extract_answer, parse_ground_truth, strip_string
from collections import Counter

import multiprocessing
import queue

def math_equal_with_timeout(pred, gt_ans, timeout):
    def target(result_queue):
        try:
            result_queue.put(math_equal(pred, gt_ans))
        except Exception as e:
            result_queue.put(e)


    result_queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=target, args=(result_queue,))
    process.start()

    process.join(timeout)

    if process.is_alive():
        print(f"Timeout occurred for prediction: {pred}")
        process.terminate()
        process.join()    
        return False

   
    try:
        result = result_queue.get_nowait()
    except queue.Empty:
        print("Result queue timed out")
        return False

    if isinstance(result, Exception):
        print(f"Error occurred: {result}")
        return False

    return result



def parallel_math_equal(all_pred, gt_ans, timeout=20):
    results = []
    for pred in all_pred:
        results.append(math_equal_with_timeout(pred, gt_ans, timeout))
    return results



def main(res_path, save=False, k=None, output_dir=None, dataset=None):
    # args = parse_args()
    with open(res_path, "r") as f:
        lines = f.readlines()
        data = [json.loads(line) for line in lines]
    
    for example in tqdm(data):
        # gt_cot, gt = parse_ground_truth(example, data_name="omni-math")
        if "model_generation" not in example:
            example["model_generation"] = example["model_output"]
        if k is not None:
            example["model_generation"] = example["model_generation"][:k]
        
        gt_ans = example["answer"]
        
        if dataset == "MATH500":
            from util.loaders.math500_loader import extract_answer_math500, check_answer_math500
            all_pred = [extract_answer_math500(p[0] if isinstance(p, list) else p) for p in example["model_generation"]]
            all_eval = [check_answer_math500(p, gt_ans) for p in all_pred]
        elif dataset == "AIME":
            from util.loaders.aime_loader import extract_answer, check_answer
            all_pred = [extract_answer(p[0] if isinstance(p, list) else p) for p in example["model_generation"]]
            all_eval = [check_answer(p, int(gt_ans)) for p in all_pred]
        else:
            gt_ans_stripped = strip_string(gt_ans, skip_unit=False)
            all_pred = [extract_answer(p[0] if isinstance(p, list) else p, data_name="omni-math") for p in example["model_generation"]]
            all_pred = [strip_string(p, skip_unit=False) for p in all_pred]
            all_eval = parallel_math_equal(all_pred, gt_ans_stripped, timeout=5)
        effective_pred = [p for p, o in zip(all_pred, example["model_generation"]) if "boxed" in o]
        if len(effective_pred) == 0:
            effective_pred = all_pred
        counter = Counter(effective_pred)
        pred = counter.most_common(1)[0][0]
        index = all_pred.index(pred)
        eval = all_eval[index]
        example["all_pred"] = all_pred
        example["all_eval"] = all_eval
        example["mv_pred"] = pred
        example["mv_eval"] = eval
        example["mv_index"] = index

    acc = sum([example["mv_eval"] for example in data]) / len(data)
    print(f"Accuracy: {acc:.3f}")
    
    if save:
        out_file = os.path.join(output_dir, "math_eval.jsonl")
        with open(out_file, "w") as f:
            for example in data:
                f.write(json.dumps(example) + "\n")
        
        metric_file= os.path.join(output_dir, "metrics.json")
        with open(metric_file, "w") as f:
            json.dump({"acc": acc}, f)

    