"""
CRUXEval Dataset Loader & Evaluator
====================================
Loads CRUXEval problems from JSONL files and provides standard evaluation utilities.
Supports both output prediction (CRUXEval-O) and input prediction (CRUXEval-I).
"""

import json
import os
import re
import ast
from typing import Optional, List, Dict


# ======================== Dataset Loading ========================

def load_cruxeval_dataset(path: str) -> List[Dict]:
    """
    Load CRUXEval problems from a single .jsonl file.

    Expected keys:
        - "code": str (Python function definition)
        - "input": str (input representation)
        - "output": str (expected return value representation)
        - "id": str

    Returns:
        List of dicts with keys: id, code, input, output.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Dataset file not found: {path}")

    problems = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"⚠️  Skipping malformed line {line_num} in {path}: {e}")
                continue

            problems.append({
                "id": item.get("id", f"cruxeval_{line_num}"),
                "code": item["code"],
                "input": item["input"],
                "output": item["output"],
            })

    if not problems:
        raise ValueError(f"No problems loaded from {path}")

    return problems


# ======================== Prompt Construction ========================

def build_cruxeval_output_prompt(code: str, input_val: str) -> list[dict]:
    """
    Construct a chat prompt for output prediction (CRUXEval-O).
    """
    return [
        {"role": "system", "content": (
            "You are given a Python function and an input to the function. "
            "Your task is to execute the function on the input and determine the final output.\n"
            "Put your final output value (as a Python literal, e.g., a number, string, list, dictionary, etc.) within \\boxed{}."
        )},
        {"role": "user", "content": f"Python Code:\n```python\n{code}\n```\n\nFunction Call:\nf({input_val})"}
    ]


def build_cruxeval_input_prompt(code: str, output_val: str) -> list[dict]:
    """
    Construct a chat prompt for input prediction (CRUXEval-I).
    """
    return [
        {"role": "system", "content": (
            "You are given a Python function f and its expected output. "
            "Your task is to find a Python input value (such as a string, list, tuple, dictionary, or function arguments) "
            "such that executing f on this input yields the expected output.\n"
            "Put your final input value (as a Python literal/tuple of arguments) within \\boxed{}."
        )},
        {"role": "user", "content": f"Python Code:\n```python\n{code}\n```\n\nExpected Output:\n{output_val}"}
    ]


# ======================== Answer Extraction ========================

def _extract_balanced_braces(text: str, start: int) -> Optional[str]:
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
        i += 1
    if depth == 0:
        return text[start:i - 1].strip()
    return None


def clean_cruxeval_prediction(pred: str, mode: str) -> str:
    """
    Clean up prediction text by removing markdown code blocks, assert prefix/suffix, etc.
    """
    if pred is None:
        return ""
    pred = pred.strip()
    # Remove markdown code block formatting
    if pred.startswith("```"):
        parts = pred.split("```")
        if len(parts) >= 3:
            pred = parts[1]
            if pred.startswith("python"):
                pred = pred[6:]
            elif pred.startswith("py"):
                pred = pred[2:]
    pred = pred.strip()

    # Recursively strip prefixes/suffixes
    changed = True
    while changed:
        changed = False
        pred_lower = pred.lower()
        
        # Suffixes to strip
        for suffix in [".", ":", "`", "*"]:
            if pred.endswith(suffix):
                pred = pred[:-len(suffix)].strip()
                changed = True
                pred_lower = pred.lower()
                
        # Prefixes to strip
        prefixes = [
            "the final output is the string",
            "the final output is string",
            "the final output is a dictionary",
            "the final output is dictionary",
            "the final output is a list",
            "the final output is list",
            "the final output is a tuple",
            "the final output is tuple",
            "the final output is",
            "the output is",
            "the answer is",
            "final output:",
            "final output",
            "output:",
            "output",
            "answer is",
            "**answer:**",
            "**answer**",
            "answer:",
            "answer",
            "boxed:",
            "boxed",
            "so, the answer is",
            "so the answer is",
            "so,"
        ]
        for prefix in prefixes:
            if pred_lower.startswith(prefix):
                pred = pred[len(prefix):].strip()
                changed = True
                pred_lower = pred.lower()
                break # break to restart loop with new length

    # Strip outer quotes if any
    if (pred.startswith("'") and pred.endswith("'")) or (pred.startswith('"') and pred.endswith('"')):
        pred = pred[1:-1].strip()

    # If the output is a full assertion, extract the right side
    # e.g., "assert f(x) == y"
    if "==" in pred:
        if mode == "output":
            pred = pred.split("==")[1].strip()
        else: # input
            # assert f(x) == y -> extract the argument string from f(argument)
            lhs = pred.split("==")[0].strip()
            if "assert f(" in lhs:
                inner = lhs.split("assert f(")[1]
                if inner.endswith(")"):
                    pred = inner[:-1].strip()
                else:
                    pred = inner.strip()
            elif "f(" in lhs:
                inner = lhs.split("f(")[1]
                if inner.endswith(")"):
                    pred = inner[:-1].strip()
                else:
                    pred = inner.strip()
    
    # Strip again after split/clean
    changed = True
    while changed:
        changed = False
        pred_lower = pred.lower()
        for suffix in [".", ":", "`", "*"]:
            if pred.endswith(suffix):
                pred = pred[:-len(suffix)].strip()
                changed = True
                pred_lower = pred.lower()
        for prefix in prefixes:
            if pred_lower.startswith(prefix):
                pred = pred[len(prefix):].strip()
                changed = True
                pred_lower = pred.lower()
                break

    # Strip outer quotes again
    if (pred.startswith("'") and pred.endswith("'")) or (pred.startswith('"') and pred.endswith('"')):
        pred = pred[1:-1].strip()

    # Clean LaTeX escapes like \{ and \} and \" and \'
    pred = pred.replace("\\{", "{").replace("\\}", "}").replace("\\'", "'").replace('\\"', '"')

    # Strip any ending comments like "# done"
    if "#" in pred:
        pred = pred.split("#")[0].strip()

    return pred


def extract_answer_cruxeval(text: str) -> Optional[str]:
    """
    Extract answer from model output.
    Priority:
      1. Final \\boxed{} after </think>
      2. Any \\boxed{} in the text
      3. Tag [ANSWER]...[/ANSWER] or <answer>...</answer>
      4. Fallback to last line of thinking-ending text (ignoring markdown blocks)
    """
    if not text:
        return None

    # Check after </think> tag
    THINK_END = "</think>"
    idx = text.rfind(THINK_END)
    search_text = text[idx + len(THINK_END):] if idx != -1 else text

    # Strategy 1: \\boxed{...} (match the LAST occurrence, allow optional backslash)
    boxed_positions = [m.end() for m in re.finditer(r"\\?boxed\s*\{", search_text)]
    if boxed_positions:
        content = _extract_balanced_braces(search_text, boxed_positions[-1])
        if content is not None:
            return content.strip()

    # Fallback checking full text for boxed
    if idx != -1:
        boxed_positions = [m.end() for m in re.finditer(r"\\?boxed\s*\{", text)]
        if boxed_positions:
            content = _extract_balanced_braces(text, boxed_positions[-1])
            if content is not None:
                return content.strip()

    # Strategy 2: [ANSWER]...[/ANSWER] or <answer>...</answer>
    for start_tag, end_tag in [("[ANSWER]", "[/ANSWER]"), ("<answer>", "</answer>")]:
        if end_tag in search_text:
            parts = search_text.split(end_tag)
            if len(parts) > 0:
                sub_part = parts[0]
                if start_tag in sub_part:
                    return sub_part.split(start_tag)[-1].strip()

    # Strategy 3: Fallback to last non-empty line (skipping markdown code blocks)
    lines = [line.strip() for line in search_text.split("\n")]
    filtered_lines = []
    for line in lines:
        l_stripped = line.strip()
        if not l_stripped:
            continue
        if l_stripped.startswith("```"):
            continue
        filtered_lines.append(l_stripped)

    if filtered_lines:
        last_line = filtered_lines[-1]

        # Handle comment on the last line
        if "#" in last_line:
            code_part, comment_part = last_line.split("#", 1)
            code_part = code_part.strip()
            comment_part = comment_part.strip()
            
            comment_lower = comment_part.lower()
            comment_val = None
            for indicator in ["output:", "output is", "returns", "->", "=="]:
                if indicator in comment_lower:
                    idx_ind = comment_lower.find(indicator)
                    comment_val = comment_part[idx_ind + len(indicator):].strip()
                    break
            
            if comment_val:
                last_line = comment_val
            else:
                try:
                    ast.literal_eval(comment_part)
                    last_line = comment_part
                except Exception:
                    last_line = code_part

        prefixes_to_strip = [
            "the final output is the string",
            "the final output is string",
            "the final output is a dictionary",
            "the final output is dictionary",
            "the final output is a list",
            "the final output is list",
            "the final output is a tuple",
            "the final output is tuple",
            "the final output is",
            "the output is",
            "the answer is",
            "final output:",
            "final output",
            "output:",
            "output",
            "answer is",
            "**answer:**",
            "**answer**",
            "answer:",
            "answer",
            "boxed:",
            "boxed"
        ]
        last_line_lower = last_line.lower()
        for prefix in prefixes_to_strip:
            if last_line_lower.startswith(prefix):
                last_line = last_line[len(prefix):].strip()
                last_line_lower = last_line.lower()
        return last_line

    return None


# ======================== Answer Checking ========================

def check_answer_cruxeval_output(code: str, input_val: str, gold_output: str, predicted: Optional[str]) -> bool:
    """
    Verify correctness of output prediction by local execution of assertion.
    """
    if predicted is None:
        return False

    pred_cleaned = clean_cruxeval_prediction(predicted, "output")

    # Try 1: direct exec using prediction
    # Construct:
    # {code}
    # assert f({input_val}) == {pred_cleaned}
    code_to_run = f"{code}\nassert f({input_val}) == {pred_cleaned}"
    try:
        exec_globals = {}
        exec(code_to_run, exec_globals)
        return True
    except Exception:
        pass

    # Try 2: Semantic literal comparison (eval function locally, compare python values)
    try:
        exec_globals = {}
        exec(code, exec_globals)
        func = exec_globals['f']
        gold_input_args = ast.literal_eval(f"({input_val},)")
        gold_output_obj = func(*gold_input_args)
        pred_obj = ast.literal_eval(pred_cleaned)
        if gold_output_obj == pred_obj:
            return True
    except Exception:
        pass

    # Try 3: If gold_output was a string and model output missing quotes, try quoting it
    try:
        gold_obj = ast.literal_eval(gold_output)
        if isinstance(gold_obj, str):
            escaped_pred = repr(pred_cleaned)
            code_to_run_2 = f"{code}\nassert f({input_val}) == {escaped_pred}"
            exec_globals = {}
            exec(code_to_run_2, exec_globals)
            return True
    except Exception:
        pass

    return False


def check_answer_cruxeval_input(code: str, gold_output: str, predicted: Optional[str]) -> bool:
    """
    Verify correctness of input prediction by local execution of assertion f(pred_cleaned) == gold_output.
    """
    if predicted is None:
        return False

    pred_cleaned = clean_cruxeval_prediction(predicted, "input")

    # Try 1: direct execution
    # Construct:
    # {code}
    # assert f({pred_cleaned}) == {gold_output}
    code_to_run = f"{code}\nassert f({pred_cleaned}) == {gold_output}"
    try:
        exec_globals = {}
        exec(code_to_run, exec_globals)
        return True
    except Exception:
        pass

    # Try 2: Semantic run prediction on f() and compare output
    try:
        exec_globals = {}
        exec(code, exec_globals)
        func = exec_globals['f']
        gold_output_obj = ast.literal_eval(gold_output)
        pred_args = ast.literal_eval(f"({pred_cleaned},)")
        res = func(*pred_args)
        if res == gold_output_obj:
            return True
    except Exception:
        pass

    # Try 3: Try quoting prediction if it fails (e.g. missing quotes for a string arg)
    try:
        escaped_pred = repr(pred_cleaned)
        code_to_run_2 = f"{code}\nassert f({escaped_pred}) == {gold_output}"
        exec_globals = {}
        exec(code_to_run_2, exec_globals)
        return True
    except Exception:
        pass

    return False
