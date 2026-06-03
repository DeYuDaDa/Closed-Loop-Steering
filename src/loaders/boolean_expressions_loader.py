"""
Boolean Expressions Dataset Loader & Evaluator
====================================
Loads boolean expressions from JSONL files and provides standard evaluation utilities.
"""

import json
import os
import re
from typing import Optional, List, Dict


# ======================== Dataset Loading ========================

def load_boolean_expressions_dataset(path: str) -> List[Dict]:
    """
    Load boolean expression problems from a .jsonl file.

    Expected keys:
        - "input": str
        - "target": str ("True" or "False")

    Returns:
        List of dicts with keys: id, problem, answer.
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
                "id": item.get("id", f"boolean_expr_{line_num}"),
                "problem": item["input"],
                "answer": item["target"],
            })

    if not problems:
        raise ValueError(f"No problems loaded from {path}")

    return problems


# ======================== Prompt Construction ========================

def build_boolean_expressions_prompt(problem: str) -> list[dict]:
    """
    Construct a chat prompt for boolean expression evaluation.
    """
    return [
        {"role": "system", "content": (
            "Evaluate the following boolean expression and determine its final truth value (True or False).\n"
            "Put your final answer (either True or False) within \\boxed{}."
        )},
        {"role": "user", "content": problem}
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


def extract_answer_boolean_expressions(text: str) -> Optional[str]:
    """
    Extract the final boolean answer (True/False) from model output.
    Priority:
      1. Final \\boxed{} after </think>
      2. Any \\boxed{} in the text
      3. Last boolean keyword (True/False) in the text
    """
    if not text:
        return None

    # Try after </think> tag first
    THINK_END = "</think>"
    idx = text.rfind(THINK_END)
    if idx != -1:
        search_text = text[idx + len(THINK_END):]
    else:
        search_text = text

    # Find all \boxed{...}
    boxed_positions = [m.end() for m in re.finditer(r"\\boxed\s*\{", search_text)]
    if boxed_positions:
        content = _extract_balanced_braces(search_text, boxed_positions[-1])
        if content:
            ans = content.strip().strip("$").strip().lower()
            if "true" in ans:
                return "True"
            if "false" in ans:
                return "False"

    # Fallback checking full text for boxed
    if idx != -1:
        boxed_positions = [m.end() for m in re.finditer(r"\\boxed\s*\{", text)]
        if boxed_positions:
            content = _extract_balanced_braces(text, boxed_positions[-1])
            if content:
                ans = content.strip().strip("$").strip().lower()
                if "true" in ans:
                    return "True"
                if "false" in ans:
                    return "False"

    # Fallback 2: Find the last occurrence of True or False word in the thinking-ending text
    words = re.findall(r"\b(true|false)\b", search_text, re.IGNORECASE)
    if words:
        return "True" if words[-1].lower() == "true" else "False"

    # Fallback 3: Find the last occurrence of True or False word in the whole text
    words = re.findall(r"\b(true|false)\b", text, re.IGNORECASE)
    if words:
        return "True" if words[-1].lower() == "true" else "False"

    return None


# ======================== Answer Checking ========================

def check_answer_boolean_expressions(predicted: Optional[str], expected: str) -> bool:
    """
    Compare predicted answer (True/False) with expected answer.
    """
    if predicted is None:
        return False
    return predicted.strip().lower() == expected.strip().lower()
