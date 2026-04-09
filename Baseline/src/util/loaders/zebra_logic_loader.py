"""
ZebraLogic Dataset Loader & Evaluator
====================================
Loads ZebraLogic puzzles from JSON files and provides
robust evaluation for logic puzzle answers.
"""

import json
import os
import re
from typing import Optional, List, Dict


# ======================== Dataset Loading ========================

def load_zebra_dataset(path: str) -> List[Dict]:
    """
    Load ZebraLogic problems from a .json file.

    Expected keys:
        - "id": str
        - "puzzle": str
        - "question": str
        - "answer": str

    Returns:
        List of dicts with keys: id, puzzle, question, answer.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Dataset file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected a list of puzzles in {path}")

    problems = []
    for item in data:
        problems.append({
            "id": item["id"],
            "puzzle": item["puzzle"],
            "question": item["question"],
            "answer": item["answer"],
        })

    return problems


# ======================== Prompt Construction ========================

def build_zebra_prompt(puzzle: str, question: str) -> list[dict]:
    """
    Construct a chat prompt for ZebraLogic puzzle solving.
    """
    return [
        {"role": "system", "content": (
            "You are a logic puzzle expert. Solve the following puzzle "
            "step by step with rigorous reasoning.\n"
            "Put your final answer (usually a name or value) within \\boxed{}."
        )},
        {"role": "user", "content": f"Puzzle:\n{puzzle}\n\nQuestion: {question}"}
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

def extract_answer_zebra(text: str) -> Optional[str]:
    """
    Robustly extract the logic puzzle answer from model output.
    Priority:
      1. Final \boxed{} after </think>
      2. Any \boxed{} in the text
      3. Last short string or specific markers
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
            # Clean up LaTeX artifacts if any
            return content.strip().strip("$")

    # Fallback checking 
    if idx != -1:
        boxed_positions = [m.end() for m in re.finditer(r"\\boxed\s*\{", text)]
        if boxed_positions:
            content = _extract_balanced_braces(text, boxed_positions[-1])
            if content:
                return content.strip().strip("$")

    return None


# ======================== Answer Checking ========================

def check_answer_zebra(predicted: Optional[str], expected: str) -> bool:
    """
    Compare predicted logic answer with expected ground truth.
    Case-insensitive and whitespace-insensitive.
    """
    if predicted is None:
        return False
    
    p_norm = predicted.strip().lower()
    e_norm = expected.strip().lower()
    
    # Also handle list-like answers or minor variations
    return p_norm == e_norm or e_norm in p_norm
