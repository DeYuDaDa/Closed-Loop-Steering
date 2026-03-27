"""
MATH500 Dataset Loader & Evaluator
====================================
Loads MATH500 problems from JSONL files and provides
robust evaluation utilities for mathematical LaTeX output.
"""

import json
import os
import re
from typing import Optional, List, Dict
import sys
import os

try:
    from util.fix_result_acc import check_equiv
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from util.fix_result_acc import check_equiv


# ======================== Dataset Loading ========================

def load_math500_dataset(path: str) -> List[Dict]:
    """
    Load MATH500 problems from a .jsonl file.

    Expected keys:
        - "problem": str
        - "answer": str (LaTeX format)
        - "unique_id": str

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
                "id": item.get("unique_id", f"math500_{line_num}"),
                "problem": item["problem"],
                "answer": item["answer"],
            })

    if not problems:
        raise ValueError(f"No problems loaded from {path}")

    return problems


# ======================== Prompt Construction ========================

def build_math500_prompt(problem: str) -> list[dict]:
    """
    Construct a chat prompt for MATH500 problem solving.
    """
    return [
        {"role": "system", "content": (
            "You are a helpful and harmless assistant. You are a tool that can "
            "help users solve problems and answer questions.\n"
            "Solve the following math problem step by step. "
            "Put your final answer within \\boxed{}."
        )},
        {"role": "user", "content": problem}
    ]


# ======================== Answer Extraction ========================

def _normalize_latex(latex: str) -> str:
    """
    Basic normalization for LaTeX math strings to improve matching robustness.
    """
    if not latex:
        return ""
    
    # Remove whitespace
    latex = re.sub(r'\s+', '', latex)
    
    # Remove LaTeX formatting that doesn't affect value
    latex = latex.replace(r'\text{', '{').replace(r'\mathrm{', '{')
    latex = latex.replace(r'\left', '').replace(r'\right', '')
    latex = latex.replace(r'$', '')
    
    # Standardize fractions: \frac12 -> \frac{1}{2} (simplified)
    # This is a basic version; complex ones might need more regex
    
    return latex.strip()

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

def extract_answer_math500(text: str) -> Optional[str]:
    """
    Robustly extract the LaTeX answer from model output.
    Priority:
      1. Final \boxed{} after </think>
      2. Any \boxed{} in the text
      3. Last line or specific markers
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
            return content

    # Fallback to searching the whole text if not found in the answer section
    if idx != -1:
        boxed_positions = [m.end() for m in re.finditer(r"\\boxed\s*\{", text)]
        if boxed_positions:
            content = _extract_balanced_braces(text, boxed_positions[-1])
            if content:
                return content

    return None


# ======================== Answer Checking ========================

def check_answer_math500(predicted: Optional[str], expected: str) -> bool:
    """
    Compare predicted LaTeX with expected ground truth.
    Uses robust evaluation from fix_result_acc.
    """
    if predicted is None:
        return False
    
    p_norm = _normalize_latex(predicted)
    e_norm = _normalize_latex(expected)
    
    if p_norm == e_norm:
        return True
        
    try:
        return check_equiv(expected, predicted)
    except Exception:
        pass
        
    return False
