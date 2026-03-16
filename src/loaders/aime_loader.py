"""
AIME Benchmark Loader & Evaluator
====================================
Loads AIME competition problems from JSONL files and provides
standard evaluation utilities:

  - load_aime_dataset(): Load problems from .jsonl files
  - build_aime_prompt(): Construct math-oriented chat prompt
  - extract_answer(): Multi-strategy integer extraction (boxed → tag → last int)
  - check_answer(): Pass@1 exact-match integer comparison
"""

import json
import os
import re
from typing import Optional


# ======================== Dataset Loading ========================

def load_aime_dataset(path: str) -> list[dict]:
    """
    Load AIME problems from a single .jsonl file.

    Each line should be a JSON object with keys:
        - "problem": str (problem statement, may contain LaTeX)
        - "answer": int  (ground truth integer, 0-999)
        - "id": int | str

    Args:
        path: Path to a .jsonl file.

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
                "id": item.get("id", line_num - 1),
                "problem": item["problem"],
                "answer": int(item["answer"]),
            })

    if not problems:
        raise ValueError(f"No problems loaded from {path}")

    return problems


def list_aime_datasets(dataset_dir: str) -> list[str]:
    """
    List all .jsonl files in the dataset directory.

    Returns:
        Sorted list of absolute paths to .jsonl files.
    """
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    files = sorted([
        os.path.join(dataset_dir, f)
        for f in os.listdir(dataset_dir)
        if f.endswith(".jsonl")
    ])
    return files


# ======================== Prompt Construction ========================

def build_aime_prompt(problem: str) -> list[dict]:
    """
    Construct a chat prompt for AIME problem solving.

    Uses a structured chat format:
    - System message instructs step-by-step reasoning
    - Requires final answer in \\boxed{} format

    Args:
        problem: The AIME problem statement (may contain LaTeX).

    Returns:
        List of message dicts.
    """
    return [
        {"role": "system", "content": (
            "You are a math competition expert. Solve the following problem "
            "step by step with rigorous mathematical reasoning.\n"
            "self-verification.\n"
            "Put your final integer answer (0-999) within \\boxed{}."
        )},
        {"role": "user", "content": problem}
    ]


# ======================== Answer Extraction ========================

def _extract_balanced_braces(text: str, start: int) -> Optional[str]:
    """
    Extract content within balanced braces starting at position `start`.

    `start` should point to the character AFTER the opening '{'.
    Returns the content inside the braces, or None if unbalanced.
    """
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

def _extract_from_section(text: str) -> Optional[int]:
    """
    Core extraction logic: try boxed, then answer tag, then last int.
    """
    if not text or not text.strip():
        return None

    # Strategy 1: \\boxed{...}  (match the LAST occurrence)
    boxed_positions = [m.end() for m in re.finditer(r"\\boxed\s*\{", text)]
    if boxed_positions:
        content = _extract_balanced_braces(text, boxed_positions[-1])
        if content is not None:
            integer = _parse_integer(content)
            if integer is not None:
                return integer

    # Strategy 2: <answer>...</answer> tag
    answer_tag_matches = re.findall(
        r"<answer>\s*(.*?)\s*</answer>", text, re.IGNORECASE
    )
    if answer_tag_matches:
        content = answer_tag_matches[-1].strip()
        integer = _parse_integer(content)
        if integer is not None:
            return integer

    # Strategy 3: Last standalone integer in the text
    # Only consider integers in valid AIME range [0, 999]
    all_ints = re.findall(r"\b(\d{1,3})\b", text)
    if all_ints:
        for candidate in reversed(all_ints):
            val = int(candidate)
            if 0 <= val <= 999:
                return val

    return None


def extract_answer(text: str) -> Optional[int]:
    """
    Extract the final integer answer from model output using multiple
    strategies (ordered by priority):

      1. \\boxed{N}  -- Standard LaTeX convention for math benchmarks
      2. <answer>N</answer> -- XML tag format
      3. Last standalone integer in the text

    Handles garbled output by focusing on the answer section
    (text after the last think-closing tag) first.

    Args:
        text: The raw model-generated text.

    Returns:
        Extracted integer, or None if no valid integer is found.
    """
    if not text or not text.strip():
        return None

    # --- Phase 1: Try answer section (after last think-closing tag) ---
    # The model outputs <think>..reasoning..</think>..answer..
    # The answer (with \\boxed{}) is in the section AFTER the closing tag.
    THINK_END = "</think>"
    last_think_end = text.rfind(THINK_END)
    if last_think_end != -1:
        answer_section = text[last_think_end + len(THINK_END):]
        result = _extract_from_section(answer_section)
        if result is not None:
            return result

    # --- Phase 2: Fallback to full text (but cap length to avoid garbled tail) ---
    # If the text is very long, the tail is likely garbled noise; only use first 15000 chars.
    MAX_MEANINGFUL_LEN = 15000
    truncated = text[:MAX_MEANINGFUL_LEN] if len(text) > MAX_MEANINGFUL_LEN else text
    return _extract_from_section(truncated)


def _parse_integer(s: str) -> Optional[int]:
    """
    Parse a string that may contain an integer, possibly with
    surrounding whitespace, dollar signs, or light formatting.

    Returns:
        The integer value, or None if parsing fails.
    """
    # Remove common LaTeX/formatting noise
    s = s.strip().strip("$").strip()

    # Try direct integer parse
    try:
        return int(s)
    except ValueError:
        pass

    # Try extracting first integer from the string
    match = re.search(r"(-?\d+)", s)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            pass

    return None


# ======================== Answer Checking ========================

def check_answer(predicted: Optional[int], expected: int) -> bool:
    """
    Pass@1 exact-match integer comparison.

    Args:
        predicted: The extracted integer from model output (may be None).
        expected: The ground truth integer answer.

    Returns:
        True if predicted == expected, False otherwise.
    """
    if predicted is None:
        return False
    return predicted == expected


# ======================== Batch Utilities ========================

def collate_prompts(problems: list[dict]) -> list[str]:
    """
    Build prompts for a batch of AIME problems.

    Args:
        problems: List of problem dicts with 'problem' key.

    Returns:
        List of prompt strings.
    """
    return [build_aime_prompt(p["problem"]) for p in problems]
