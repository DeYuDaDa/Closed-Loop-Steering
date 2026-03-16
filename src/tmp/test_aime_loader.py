"""
AIME Loader — Offline Test Suite (No GPU required)
=====================================================
Verifies dataset loading, answer extraction, prompt building,
and answer checking logic against known cases.

Usage:
    cd f:\\academic\\Closed-Loop-Steering-System\\src
    python test_aime_loader.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from aime_loader import (
    load_aime_dataset,
    list_aime_datasets,
    build_aime_prompt,
    extract_answer,
    check_answer,
)


def test_load_datasets():
    """Test: Load both AIME datasets and verify counts."""
    print("\n" + "=" * 60)
    print("  TEST 1: Load AIME Datasets")
    print("=" * 60)

    dataset_dir = os.path.join(os.path.dirname(__file__), "dataset")
    files = list_aime_datasets(dataset_dir)

    assert len(files) >= 2, f"Expected at least 2 .jsonl files, got {len(files)}"
    print(f"  Found {len(files)} dataset files: {[os.path.basename(f) for f in files]}")

    for fpath in files:
        name = os.path.basename(fpath)
        problems = load_aime_dataset(fpath)
        assert len(problems) == 30, \
            f"Expected 30 problems in {name}, got {len(problems)}"

        # Verify each problem has required fields
        for p in problems:
            assert "id" in p, f"Missing 'id' in problem"
            assert "problem" in p, f"Missing 'problem' in problem"
            assert "answer" in p, f"Missing 'answer' in problem"
            assert isinstance(p["answer"], int), \
                f"Answer should be int, got {type(p['answer'])}"

        # Spot-check a known answer
        if "2025" in name:
            # Problem 0: answer should be 70
            assert problems[0]["answer"] == 70, \
                f"Expected answer=70 for 2025 problem 0, got {problems[0]['answer']}"
        elif "2026" in name:
            # Problem 0 (id=1): answer should be 277
            assert problems[0]["answer"] == 277, \
                f"Expected answer=277 for 2026 problem 0, got {problems[0]['answer']}"

        print(f"  ✅ {name}: {len(problems)} problems loaded, all fields valid")

    return True


def test_extract_answer():
    """Test: Multi-strategy answer extraction."""
    print("\n" + "=" * 60)
    print("  TEST 2: Answer Extraction")
    print("=" * 60)

    test_cases = [
        # (input_text, expected_output)

        # Strategy 1: \boxed{}
        (r"The answer is \boxed{42}", 42),
        (r"Therefore, $x = \boxed{735}$.", 735),
        (r"We get \boxed{0}", 0),
        (r"\boxed{999}", 999),
        # Multiple \boxed{}, take last
        (r"First \boxed{10}, then \boxed{42}", 42),

        # Strategy 2: <answer> tag
        ("<answer>117</answer>", 117),
        ("The result is <answer>  279  </answer>.", 279),

        # Strategy 3: Last standalone integer
        ("The final answer is 610.", 610),
        ("We conclude that the answer is 149", 149),

        # Edge cases
        ("", None),
        ("No numbers here at all.", None),
        ("   ", None),

        # Large number (> 999 should not match in fallback)
        ("The answer is 12345", None),
    ]

    all_pass = True
    for text, expected in test_cases:
        result = extract_answer(text)
        status = "✅" if result == expected else "❌"
        if result != expected:
            all_pass = False
        preview = text[:50] + "..." if len(text) > 50 else text
        print(f"  {status} extract({preview!r}) → {result} (expected {expected})")

    assert all_pass, "Some extraction tests failed"
    print("\n  ✅ All extraction tests passed")
    return True


def test_build_prompt():
    """Test: Prompt construction format."""
    print("\n" + "=" * 60)
    print("  TEST 3: Prompt Construction")
    print("=" * 60)

    problem = "Find the sum of all integer bases $b>9$..."
    messages = build_aime_prompt(problem)

    # Verify essential components
    assert isinstance(messages, list), "Prompt should be a list of messages"
    assert len(messages) == 2, "Expected 2 messages (system, user)"
    
    system_msg = messages[0]
    user_msg = messages[1]
    
    assert system_msg["role"] == "system"
    assert "math competition expert" in system_msg["content"]
    assert "\\boxed{}" in system_msg["content"]
    assert "0-999" in system_msg["content"]
    
    assert user_msg["role"] == "user"
    assert user_msg["content"] == problem

    print(f"  Messages count: {len(messages)}")
    print(f"  ✅ Prompt message structure is valid")
    return True


def test_check_answer():
    """Test: Pass@1 exact-match comparison."""
    print("\n" + "=" * 60)
    print("  TEST 4: Answer Checking (Pass@1)")
    print("=" * 60)

    assert check_answer(42, 42) == True, "Equal integers should match"
    assert check_answer(42, 43) == False, "Different integers should not match"
    assert check_answer(None, 42) == False, "None prediction should not match"
    assert check_answer(0, 0) == True, "Zero should match zero"

    print("  ✅ All check_answer tests passed")
    return True


# ------------------------------------------------------------------ #

def main():
    print("=" * 60)
    print("  AIME Loader — Offline Test Suite")
    print("=" * 60)

    tests = [
        test_load_datasets,
        test_extract_answer,
        test_build_prompt,
        test_check_answer,
    ]

    results = []
    for test_fn in tests:
        try:
            passed = test_fn()
            results.append((test_fn.__name__, passed))
        except Exception as e:
            print(f"  ❌ FAIL — {test_fn.__name__}: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_fn.__name__, False))

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    all_pass = True
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}  {name}")
        if not passed:
            all_pass = False

    print()
    if all_pass:
        print("  🎉 All tests passed! AIME loader is ready.")
    else:
        print("  ⚠️  Some tests failed.")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
