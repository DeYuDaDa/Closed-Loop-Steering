import sys
import os

# Add src to path if needed
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from loaders.math500_loader import extract_answer_math500, check_answer_math500
from loaders.zebra_logic_loader import extract_answer_zebra, check_answer_zebra
from loaders.aime_loader import extract_answer as extract_answer_aime, check_answer as check_answer_aime

def test_math500_robustness():
    print("Testing MATH500 Answer Extraction...")
    
    # Case 1: Simple boxed
    text1 = "<think>reasoning</think> The answer is \\boxed{42}."
    ans1 = extract_answer_math500(text1)
    print(f"Test 1 (Simple): {ans1} {'✅' if ans1 == '42' else '❌'}")
    
    # Case 2: Complex LaTeX in boxed
    text2 = "The result is \\boxed{\\frac{1}{2} \\pi}."
    ans2 = extract_answer_math500(text2)
    print(f"Test 2 (Complex LaTeX): {ans2} {'✅' if ans2 == '\\frac{1}{2} \\pi' else '❌'}")
    
    # Case 3: Comparison with normalization
    expected = "\\frac{ 1 }{ 2 } \\pi"
    match = check_answer_math500(ans2, expected)
    print(f"Test 3 (Normalization): Expected '{expected}', Match: {match} {'✅' if match else '❌'}")

def test_zebra_robustness():
    print("\nTesting ZebraLogic Answer Extraction...")
    
    # Case 1: Simple boxed
    text1 = "<think>reasoning</think> So it is \\boxed{Bob}."
    ans1 = extract_answer_zebra(text1)
    print(f"Test 1 (Simple): {ans1} {'✅' if ans1 == 'Bob' else '❌'}")
    
    # Case 2: Extra spaces and $
    text2 = "The answer is \\boxed{  Alice  }."
    ans2 = extract_answer_zebra(text2)
    print(f"Test 2 (Spaces): {ans2} {'✅' if ans2 == 'Alice' else '❌'}")
    
    # Case 3: Comparison
    match = check_answer_zebra(ans1, "BOB")
    print(f"Test 3 (Case Insensitive): {match} {'✅' if match else '❌'}")

def test_aime_reorg():
    print("\nTesting AIME Reorg...")
    text = "<think>...</think> \\boxed{123}"
    ans = extract_answer_aime(text)
    print(f"AIME extraction: {ans} {'✅' if ans == 123 else '❌'}")

if __name__ == "__main__":
    test_math500_robustness()
    test_zebra_robustness()
    test_aime_reorg()
