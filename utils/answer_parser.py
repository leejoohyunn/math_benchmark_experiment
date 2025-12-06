"""
Answer Extraction and Parsing Utilities
"""

import re
from typing import Optional


def extract_answer(text: str) -> Optional[str]:
    """
    Extract the final answer from model output

    Looks for patterns like:
    - \\boxed{answer}
    - answer is X
    - = X (at end of line)

    Args:
        text: Generated text from model

    Returns:
        Extracted answer string or None
    """

    # Pattern 1: LaTeX boxed answer - most reliable
    boxed_pattern = r'\\boxed\{([^}]+)\}'
    boxed_match = re.findall(boxed_pattern, text)
    if boxed_match:
        return boxed_match[-1].strip()  # Return last boxed answer

    # Pattern 2: "answer is X" or "Answer is X"
    answer_pattern = r'(?:answer|Answer)\s+is\s+([^\.\n]+)'
    answer_match = re.search(answer_pattern, text)
    if answer_match:
        return answer_match.group(1).strip()

    # Pattern 3: "= X" at end of line
    equals_pattern = r'=\s*([^\n=]+?)(?:\.|$)'
    equals_matches = re.findall(equals_pattern, text)
    if equals_matches:
        return equals_matches[-1].strip()

    # Pattern 4: Last number in the text (fallback)
    number_pattern = r'[-+]?\d*\.?\d+'
    number_matches = re.findall(number_pattern, text)
    if number_matches:
        return number_matches[-1].strip()

    return None


def normalize_answer(answer: str) -> str:
    """
    Normalize answer for comparison

    - Remove spaces
    - Convert to lowercase
    - Remove commas in numbers
    - Standardize fractions

    Args:
        answer: Answer string

    Returns:
        Normalized answer
    """
    if answer is None:
        return ""

    # Remove whitespace
    answer = answer.strip().replace(" ", "")

    # Convert to lowercase
    answer = answer.lower()

    # Remove commas from numbers (e.g., 1,000 -> 1000)
    answer = answer.replace(",", "")

    # Remove $ signs
    answer = answer.replace("$", "")

    # Remove % signs but keep the number
    answer = answer.replace("%", "")

    # Normalize fractions: \frac{a}{b} -> a/b
    frac_pattern = r'\\frac\{([^}]+)\}\{([^}]+)\}'
    answer = re.sub(frac_pattern, r'\1/\2', answer)

    return answer


def check_answer_correctness(
    predicted: str,
    ground_truth: str,
    tolerance: float = 1e-4
) -> bool:
    """
    Check if predicted answer matches ground truth

    Args:
        predicted: Predicted answer
        ground_truth: Ground truth answer
        tolerance: Numerical tolerance for floating point comparison

    Returns:
        True if answers match, False otherwise
    """

    # Normalize both answers
    pred_norm = normalize_answer(predicted)
    gt_norm = normalize_answer(ground_truth)

    # Exact string match
    if pred_norm == gt_norm:
        return True

    # Try numerical comparison
    try:
        pred_val = float(eval(pred_norm))
        gt_val = float(eval(gt_norm))
        return abs(pred_val - gt_val) < tolerance
    except:
        pass

    return False


def extract_answer_from_dataset(example: dict, dataset_name: str) -> str:
    """
    Extract ground truth answer from dataset example

    Args:
        example: Dataset example
        dataset_name: Name of the dataset

    Returns:
        Ground truth answer string
    """

    if dataset_name == "gsm8k":
        # GSM8K format: answer is at the end after ####
        answer_text = example['answer']
        if '####' in answer_text:
            return answer_text.split('####')[1].strip()
        return answer_text

    elif dataset_name in ["math-500", "omni-math"]:
        # MATH dataset format
        if 'solution' in example:
            # Extract from boxed answer in solution
            solution = example['solution']
            extracted = extract_answer(solution)
            if extracted:
                return extracted
        if 'answer' in example:
            return str(example['answer'])

    elif dataset_name == "aime-25":
        # AIME format
        if 'answer' in example:
            return str(example['answer'])

    return ""
