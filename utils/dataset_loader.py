"""
Dataset Loading Utilities
"""

from datasets import load_dataset
from typing import List, Dict, Any


def load_dataset_split(
    dataset_name: str,
    hf_dataset: str,
    split: str,
    sample_size: int = None,
    hf_config: str = None
) -> List[Dict[str, Any]]:
    """
    Load a dataset split from HuggingFace

    Args:
        dataset_name: Name identifier for the dataset
        hf_dataset: HuggingFace dataset identifier
        split: Dataset split to load
        sample_size: Number of samples to load (None for all)
        hf_config: HuggingFace dataset config name (if needed)

    Returns:
        List of dataset examples
    """

    print(f"Loading dataset: {dataset_name} ({hf_dataset}, split={split})")

    # Load dataset
    if hf_config:
        dataset = load_dataset(hf_dataset, hf_config, split=split)
    else:
        dataset = load_dataset(hf_dataset, split=split)

    # Sample if needed
    if sample_size is not None and sample_size < len(dataset):
        dataset = dataset.select(range(sample_size))
        print(f"Sampled {sample_size} examples")
    else:
        print(f"Loaded {len(dataset)} examples")

    # Convert to list of dicts
    examples = []
    for item in dataset:
        examples.append(dict(item))

    return examples


def get_question_from_example(example: Dict[str, Any], dataset_name: str) -> str:
    """
    Extract question text from dataset example

    Args:
        example: Dataset example
        dataset_name: Name of the dataset

    Returns:
        Question text
    """

    if dataset_name == "gsm8k":
        return example['question']

    elif dataset_name in ["math-500", "omni-math"]:
        return example.get('problem', example.get('question', ''))

    elif dataset_name == "aime-25":
        return example.get('problem', example.get('question', ''))

    return str(example)
