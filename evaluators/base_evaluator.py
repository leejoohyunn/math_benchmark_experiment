"""
Base Evaluator for Math Reasoning Benchmarks
"""

import time
from typing import Dict, List, Any
from tqdm import tqdm

from models import ModelWrapper, get_model_response
from prompts import get_prompt
from utils import (
    extract_answer,
    check_answer_correctness,
    get_question_from_example,
    extract_answer_from_dataset,
    compute_metrics
)


class BaseEvaluator:
    """Base class for evaluating models on math datasets"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def evaluate_model_on_dataset(
        self,
        model: ModelWrapper,
        model_name: str,
        dataset_examples: List[Dict[str, Any]],
        dataset_name: str
    ) -> Dict[str, Any]:
        """
        Evaluate a single model on a dataset

        Args:
            model: Loaded model wrapper
            model_name: Name of the model
            dataset_examples: List of dataset examples
            dataset_name: Name of the dataset

        Returns:
            Dictionary containing evaluation results
        """

        print(f"\n{'='*60}")
        print(f"Evaluating {model_name} on {dataset_name}")
        print(f"{'='*60}\n")

        results = []
        predictions = []

        for idx, example in enumerate(tqdm(dataset_examples, desc=f"Evaluating")):
            # Extract question and ground truth
            question = get_question_from_example(example, dataset_name)
            ground_truth = extract_answer_from_dataset(example, dataset_name)

            # Create prompt
            prompt = get_prompt(question, model_name, self.config)

            # Generate answer
            start_time = time.time()
            try:
                generated_text = get_model_response(
                    model,
                    prompt,
                    max_new_tokens=self.config['experiment'].get('max_new_tokens', 512),
                    temperature=self.config['models'][model_name].get('temperature', 0.5)
                )
                latency = time.time() - start_time

                # Extract predicted answer
                predicted_answer = extract_answer(generated_text)

                # Check correctness
                is_correct = check_answer_correctness(
                    predicted_answer or "",
                    ground_truth
                )

            except Exception as e:
                print(f"Error on example {idx}: {e}")
                generated_text = ""
                predicted_answer = ""
                is_correct = False
                latency = 0.0

            # Store results
            result = {
                'example_id': idx,
                'question': question,
                'ground_truth': ground_truth,
                'generated_text': generated_text,
                'predicted_answer': predicted_answer,
                'correct': is_correct,
                'latency': latency,
                'token_count': len(generated_text.split())  # Rough estimate
            }

            results.append(result)
            predictions.append(result)

        # Compute metrics
        metrics = compute_metrics(results)

        # Prepare output
        output = {
            'model_name': model_name,
            'dataset_name': dataset_name,
            'metrics': metrics,
            'predictions': predictions if self.config['evaluation'].get('save_predictions', True) else []
        }

        print(f"\nResults for {model_name} on {dataset_name}:")
        print(f"Accuracy: {metrics['accuracy']:.4f} ({metrics['correct_count']}/{metrics['total_examples']})")
        print(f"Avg Latency: {metrics['avg_latency']:.2f}s")

        return output
