"""
Evaluation Metrics
"""

from typing import List, Dict, Any
import numpy as np


def calculate_accuracy(predictions: List[bool]) -> float:
    """
    Calculate accuracy from list of correctness predictions

    Args:
        predictions: List of boolean correctness values

    Returns:
        Accuracy as float between 0 and 1
    """
    if len(predictions) == 0:
        return 0.0

    return sum(predictions) / len(predictions)


def compute_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute evaluation metrics from results

    Args:
        results: List of result dictionaries with keys:
            - 'correct': bool
            - 'latency': float
            - 'token_count': int (optional)

    Returns:
        Dictionary of computed metrics
    """

    if len(results) == 0:
        return {
            'accuracy': 0.0,
            'total_examples': 0,
            'correct_count': 0,
            'avg_latency': 0.0,
            'total_tokens': 0
        }

    correctness = [r['correct'] for r in results]
    latencies = [r['latency'] for r in results]
    token_counts = [r.get('token_count', 0) for r in results]

    metrics = {
        'accuracy': calculate_accuracy(correctness),
        'total_examples': len(results),
        'correct_count': sum(correctness),
        'avg_latency': np.mean(latencies),
        'median_latency': np.median(latencies),
        'total_tokens': sum(token_counts),
        'avg_tokens': np.mean(token_counts) if any(token_counts) else 0
    }

    return metrics


def format_metrics_report(metrics: Dict[str, Any]) -> str:
    """
    Format metrics into a readable report

    Args:
        metrics: Metrics dictionary

    Returns:
        Formatted string report
    """

    report = f"""
Evaluation Metrics:
==================
Accuracy: {metrics['accuracy']:.4f} ({metrics['correct_count']}/{metrics['total_examples']})
Avg Latency: {metrics['avg_latency']:.2f}s
Median Latency: {metrics['median_latency']:.2f}s
Total Tokens: {metrics['total_tokens']}
Avg Tokens/Example: {metrics['avg_tokens']:.1f}
"""

    return report
