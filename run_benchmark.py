"""
Main Script for Running Math Reasoning Benchmarks

Usage:
    python run_benchmark.py --models gpt-4o-mini llama-3.1-8b-instruct --datasets gsm8k math-500

    # Run all models on all datasets
    python run_benchmark.py --all

    # Run specific model on specific dataset
    python run_benchmark.py --models gpt-4o-mini --datasets gsm8k
"""

import argparse
import yaml
import json
import os
from datetime import datetime
from pathlib import Path

from models import load_model
from utils import load_dataset_split
from evaluators import BaseEvaluator


def load_config(config_path: str = "config.yaml"):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_results(results, output_dir, filename):
    """Save results to JSON file"""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {output_path}")


def run_experiment(config, model_names=None, dataset_names=None):
    """
    Run benchmarking experiment

    Args:
        config: Configuration dictionary
        model_names: List of model names to evaluate (None for all)
        dataset_names: List of dataset names to use (None for all)
    """

    # Create output directories
    os.makedirs(config['output']['raw_dir'], exist_ok=True)
    os.makedirs(config['output']['processed_dir'], exist_ok=True)

    # Determine which models and datasets to use
    if model_names is None:
        model_names = list(config['models'].keys())
    if dataset_names is None:
        dataset_names = list(config['datasets'].keys())

    print(f"\n{'='*60}")
    print(f"Math Reasoning Benchmark Experiment")
    print(f"{'='*60}")
    print(f"Models: {', '.join(model_names)}")
    print(f"Datasets: {', '.join(dataset_names)}")
    print(f"{'='*60}\n")

    # Load datasets first
    datasets = {}
    for dataset_name in dataset_names:
        dataset_config = config['datasets'][dataset_name]
        datasets[dataset_name] = load_dataset_split(
            dataset_name=dataset_name,
            hf_dataset=dataset_config['hf_dataset'],
            split=dataset_config['split'],
            sample_size=dataset_config.get('sample_size'),
            hf_config=dataset_config.get('hf_config')
        )

    # Initialize evaluator
    evaluator = BaseEvaluator(config)

    # Store all results
    all_results = []

    # Evaluate each model on each dataset
    for model_name in model_names:
        model_config = config['models'][model_name]

        # Load model
        print(f"\nLoading model: {model_name}")
        try:
            model = load_model(model_name, model_config)
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            print(f"Skipping {model_name}")
            continue

        # Evaluate on each dataset
        for dataset_name in dataset_names:
            try:
                result = evaluator.evaluate_model_on_dataset(
                    model=model,
                    model_name=model_name,
                    dataset_examples=datasets[dataset_name],
                    dataset_name=dataset_name
                )

                all_results.append(result)

                # Save individual result
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{model_name}_{dataset_name}_{timestamp}.json"
                save_results(result, config['output']['raw_dir'], filename)

            except Exception as e:
                print(f"Error evaluating {model_name} on {dataset_name}: {e}")
                continue

        # Unload model to free memory
        model.unload()
        print(f"\nModel {model_name} unloaded from memory")

    # Save aggregated results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_filename = f"summary_{timestamp}.json"
    save_results(all_results, config['output']['processed_dir'], summary_filename)

    # Print summary
    print(f"\n{'='*60}")
    print(f"Experiment Complete!")
    print(f"{'='*60}")
    print(f"\nTotal evaluations: {len(all_results)}")
    print(f"Results saved to: {config['output']['processed_dir']}/{summary_filename}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Run Math Reasoning Benchmarks")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--models', nargs='+', default=None, help='Model names to evaluate')
    parser.add_argument('--datasets', nargs='+', default=None, help='Dataset names to use')
    parser.add_argument('--all', action='store_true', help='Run all models on all datasets')

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Determine which models and datasets to run
    model_names = None if args.all else args.models
    dataset_names = None if args.all else args.datasets

    # Run experiment
    results = run_experiment(config, model_names, dataset_names)

    print("\nExperiment finished successfully!")


if __name__ == "__main__":
    main()
