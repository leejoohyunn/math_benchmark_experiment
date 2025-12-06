"""
Results Analysis and Visualization Script

Usage:
    python analyze_results.py --results results/processed/summary_*.json
"""

import argparse
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np


def load_results(results_path):
    """Load results from JSON file"""
    with open(results_path, 'r') as f:
        results = json.load(f)
    return results


def create_summary_dataframe(results):
    """Create summary dataframe from results"""
    data = []
    for result in results:
        data.append({
            'model': result['model_name'],
            'dataset': result['dataset_name'],
            'accuracy': result['metrics']['accuracy'],
            'correct': result['metrics']['correct_count'],
            'total': result['metrics']['total_examples'],
            'avg_latency': result['metrics']['avg_latency'],
            'median_latency': result['metrics']['median_latency'],
            'total_tokens': result['metrics']['total_tokens'],
            'avg_tokens': result['metrics']['avg_tokens']
        })

    df = pd.DataFrame(data)
    return df


def plot_accuracy_comparison(df, output_dir):
    """Create accuracy comparison plot"""
    plt.figure(figsize=(14, 8))

    # Pivot for heatmap
    pivot = df.pivot(index='model', columns='dataset', values='accuracy')

    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', center=0.5,
                cbar_kws={'label': 'Accuracy'})

    plt.title('Model Accuracy Across Datasets', fontsize=16, fontweight='bold')
    plt.xlabel('Dataset', fontsize=12)
    plt.ylabel('Model', fontsize=12)
    plt.tight_layout()

    output_path = Path(output_dir) / 'accuracy_heatmap.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_model_comparison(df, output_dir):
    """Create model comparison bar plot"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    datasets = df['dataset'].unique()

    for idx, dataset in enumerate(datasets):
        if idx >= 4:
            break

        ax = axes[idx // 2, idx % 2]
        dataset_df = df[df['dataset'] == dataset].sort_values('accuracy', ascending=False)

        ax.barh(dataset_df['model'], dataset_df['accuracy'], color='skyblue')
        ax.set_xlabel('Accuracy', fontsize=10)
        ax.set_title(f'{dataset}', fontsize=12, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.grid(axis='x', alpha=0.3)

        # Add value labels
        for i, (idx, row) in enumerate(dataset_df.iterrows()):
            ax.text(row['accuracy'] + 0.02, i, f"{row['accuracy']:.3f}",
                   va='center', fontsize=9)

    plt.tight_layout()
    output_path = Path(output_dir) / 'model_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_latency_vs_accuracy(df, output_dir):
    """Create latency vs accuracy scatter plot"""
    plt.figure(figsize=(12, 8))

    datasets = df['dataset'].unique()
    colors = sns.color_palette('husl', len(datasets))

    for dataset, color in zip(datasets, colors):
        dataset_df = df[df['dataset'] == dataset]
        plt.scatter(dataset_df['avg_latency'], dataset_df['accuracy'],
                   label=dataset, s=100, alpha=0.7, color=color)

        # Add model labels
        for idx, row in dataset_df.iterrows():
            plt.annotate(row['model'].split('-')[0],
                        (row['avg_latency'], row['accuracy']),
                        fontsize=8, alpha=0.7)

    plt.xlabel('Average Latency (seconds)', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Latency vs Accuracy Trade-off', fontsize=16, fontweight='bold')
    plt.legend(title='Dataset')
    plt.grid(alpha=0.3)
    plt.tight_layout()

    output_path = Path(output_dir) / 'latency_vs_accuracy.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def generate_summary_table(df, output_dir):
    """Generate summary table"""
    summary = df.groupby('model').agg({
        'accuracy': 'mean',
        'avg_latency': 'mean',
        'total_tokens': 'sum'
    }).round(3)

    summary.columns = ['Avg Accuracy', 'Avg Latency (s)', 'Total Tokens']
    summary = summary.sort_values('Avg Accuracy', ascending=False)

    print("\n" + "="*60)
    print("Summary Table")
    print("="*60)
    print(summary)
    print("")

    # Save to CSV
    output_path = Path(output_dir) / 'summary_table.csv'
    summary.to_csv(output_path)
    print(f"Saved: {output_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Analyze benchmark results")
    parser.add_argument('--results', type=str, required=True, help='Path to results JSON file')
    parser.add_argument('--output_dir', type=str, default='results/visualizations',
                       help='Output directory for plots')

    args = parser.parse_args()

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Load results
    print(f"Loading results from: {args.results}")
    results = load_results(args.results)

    # Create dataframe
    df = create_summary_dataframe(results)

    print(f"\nLoaded {len(df)} evaluation results")
    print(f"Models: {df['model'].nunique()}")
    print(f"Datasets: {df['dataset'].nunique()}")

    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_accuracy_comparison(df, args.output_dir)
    plot_model_comparison(df, args.output_dir)
    plot_latency_vs_accuracy(df, args.output_dir)

    # Generate summary table
    summary = generate_summary_table(df, args.output_dir)

    print(f"\n{'='*60}")
    print("Analysis complete!")
    print(f"{'='*60}")
    print(f"Visualizations saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
