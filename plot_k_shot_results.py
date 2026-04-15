#!/usr/bin/env python3
"""
Plot k-shot evaluation results from CSV files.

This script reads evaluation results CSV files and creates plots with:
- One subplot per checkpoint_id
- X-axis: eval_task_id
- Y-axis: accuracy (or loss) with one line per k-value
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd

RESULTS_DIR = Path('results/k_shot_evaluation')
PLOTS_DIR = Path('plots')


def load_evaluation_results(csv_path: Path) -> pd.DataFrame:
    """Load evaluation results from CSV file into a pandas DataFrame."""
    return pd.read_csv(csv_path)


def group_results_by_checkpoint(results: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Group results by checkpoint_id."""
    return {name: group for name, group in results.groupby('checkpoint_id')}


def get_dataset_name_from_csv_path(csv_path: Path) -> str:
    """Extract the dataset name from a results CSV filename."""
    base_name = csv_path.stem.replace('evaluation_results_', '')
    tokens = base_name.split('_')
    for token in tokens:
        if token.startswith(('seq-', 'std-', 'structured-', 'perm-', 'rot-', 'bias-', 'bias', 'cifar', 'mnist', 'tiny', 'imagenet', 'eurosat', 'mit', 'resisc', 'isic')):
            return token
    return tokens[-1]


def get_dataset_name_from_model(model: str) -> str:
    """Extract the dataset portion from a model_dataset string."""
    tokens = model.split('_')
    for token in tokens:
        if token.startswith(('seq-', 'std-', 'structured-', 'perm-', 'rot-', 'bias-', 'bias', 'cifar', 'mnist', 'tiny', 'imagenet', 'eurosat', 'mit', 'resisc', 'isic')):
            return token
    return tokens[-1]


def plot_checkpoint_results(checkpoint_id: str, results: pd.DataFrame, metric: str = 'accuracy') -> None:
    """Plot results for a single checkpoint."""
    # Get unique k_values and sort
    k_values = sorted(results['k_value'].unique())

    for k_val in k_values:
        subset = results[results['k_value'] == k_val].sort_values('eval_task_id')
        plt.plot(subset['eval_task_id'], subset[metric], marker='o', label=f'k={k_val}')

    plt.xlabel('Evaluation Task ID')
    plt.ylabel(metric.capitalize())
    plt.title(f'Checkpoint: {int(checkpoint_id.split("_")[-1])+1}')
    plt.legend()
    plt.grid(True)


def plot_k_shot_results(csv_path: Path, metric: str = 'accuracy') -> None:
    """Plot results from CSV file with one subplot per checkpoint."""
    results = load_evaluation_results(csv_path)
    grouped_results = group_results_by_checkpoint(results)

    num_checkpoints = len(grouped_results)
    if num_checkpoints == 0:
        print("No results found in CSV.")
        return

    max_cols = 5
    ncols = min(max_cols, num_checkpoints)
    nrows = (num_checkpoints + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    for i, (checkpoint_id, ckpt_results) in enumerate(grouped_results.items()):
        plt.sca(axes_flat[int(checkpoint_id.split('_')[-1])])  # Use checkpoint number for subplot index
        plot_checkpoint_results(checkpoint_id, ckpt_results, metric)
        plt.xticks(ticks=range(num_checkpoints), labels=range(1, num_checkpoints+1))
        plt.axvline(x=int(checkpoint_id.split('_')[-1]), color='green', linestyle='--', alpha=0.5)

    # Hide unused subplots
    for ax in axes_flat[num_checkpoints:]:
        ax.set_visible(False)

    plt.tight_layout()
    dataset_name = get_dataset_name_from_csv_path(csv_path)
    plot_dir = PLOTS_DIR / dataset_name
    plot_dir.mkdir(exist_ok=True, parents=True)
    output_path = plot_dir / f'{metric}_{csv_path.stem.replace("evaluation_results_", "")}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {output_path}")


def plot_plasticity_scores(model: str) -> None:
    """Plot average plasticity scores for a given model across all checkpoints and tasks, including forward and backward splits."""
    csv_path = RESULTS_DIR / f'evaluation_results_{model}.csv'
    if not csv_path.exists():
        print(f"Error: CSV file {csv_path} does not exist for model {model}")
        return

    results = load_evaluation_results(csv_path)
    if results.empty:
        print(f"Error: No data found in CSV for model {model}")
        return

    # Add checkpoint_num column
    def extract_checkpoint_num(ckpt_id: str) -> int:
        parts = ckpt_id.rsplit('_', 1)
        return int(parts[1]) if len(parts) == 2 and parts[1].isdigit() else 0

    results = results.copy()
    results['checkpoint_num'] = results['checkpoint_id'].apply(extract_checkpoint_num)

    # Overall average
    avg_plasticity_overall = results.groupby('checkpoint_id')[['AUAC', 'Clipped_AUAC', 'SAUCE', 'Clipped_SAUCE']].mean().reset_index()
    avg_plasticity_overall['checkpoint_num'] = avg_plasticity_overall['checkpoint_id'].apply(extract_checkpoint_num)
    avg_plasticity_overall = avg_plasticity_overall.sort_values('checkpoint_num')

    # Forward plasticity: eval_task_id > checkpoint_num
    forward_results = results[results['eval_task_id'] > results['checkpoint_num']]
    if not forward_results.empty:
        avg_plasticity_forward = forward_results.groupby('checkpoint_id')[['AUAC', 'Clipped_AUAC', 'SAUCE', 'Clipped_SAUCE']].mean().reset_index()
        avg_plasticity_forward['checkpoint_num'] = avg_plasticity_forward['checkpoint_id'].apply(extract_checkpoint_num)
        avg_plasticity_forward = avg_plasticity_forward.sort_values('checkpoint_num')
    else:
        avg_plasticity_forward = None

    # Backward plasticity: eval_task_id < checkpoint_num
    backward_results = results[results['eval_task_id'] < results['checkpoint_num']]
    if not backward_results.empty:
        avg_plasticity_backward = backward_results.groupby('checkpoint_id')[['AUAC', 'Clipped_AUAC', 'SAUCE', 'Clipped_SAUCE']].mean().reset_index()
        avg_plasticity_backward['checkpoint_num'] = avg_plasticity_backward['checkpoint_id'].apply(extract_checkpoint_num)
        avg_plasticity_backward = avg_plasticity_backward.sort_values('checkpoint_num')
    else:
        avg_plasticity_backward = None

    dataset_name = get_dataset_name_from_model(model)
    plot_dir = PLOTS_DIR / dataset_name
    plot_dir.mkdir(exist_ok=True, parents=True)

    # Plot overall
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(avg_plasticity_overall['checkpoint_num'], avg_plasticity_overall['AUAC'], marker='o', label='AUAC')
    ax1.plot(avg_plasticity_overall['checkpoint_num'], avg_plasticity_overall['Clipped_AUAC'], marker='s', label='Clipped_AUAC')
    ax1.set_xlabel('Checkpoint Number')
    ax1.set_ylabel('Average Plasticity Score')
    ax1.set_title(f'AUAC Metrics for {model} (Overall)')
    ax1.legend()
    ax1.grid(True)
    ax2.plot(avg_plasticity_overall['checkpoint_num'], avg_plasticity_overall['SAUCE'], marker='o', label='SAUCE')
    ax2.plot(avg_plasticity_overall['checkpoint_num'], avg_plasticity_overall['Clipped_SAUCE'], marker='s', label='Clipped_SAUCE')
    ax2.set_xlabel('Checkpoint Number')
    ax2.set_ylabel('Average Plasticity Score')
    ax2.set_title(f'SAUCE Metrics for {model} (Overall)')
    ax2.legend()
    ax2.grid(True)
    plt.tight_layout()
    output_path = plot_dir / f'plasticity_{model}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Overall plasticity plot saved to {output_path}")

    # Plot forward
    if avg_plasticity_forward is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.plot(avg_plasticity_forward['checkpoint_num'], avg_plasticity_forward['AUAC'], marker='o', label='AUAC')
        ax1.plot(avg_plasticity_forward['checkpoint_num'], avg_plasticity_forward['Clipped_AUAC'], marker='s', label='Clipped_AUAC')
        ax1.set_xlabel('Checkpoint Number')
        ax1.set_ylabel('Average Plasticity Score')
        ax1.set_title(f'AUAC Metrics for {model} (Forward)')
        ax1.legend()
        ax1.grid(True)
        ax2.plot(avg_plasticity_forward['checkpoint_num'], avg_plasticity_forward['SAUCE'], marker='o', label='SAUCE')
        ax2.plot(avg_plasticity_forward['checkpoint_num'], avg_plasticity_forward['Clipped_SAUCE'], marker='s', label='Clipped_SAUCE')
        ax2.set_xlabel('Checkpoint Number')
        ax2.set_ylabel('Average Plasticity Score')
        ax2.set_title(f'SAUCE Metrics for {model} (Forward)')
        ax2.legend()
        ax2.grid(True)
        plt.tight_layout()
        output_path = plot_dir / f'forward_plasticity_{model}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Forward plasticity plot saved to {output_path}")
    else:
        print(f"No forward plasticity data for {model}")

    # Plot backward
    if avg_plasticity_backward is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.plot(avg_plasticity_backward['checkpoint_num'], avg_plasticity_backward['AUAC'], marker='o', label='AUAC')
        ax1.plot(avg_plasticity_backward['checkpoint_num'], avg_plasticity_backward['Clipped_AUAC'], marker='s', label='Clipped_AUAC')
        ax1.set_xlabel('Checkpoint Number')
        ax1.set_ylabel('Average Plasticity Score')
        ax1.set_title(f'AUAC Metrics for {model} (Backward)')
        ax1.legend()
        ax1.grid(True)
        ax2.plot(avg_plasticity_backward['checkpoint_num'], avg_plasticity_backward['SAUCE'], marker='o', label='SAUCE')
        ax2.plot(avg_plasticity_backward['checkpoint_num'], avg_plasticity_backward['Clipped_SAUCE'], marker='s', label='Clipped_SAUCE')
        ax2.set_xlabel('Checkpoint Number')
        ax2.set_ylabel('Average Plasticity Score')
        ax2.set_title(f'SAUCE Metrics for {model} (Backward)')
        ax2.legend()
        ax2.grid(True)
        plt.tight_layout()
        output_path = plot_dir / f'backward_plasticity_{model}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Backward plasticity plot saved to {output_path}")
    else:
        print(f"No backward plasticity data for {model}")


def plot_plasticity_comparisons(results_dir: Path = RESULTS_DIR, dataset: Optional[str] = None) -> None:
    """Plot forward/backward plasticity comparisons for each metric across all models.
    
    Args:
        results_dir: Directory containing evaluation result CSVs
        dataset: If specified, only plot results for this dataset
    """
    if not results_dir.exists():
        print(f"Error: Results directory {results_dir} does not exist.")
        return

    csv_files = sorted(results_dir.glob('*.csv'))
    if dataset:
        csv_files = [f for f in csv_files if dataset in f.name]
    
    if not csv_files:
        if dataset:
            print(f"No CSV files found in {results_dir} for dataset '{dataset}'")
        else:
            print(f"No CSV files found in {results_dir}")
        return

    metric_names = ['AUAC', 'Clipped_AUAC', 'SAUCE', 'Clipped_SAUCE']
    direction_labels = {
        'forward': 'Forward',
        'backward': 'Backward',
        'overall': 'Overall'
    }

    def extract_checkpoint_num(ckpt_id: str) -> int:
        parts = ckpt_id.rsplit('_', 1)
        return int(parts[1]) if len(parts) == 2 and parts[1].isdigit() else 0

    # Load all model results and compute checkpoint numbers.
    models_data = {}
    datasets = {}
    for csv_path in csv_files:
        dataset_name = get_dataset_name_from_csv_path(csv_path)
        model_dataset = csv_path.stem.replace('evaluation_results_', '')
        results = load_evaluation_results(csv_path)
        if results.empty:
            print(f"Skipping empty CSV for {model_dataset}")
            continue
        results = results.copy()
        results['checkpoint_num'] = results['checkpoint_id'].apply(extract_checkpoint_num)
        datasets.setdefault(dataset_name, []).append((model_dataset, results))

    for dataset_name, model_results in datasets.items():
        fig_base = PLOTS_DIR / dataset_name
        fig_base.mkdir(exist_ok=True, parents=True)

        for direction in ['forward', 'backward', 'overall']:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10), squeeze=False)
            for idx, metric in enumerate(metric_names):
                ax = axes[idx // 2][idx % 2]
                for model, results in model_results:
                    if direction == 'forward':
                        subset = results[results['eval_task_id'] > results['checkpoint_num']]
                    elif direction == 'backward':
                        subset = results[results['eval_task_id'] < results['checkpoint_num']]
                    else:  # overall
                        subset = results

                    if subset.empty:
                        continue

                    plot_data = subset.groupby('checkpoint_num')[[metric]].mean().reset_index()
                    if plot_data.empty:
                        continue

                    ax.plot(plot_data['checkpoint_num'], plot_data[metric], marker='o', label=model)

                ax.set_title(f'{metric} ({direction_labels[direction]})')
                ax.set_xlabel('Checkpoint Number')
                ax.set_ylabel('Average Plasticity Score')
                ax.grid(True)
                ax.legend(fontsize='small')

            plt.tight_layout()
            if direction == 'overall':
                output_path = fig_base / f'plasticity_{dataset_name}.png'
            else:
                output_path = fig_base / f'plasticity_{direction}_{dataset_name}.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved aggregate {direction} plasticity comparison for {dataset_name} to {output_path}")


def plot_all(metric: str = 'accuracy', results_dir: Path = RESULTS_DIR, plot_plasticity=False) -> None:
    """Plot all CSV files in RESULTS_DIR, skipping any with errors."""
    if not results_dir.exists():
        print(f"Error: Results directory {results_dir} does not exist.")
        return

    csv_files = list(results_dir.glob('*.csv'))
    if not csv_files:
        print(f"No CSV files found in {results_dir}")
        return

    print(f"Found {len(csv_files)} CSV files to plot.")
    for csv_path in csv_files:
        try:
            print(f"Plotting {csv_path.name}...", end=' ')
            plot_k_shot_results(csv_path, metric)
            if plot_plasticity:
                # Extract model name from CSV filename, e.g., 'evaluation_results_der_seq-cifar100.csv' -> 'der_seq-cifar100'
                model = csv_path.stem.replace('evaluation_results_', '')
                plot_plasticity_scores(model)
        except Exception as e:
            print(f"SKIPPED ({type(e).__name__}: {e})")



def main() -> None:
    parser = argparse.ArgumentParser(description='Plot k-shot evaluation results')
    parser.add_argument('csv_file', type=str, nargs='?', help='Path to the evaluation results CSV file')
    parser.add_argument('--metric', type=str, choices=['accuracy', 'loss'], default='accuracy',
                        help='Metric to plot (default: accuracy)')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Dataset name to filter results (e.g., seq-cifar100)')
    parser.add_argument('--plot-all', action='store_true',
                        help='Plot all CSV files in RESULTS_DIR instead of a single file')
    parser.add_argument('--plot-plasticity-comparisons', action='store_true',
                        help='Plot aggregate forward/backward plasticity comparisons across models')
    args = parser.parse_args()

    if args.plot_plasticity_comparisons:
        plot_plasticity_comparisons(dataset=args.dataset)
    elif args.plot_all:
        plot_all('accuracy')
        plot_all('loss')
    elif args.csv_file:
        csv_path = Path(os.path.join(RESULTS_DIR, args.csv_file))
        if not csv_path.exists():
            print(f"Error: CSV file {csv_path} does not exist.")
            return
        plot_k_shot_results(csv_path, args.metric)
    else:
        parser.error("Either provide a CSV file, use --plot-all, or use --plot-plasticity-comparisons")


if __name__ == '__main__':
    main()
