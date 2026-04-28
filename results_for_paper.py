#!/usr/bin/env python3
"""
Finalized results plots for paper.

This module contains functions to create publication-ready plots
for k-shot evaluation results.
"""

import argparse
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_DIR = Path('results/k_shot_evaluation')
PLOTS_DIR = Path('plots')


def load_evaluation_results(csv_path: Path) -> pd.DataFrame:
    """Load evaluation results from CSV file into a pandas DataFrame."""
    return pd.read_csv(csv_path)


def get_dataset_name_from_csv_path(csv_path: Path) -> str:
    """Extract the dataset name from a results CSV filename."""
    base_name = csv_path.stem.replace('evaluation_results_', '')
    tokens = base_name.split('_')
    for token in tokens:
        if token.startswith(('seq-', 'std-', 'struct-', 'smooth-', 'perm-', 'rot-', 'bias-', 'bias', 'cifar', 'mnist', 'tiny', 'imagenet', 'eurosat', 'mit', 'resisc', 'isic')):
            return token
    return tokens[-1]


def extract_checkpoint_num(ckpt_id: str) -> int:
    """Extract checkpoint number from checkpoint_id string."""
    parts = ckpt_id.rsplit('_', 1)
    return int(parts[1]) if len(parts) == 2 and parts[1].isdigit() else 0


def get_method_label(method: str) -> str:
    """Create shortened method label for plotting."""
    parts = method.split('_')
    base_method = parts[0]
    # If method starts with 'meta', include meta-method and strategy
    if base_method.startswith('meta') and len(parts) > 1:
        meta_parts = []
        for part in parts[1:]:
            if part in ['maml', 'reptile', 'no', 'meta']:
                meta_parts.append(part)
            elif part in ['sequential', 'parallel']:
                meta_parts.append(part)
        if meta_parts:
            return f"{base_method}-{'-'.join(meta_parts)}"
        return base_method
    return base_method


def plot_k_shot_stability(
    dataset: str,
    k_values: List[int] = None,
    metric: str = 'accuracy',
    results_dir: Path = RESULTS_DIR,
    with_meta: bool = True,
    include_20task: bool = False
) -> None:
    """Plot backward 0-shot and k-shot performance as side-by-side bars for each k > 0.
    
    Creates one plot per k-value (k > 0), showing backward performance comparison
    between 0-shot (k=0) and k-shot for each method.
    
    Args:
        dataset: Dataset name (e.g., 'seq-cifar100', 'struct-cifar100')
        k_values: List of k-values to plot (default: [1, 2, 5, 10]). Use 'avg' for average across all k-values.
        metric: Metric to plot ('accuracy' or 'loss')
        results_dir: Directory containing evaluation result CSVs
        with_meta: If True, include CSVs containing 'meta' in the name
        include_20task: If True, include CSVs with '20task' in the name
    """
    if not results_dir.exists():
        print(f"Error: Results directory {results_dir} does not exist.")
        return

    # Default k_values if not specified
    if k_values is None:
        k_values = [1, 2, 5, 10]
    
    # Handle 'avg' special case - compute average across all available k-values
    use_avg = 'avg' in [str(k) for k in k_values]
    if use_avg:
        k_values = [k for k in k_values if str(k) != 'avg']
    
    # Filter to only k > 0
    k_values = [k for k in k_values if k > 0]
    if not k_values:
        print("No k-values > 0 specified.")
        return

    # Find all CSV files for this dataset
    csv_files = [f for f in results_dir.glob('*.csv') if dataset in f.name]
    
    if not csv_files:
        print(f"No CSV files found for dataset '{dataset}' in {results_dir}")
        return

    # Filter by with_meta flag
    if not with_meta:
        # Exclude files that have 'meta-' prefix or '_meta_' in the name
        csv_files = [f for f in csv_files if not (
            f.name.startswith('evaluation_results_meta-') or 
            '_meta_' in f.name
        )]
    
    # Filter by include_20task flag
    if not include_20task:
        csv_files = [f for f in csv_files if '20task' not in f.name]
    else:
        csv_files = [f for f in csv_files if '20task' in f.name]
    
    if not csv_files:
        print(f"No CSV files found after filtering for dataset '{dataset}'")
        return

    print(f"Found {len(csv_files)} CSV files for dataset '{dataset}'")

    # Load all model results
    model_results = {}
    for csv_path in csv_files:
        model_dataset = csv_path.stem.replace('evaluation_results_', '')
        results = load_evaluation_results(csv_path)
        if results.empty:
            print(f"Skipping empty CSV for {model_dataset}")
            continue
        results = results.copy()
        results['checkpoint_num'] = results['checkpoint_id'].apply(extract_checkpoint_num)
        model_results[model_dataset] = results
        print(f"  Loaded {model_dataset}: {len(results)} rows")

    if not model_results:
        print("No valid results to plot.")
        return

    # Check available k-values in data
    available_k_values = []
    for model, results in model_results.items():
        available_k_values.extend(results['k_value'].unique())
    available_k_values = sorted(set(available_k_values))
    
    # If 'avg' is requested, use all available k-values > 0
    if use_avg:
        k_values = [k for k in available_k_values if k > 0]
    else:
        k_values = [k for k in k_values if k in available_k_values]
    
    if 0 not in available_k_values:
        print(f"Warning: k=0 not found in data. Available k-values: {available_k_values}")
    
    if not k_values:
        print(f"None of the requested k-values {k_values} found in data. Available: {available_k_values}")
        return

    print(f"Plotting for k-values: {k_values}")

    # Use Dark2 colormap
    cmap = plt.cm.Dark2
    
    # Create figure with one subplot per k-value
    nrows = 1
    ncols = len(k_values)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3, 4), squeeze=False)
    axes_flat = axes.flatten()
    
    col_titles = [f'k={k}' for k in k_values]
    if use_avg:
        col_titles = ['avg']
    
    # Get sorted list of methods
    all_methods = sorted(model_results.keys())
    non_meta = [m for m in all_methods if not m.startswith('meta')]
    meta = [m for m in all_methods if m.startswith('meta')]
    sorted_methods = non_meta + meta
    
    # Plot each k-value
    for col_idx, k_val in enumerate(k_values):
        ax = axes_flat[col_idx]
        
        # Collect 0-shot and k-shot values for each method
        method_0shot = {}
        method_kshot = {}
        
        for method_idx, (method, results) in enumerate(model_results.items()):
            # Get 0-shot backward performance (k=0, eval_task_id < checkpoint_num)
            if 0 in available_k_values:
                k0_subset = results[results['k_value'] == 0]
                backward_0shot = k0_subset[k0_subset['eval_task_id'] < k0_subset['checkpoint_num']]
                if not backward_0shot.empty:
                    method_0shot[method] = backward_0shot[metric].mean()
            
            # Get k-shot backward performance (k=k_val, eval_task_id < checkpoint_num)
            k_subset = results[results['k_value'] == k_val]
            backward_kshot = k_subset[k_subset['eval_task_id'] < k_subset['checkpoint_num']]
            if not backward_kshot.empty:
                method_kshot[method] = backward_kshot[metric].mean()
        
        # If using avg, compute average across all k-values
        if use_avg and not method_kshot:
            # Compute average across all available k-values > 0
            all_kshot = []
            for k_v in k_values:
                k_sub = results[results['k_value'] == k_v]
                backward_k = k_sub[k_sub['eval_task_id'] < k_sub['checkpoint_num']]
                if not backward_k.empty:
                    all_kshot.append(backward_k[metric].mean())
            if all_kshot:
                method_kshot[method] = np.mean(all_kshot)
        
        # Prepare data for side-by-side bars
        methods_with_data = [m for m in sorted_methods if m in method_0shot or m in method_kshot]
        
        x_positions = np.arange(len(methods_with_data))
        bar_width = 0.35
        
        # 0-shot bars
        values_0shot = [method_0shot.get(m, 0) for m in methods_with_data]
        bars_0shot = ax.bar(x_positions - bar_width/2, values_0shot, bar_width, 
                           label='0-shot', color='steelblue', alpha=0.8)
        
        # k-shot bars
        values_kshot = [method_kshot.get(m, 0) for m in methods_with_data]
        bars_kshot = ax.bar(x_positions + bar_width/2, values_kshot, bar_width, 
                           label=f'{k_val}-shot', color='coral', alpha=0.8)
        
        # Add legend in top right of each subplot
        ax.legend(loc='upper right', fontsize=8)
        
        ax.set_xlabel('Method')
        ax.set_ylabel(metric.capitalize() if col_idx == 0 else '')
        ax.set_xticks(x_positions)
        ax.set_xticklabels([get_method_label(m) for m in methods_with_data], 
                          rotation=45, ha='right', fontsize=8)
        ax.grid(True, axis='y', alpha=0.3)
        ax.set_title(dataset)
        
        # Set y-axis limits based on metric with extra headroom for labels
        if metric == 'accuracy':
            ax.set_ylim(0, 110)
        else:
            ax.set_ylim(0, 1.1)
        
        # Add value labels on bars
        for bar, val in zip(bars_0shot, values_0shot):
            if val > 0:
                ax.annotate(f'{val:.1f}',
                           xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=6, rotation=90)
        for bar, val in zip(bars_kshot, values_kshot):
            if val > 0:
                ax.annotate(f'{val:.1f}',
                           xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=6, rotation=90)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    
    # Save to dataset-specific directory
    plot_dir = PLOTS_DIR / "paper_plots" / dataset
    plot_dir.mkdir(exist_ok=True, parents=True)
    
    # Build filename based on options
    filename_parts = ['stability', dataset, metric]
    if not with_meta:
        filename_parts.insert(1, 'no_meta')
    if include_20task:
        filename_parts.append('20task')
    if use_avg:
        filename_parts.append('avg')
    else:
        filename_parts.append('k' + '-'.join(str(k) for k in k_values))
    output_path = plot_dir / f'{"_".join(filename_parts)}.png'
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved k-shot stability plot to {output_path}")


def plot_forward_transfer(
    dataset: str,
    k_values: List[int] = None,
    metric: str = 'accuracy',
    results_dir: Path = RESULTS_DIR,
    with_meta: bool = True,
    include_20task: bool = False
) -> None:
    """Plot 0-shot current vs k-shot forward transfer for each k-value.
    
    Creates a single plot with:
    - Left side: 0-shot 'current' accuracy (CL Plasticity)
    - Right side: k-shot 'forward' accuracy (k-shot Forward Transfer)
    - Separated by a vertical line
    - One column per k-value
    
    Args:
        dataset: Dataset name (e.g., 'seq-cifar100', 'struct-cifar100')
        k_values: List of k-values to plot (default: [1, 2, 5, 10]). Use 'avg' for average across all k-values.
        metric: Metric to plot ('accuracy' or 'loss')
        results_dir: Directory containing evaluation result CSVs
        with_meta: If True, include CSVs containing 'meta' in the name
        include_20task: If True, include CSVs with '20task' in the name
    """
    if not results_dir.exists():
        print(f"Error: Results directory {results_dir} does not exist.")
        return

    # Default k_values if not specified
    if k_values is None:
        k_values = [1, 2, 5, 10]
    
    # Handle 'avg' special case - compute average across all available k-values
    use_avg = 'avg' in [str(k) for k in k_values]
    if use_avg:
        k_values = [k for k in k_values if str(k) != 'avg']
    
    # Filter to only k > 0
    k_values = [k for k in k_values if k > 0]
    if not k_values:
        print("No k-values > 0 specified.")
        return

    # Find all CSV files for this dataset
    csv_files = [f for f in results_dir.glob('*.csv') if dataset in f.name]
    
    if not csv_files:
        print(f"No CSV files found for dataset '{dataset}' in {results_dir}")
        return

    # Filter by with_meta flag
    if not with_meta:
        csv_files = [f for f in csv_files if not (
            f.name.startswith('evaluation_results_meta-') or 
            '_meta_' in f.name
        )]
    
    # Filter by include_20task flag
    if not include_20task:
        csv_files = [f for f in csv_files if '20task' not in f.name]
    else:
        csv_files = [f for f in csv_files if '20task' in f.name]
    
    if not csv_files:
        print(f"No CSV files found after filtering for dataset '{dataset}'")
        return

    print(f"Found {len(csv_files)} CSV files for dataset '{dataset}'")

    # Load all model results
    model_results = {}
    for csv_path in csv_files:
        model_dataset = csv_path.stem.replace('evaluation_results_', '')
        results = load_evaluation_results(csv_path)
        if results.empty:
            print(f"Skipping empty CSV for {model_dataset}")
            continue
        results = results.copy()
        results['checkpoint_num'] = results['checkpoint_id'].apply(extract_checkpoint_num)
        model_results[model_dataset] = results
        print(f"  Loaded {model_dataset}: {len(results)} rows")

    if not model_results:
        print("No valid results to plot.")
        return

    # Check available k-values in data
    available_k_values = []
    for model, results in model_results.items():
        available_k_values.extend(results['k_value'].unique())
    available_k_values = sorted(set(available_k_values))
    
    # If 'avg' is requested, use all available k-values > 0
    if use_avg:
        k_values = [k for k in available_k_values if k > 0]
    else:
        k_values = [k for k in k_values if k in available_k_values]
    
    if 0 not in available_k_values:
        print(f"Warning: k=0 not found in data. Available k-values: {available_k_values}")
    
    if not k_values:
        print(f"None of the requested k-values {k_values} found in data. Available: {available_k_values}")
        return

    print(f"Plotting for k-values: {k_values}")
    
    # Filter by include_20task flag
    if not include_20task:
        csv_files = [f for f in csv_files if '20task' not in f.name]
    else:
        csv_files = [f for f in csv_files if '20task' in f.name]
    
    if not csv_files:
        print(f"No CSV files found after filtering for dataset '{dataset}'")
        return

    print(f"Found {len(csv_files)} CSV files for dataset '{dataset}'")

    # Load all model results
    model_results = {}
    for csv_path in csv_files:
        model_dataset = csv_path.stem.replace('evaluation_results_', '')
        results = load_evaluation_results(csv_path)
        if results.empty:
            print(f"Skipping empty CSV for {model_dataset}")
            continue
        results = results.copy()
        results['checkpoint_num'] = results['checkpoint_id'].apply(extract_checkpoint_num)
        model_results[model_dataset] = results
        print(f"  Loaded {model_dataset}: {len(results)} rows")

    if not model_results:
        print("No valid results to plot.")
        return

    # Check available k-values in data
    available_k_values = []
    for model, results in model_results.items():
        available_k_values.extend(results['k_value'].unique())
    available_k_values = sorted(set(available_k_values))
    
    # Filter to requested k_values that exist in data
    k_values = [k for k in k_values if k in available_k_values]
    if 0 not in available_k_values:
        print(f"Warning: k=0 not found in data. Available k-values: {available_k_values}")
    
    if not k_values:
        print(f"None of the requested k-values {k_values} found in data. Available: {available_k_values}")
        return

    print(f"Plotting for k-values: {k_values}")

    # Get sorted list of methods
    all_methods = sorted(model_results.keys())
    non_meta = [m for m in all_methods if not m.startswith('meta')]
    meta = [m for m in all_methods if m.startswith('meta')]
    sorted_methods = non_meta + meta
    
    n_non_meta = len(non_meta)
    n_meta = len(meta)
    
    # Create figure with one column per k-value
    nrows = 1
    ncols = len(k_values)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 6), squeeze=False)
    axes_flat = axes.flatten()
    
    # Plot each k-value
    for col_idx, k_val in enumerate(k_values):
        ax = axes_flat[col_idx]
        
        # Collect 0-shot current and k-shot forward values for each method
        method_current = {}
        method_forward = {}
        
        for method_idx, (method, results) in enumerate(model_results.items()):
            # Get 0-shot current performance (k=0, eval_task_id == checkpoint_num)
            if 0 in available_k_values:
                k0_subset = results[results['k_value'] == 0]
                current = k0_subset[k0_subset['eval_task_id'] == k0_subset['checkpoint_num']]
                if not current.empty:
                    method_current[method] = current[metric].mean()
            
            # Get k-shot forward performance (k=k_val, eval_task_id > checkpoint_num)
            k_subset = results[results['k_value'] == k_val]
            forward = k_subset[k_subset['eval_task_id'] > k_subset['checkpoint_num']]
            if not forward.empty:
                method_forward[method] = forward[metric].mean()
        
        # If using avg, compute average across all k-values
        if use_avg and not method_forward:
            all_forward = []
            for k_v in k_values:
                k_sub = results[results['k_value'] == k_v]
                forward_k = k_sub[k_sub['eval_task_id'] > k_sub['checkpoint_num']]
                if not forward_k.empty:
                    all_forward.append(forward_k[metric].mean())
            if all_forward:
                method_forward[method] = np.mean(all_forward)
        
        # Prepare data for side-by-side bars
        methods_with_data = [m for m in sorted_methods if m in method_current or m in method_forward]
        bar_width = 0.5


        x_current = [x + bar_width + 0.15 for x in range(len(methods_with_data))]
        x_forward = [x_current[-1] + 1 + x + bar_width + 0.15 for x in range(len(methods_with_data))]

        current_values = [method_current.get(m, 0) for m in methods_with_data]
        forward_values = [method_forward.get(m, 0) for m in methods_with_data]
        colors = plt.cm.Dark2.colors[:len(methods_with_data)] * 2

        ax.bar(x_current+x_forward, current_values+forward_values, width=bar_width, alpha=0.85, color=colors)

        all_labels = [get_method_label(m) for m in methods_with_data] * 2
        tick_positions = x_current + x_forward
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(all_labels, rotation=45, ha='right', fontsize=8)
        secax = ax.secondary_xaxis('top')
        secax.set_xticks([sum(x_current)//len(x_current), 1+sum(x_forward)//len(x_forward)], labels=["CL Plasticity", f'{k_val}-shot Forward Transfer'])
        ax.axvline(x=(x_current[-1] + x_forward[0]) / 2, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
        
        ax.set_ylabel(metric.capitalize() if col_idx == 0 else '')
        ax.grid(True, axis='y', alpha=0.3)
        
        # Set y-axis limits based on metric with extra headroom
        if metric == 'accuracy':
            ax.set_ylim(0, 110)
        else:
            ax.set_ylim(0, 1.1)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    
    # Save to dataset-specific directory
    plot_dir = PLOTS_DIR / "paper_plots" / dataset
    plot_dir.mkdir(exist_ok=True, parents=True)
    
    # Build filename based on options
    filename_parts = ['forward_transfer', dataset, metric]
    if not with_meta:
        filename_parts.insert(1, 'no_meta')
    if include_20task:
        filename_parts.append('20task')
    if use_avg:
        filename_parts.append('avg')
    else:
        filename_parts.append('k' + '-'.join(str(k) for k in k_values))
    output_path = plot_dir / f'{"_".join(filename_parts)}.png'
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved forward transfer plot to {output_path}")


def plot_improvement(
    dataset: str,
    k_values: List[int] = None,
    metric: str = 'accuracy',
    results_dir: Path = RESULTS_DIR,
    with_meta: bool = True,
    include_20task: bool = False
) -> None:
    """Plot backward and forward k-shot improvement as 2 vertically stacked subplots.
    
    Creates a plot with 2 rows:
    - Top: Backward k-shot improvement (eval_task_id < checkpoint_num)
    - Bottom: Forward k-shot improvement (eval_task_id > checkpoint_num)
    
    Each row shows performance as checkpoint_id increases.
    
    Args:
        dataset: Dataset name (e.g., 'seq-cifar100', 'struct-cifar100')
        k_values: List of k-values to plot (default: [1, 2, 5, 10]). Use 'avg' for average across all k-values.
        metric: Metric to plot ('accuracy' or 'loss')
        results_dir: Directory containing evaluation result CSVs
        with_meta: If True, include CSVs containing 'meta' in the name
        include_20task: If True, include CSVs with '20task' in the name
    """
    if not results_dir.exists():
        print(f"Error: Results directory {results_dir} does not exist.")
        return

    # Default k_values if not specified
    if k_values is None:
        k_values = [1, 2, 5, 10]
    
    # Handle 'avg' special case
    use_avg = 'avg' in [str(k) for k in k_values]
    if use_avg:
        k_values = [k for k in k_values if str(k) != 'avg']
    
    # Filter to only k > 0
    k_values = [k for k in k_values if k > 0]
    if not k_values:
        print("No k-values > 0 specified.")
        return

    # Find all CSV files for this dataset
    csv_files = [f for f in results_dir.glob('*.csv') if dataset in f.name]
    
    if not csv_files:
        print(f"No CSV files found for dataset '{dataset}' in {results_dir}")
        return

    # Filter by with_meta flag
    if not with_meta:
        csv_files = [f for f in csv_files if not (
            f.name.startswith('evaluation_results_meta-') or 
            '_meta_' in f.name
        )]
    
    # Filter by include_20task flag
    if not include_20task:
        csv_files = [f for f in csv_files if '20task' not in f.name]
    else:
        csv_files = [f for f in csv_files if '20task' in f.name]
    
    if not csv_files:
        print(f"No CSV files found after filtering for dataset '{dataset}'")
        return

    print(f"Found {len(csv_files)} CSV files for dataset '{dataset}'")

    # Load all model results
    model_results = {}
    for csv_path in csv_files:
        model_dataset = csv_path.stem.replace('evaluation_results_', '')
        results = load_evaluation_results(csv_path)
        if results.empty:
            print(f"Skipping empty CSV for {model_dataset}")
            continue
        results = results.copy()
        results['checkpoint_num'] = results['checkpoint_id'].apply(extract_checkpoint_num)
        model_results[model_dataset] = results
        print(f"  Loaded {model_dataset}: {len(results)} rows")

    if not model_results:
        print("No valid results to plot.")
        return

    # Check available k-values in data
    available_k_values = []
    for model, results in model_results.items():
        available_k_values.extend(results['k_value'].unique())
    available_k_values = sorted(set(available_k_values))
    
    # If 'avg' is requested, use all available k-values > 0
    if use_avg:
        k_values = [k for k in available_k_values if k > 0]
    else:
        k_values = [k for k in k_values if k in available_k_values]
    
    if 0 not in available_k_values:
        print(f"Warning: k=0 not found in data. Available k-values: {available_k_values}")
    
    if not k_values:
        print(f"None of the requested k-values {k_values} found in data. Available: {available_k_values}")
        return

    print(f"Plotting for k-values: {k_values}")

    # Get sorted list of methods
    all_methods = sorted(model_results.keys())
    non_meta = [m for m in all_methods if not m.startswith('meta')]
    meta = [m for m in all_methods if m.startswith('meta')]
    sorted_methods = non_meta + meta
    
    # Use Dark2 colormap
    cmap = plt.cm.Dark2
    
    # Create figure with 2 rows (backward, forward) and one column
    nrows = 2
    ncols = 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 10), squeeze=False)
    
    row_titles = ['Backward\n(eval_task_id < checkpoint_num)', 
                  'Forward\n(eval_task_id > checkpoint_num)']
    
    # Collect handles and labels for legend
    all_handles = []
    all_labels = []
    
    # Plot each row
    for row_idx, (direction, row_title) in enumerate(zip(
        ['backward', 'forward'], row_titles)):
        ax = axes[row_idx, 0]
        
        for method_idx, (method, results) in enumerate(model_results.items()):
            # For each method, compute performance across checkpoints
            method_data = []
            
            for k_val in k_values:
                # Filter by k_value
                k_subset = results[results['k_value'] == k_val]
                if k_subset.empty:
                    continue
                
                if direction == 'forward':
                    subset = k_subset[k_subset['eval_task_id'] > k_subset['checkpoint_num']]
                else:  # backward
                    subset = k_subset[k_subset['eval_task_id'] < k_subset['checkpoint_num']]
                
                if subset.empty:
                    continue
                
                # Group by checkpoint_num and compute mean
                plot_data = subset.groupby('checkpoint_num')[[metric]].mean().reset_index()
                if plot_data.empty:
                    continue
                
                method_data.append(plot_data)
                num_checkpoints = plot_data['checkpoint_num'].nunique()
                if direction == 'forward':
                    ax.set_xticks(ticks=range(num_checkpoints), labels=range(1,num_checkpoints+1))
                else:
                    ax.set_xticks(ticks=range(1, num_checkpoints+1), labels=range(2, num_checkpoints+2))
            
            # If using avg, compute average across all k-values for each checkpoint
            if use_avg and method_data:
                # Combine all k-value data and compute mean per checkpoint
                combined = pd.concat(method_data, ignore_index=True)
                plot_data = combined.groupby('checkpoint_num')[[metric]].mean().reset_index()
            elif method_data:
                # Use the last k-value's data (or could average)
                plot_data = method_data[-1] if method_data else None
            else:
                plot_data = None
            
            if plot_data is not None and not plot_data.empty:
                plot_data = plot_data.sort_values('checkpoint_num')
                color = cmap(method_idx / max(len(model_results) - 1, 1))
                line, = ax.plot(plot_data['checkpoint_num'], plot_data[metric], 
                       marker='o', label=get_method_label(method), color=color)
                all_handles.append(line)
                all_labels.append(get_method_label(method))
        
        ax.set_title(row_title)
        ax.set_xlabel('Checkpoint Number')
        ax.set_ylabel(metric.capitalize())
        ax.grid(True)
        
        # # Set y-axis limits based on metric with extra headroom
        # if metric == 'accuracy':
        #     ax.set_ylim(0, 110)
        # else:
        #     ax.set_ylim(0, 1.1)
    
    # Add single legend outside the subplots to the right
    # Remove duplicates from legend handles/labels
    unique_pairs = []
    seen_labels = set()
    for handle, label in zip(all_handles, all_labels):
        if label not in seen_labels:
            unique_pairs.append((handle, label))
            seen_labels.add(label)
    
    if unique_pairs:
        handles, labels = zip(*unique_pairs)
        fig.legend(handles, labels, loc='center right', 
                   bbox_to_anchor=(1.02, 0.5), fontsize='small', 
                   title='Method', title_fontsize='small')
    
    plt.tight_layout()
    plt.subplots_adjust(right=0.88)
    
    # Save to dataset-specific directory
    plot_dir = PLOTS_DIR / "paper_plots" / dataset
    plot_dir.mkdir(exist_ok=True, parents=True)
    
    # Build filename based on options
    filename_parts = ['improvement', dataset, metric]
    if not with_meta:
        filename_parts.insert(1, 'no_meta')
    if include_20task:
        filename_parts.append('20task')
    if use_avg:
        filename_parts.append('avg')
    else:
        filename_parts.append('k' + '-'.join(str(k) for k in k_values))
    output_path = plot_dir / f'{"_".join(filename_parts)}.png'
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved improvement plot to {output_path}")


def plot_plasticity(
    dataset: str,
    k_values: List[int] = None,
    metric: str = 'accuracy',
    results_dir: Path = RESULTS_DIR,
    with_meta: bool = True,
    include_20task: bool = False
) -> None:
    """Plot backward and forward k-shot improvement as 2 vertically stacked subplots.
    
    Creates a plot with 2 rows:
    - Top: Backward k-shot improvement (eval_task_id < checkpoint_num)
    - Bottom: Forward k-shot improvement (eval_task_id > checkpoint_num)
    
    Each row shows performance as checkpoint_id increases.
    
    Args:
        dataset: Dataset name (e.g., 'seq-cifar100', 'struct-cifar100')
        k_values: List of k-values to plot (default: [1, 2, 5, 10]). Use 'avg' for average across all k-values.
        metric: Metric to plot ('accuracy' or 'loss')
        results_dir: Directory containing evaluation result CSVs
        with_meta: If True, include CSVs containing 'meta' in the name
        include_20task: If True, include CSVs with '20task' in the name
    """
    if not results_dir.exists():
        print(f"Error: Results directory {results_dir} does not exist.")
        return

    # Default k_values if not specified
    if k_values is None:
        k_values = [1, 2, 5, 10]
    
    # Handle 'avg' special case
    use_avg = 'avg' in [str(k) for k in k_values]
    if use_avg:
        k_values = [k for k in k_values if str(k) != 'avg']
    
    # Filter to only k > 0
    k_values = [k for k in k_values if k > 0]
    if not k_values:
        print("No k-values > 0 specified.")
        return

    # Find all CSV files for this dataset
    csv_files = [f for f in results_dir.glob('*.csv') if dataset in f.name]
    
    if not csv_files:
        print(f"No CSV files found for dataset '{dataset}' in {results_dir}")
        return

    # Filter by with_meta flag
    if not with_meta:
        csv_files = [f for f in csv_files if not (
            f.name.startswith('evaluation_results_meta-') or 
            '_meta_' in f.name
        )]
    
    # Filter by include_20task flag
    if not include_20task:
        csv_files = [f for f in csv_files if '20task' not in f.name]
    else:
        csv_files = [f for f in csv_files if '20task' in f.name]
    
    if not csv_files:
        print(f"No CSV files found after filtering for dataset '{dataset}'")
        return

    print(f"Found {len(csv_files)} CSV files for dataset '{dataset}'")

    # Load all model results
    model_results = {}
    for csv_path in csv_files:
        model_dataset = csv_path.stem.replace('evaluation_results_', '')
        results = load_evaluation_results(csv_path)
        if results.empty:
            print(f"Skipping empty CSV for {model_dataset}")
            continue
        results = results.copy()
        results['checkpoint_num'] = results['checkpoint_id'].apply(extract_checkpoint_num)
        model_results[model_dataset] = results
        print(f"  Loaded {model_dataset}: {len(results)} rows")

    if not model_results:
        print("No valid results to plot.")
        return

    # Check available k-values in data
    available_k_values = []
    for model, results in model_results.items():
        available_k_values.extend(results['k_value'].unique())
    available_k_values = sorted(set(available_k_values))
    
    # If 'avg' is requested, use all available k-values > 0
    if use_avg:
        k_values = [k for k in available_k_values if k > 0]
    else:
        k_values = [k for k in k_values if k in available_k_values]
    
    if 0 not in available_k_values:
        print(f"Warning: k=0 not found in data. Available k-values: {available_k_values}")
    
    if not k_values:
        print(f"None of the requested k-values {k_values} found in data. Available: {available_k_values}")
        return

    print(f"Plotting for k-values: {k_values}")

    # Get sorted list of methods
    all_methods = sorted(model_results.keys())
    non_meta = [m for m in all_methods if not m.startswith('meta')]
    meta = [m for m in all_methods if m.startswith('meta')]
    sorted_methods = non_meta + meta
    
    # Use Dark2 colormap
    cmap = plt.cm.Dark2
    
    # Create figure with 2 rows (backward, forward) and one column
    nrows = 2
    ncols = 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 10), squeeze=False)
    
    row_titles = ['Backward\n(eval_task_id < checkpoint_num)', 
                  'Forward\n(eval_task_id > checkpoint_num)']
    
    # Collect handles and labels for legend
    all_handles = []
    all_labels = []
    
    # Plot each row
    for row_idx, (direction, row_title) in enumerate(zip(
        ['backward', 'forward'], row_titles)):
        ax = axes[row_idx, 0]
        
        for method_idx, (method, results) in enumerate(model_results.items()):
            # For each method, compute performance across checkpoints
            method_data = []
            
            for k_val in k_values:
                # Filter by k_value
                k_subset = results[results['k_value'] == k_val]
                if k_subset.empty:
                    continue
                
                if direction == 'forward':
                    subset = k_subset[k_subset['eval_task_id'] > k_subset['checkpoint_num']]
                else:  # backward
                    subset = k_subset[k_subset['eval_task_id'] < k_subset['checkpoint_num']]
                
                if subset.empty:
                    continue
                
                # Group by checkpoint_num and compute mean
                plot_data = subset.groupby('checkpoint_num')[[metric]].mean().reset_index()
                if plot_data.empty:
                    continue
                
                method_data.append(plot_data)
                num_checkpoints = plot_data['checkpoint_num'].nunique()
                if direction == 'forward':
                    ax.set_xticks(ticks=range(num_checkpoints), labels=range(1,num_checkpoints+1))
                else:
                    ax.set_xticks(ticks=range(1, num_checkpoints+1), labels=range(2, num_checkpoints+2))
            
            # If using avg, compute average across all k-values for each checkpoint
            if use_avg and method_data:
                # Combine all k-value data and compute mean per checkpoint
                combined = pd.concat(method_data, ignore_index=True)
                plot_data = combined.groupby('checkpoint_num')[[metric]].mean().reset_index()
            elif method_data:
                # Use the last k-value's data (or could average)
                plot_data = method_data[-1] if method_data else None
            else:
                plot_data = None
            
            if plot_data is not None and not plot_data.empty:
                plot_data = plot_data.sort_values('checkpoint_num')
                color = cmap(method_idx / max(len(model_results) - 1, 1))
                line, = ax.plot(plot_data['checkpoint_num'], plot_data[metric], 
                       marker='o', label=get_method_label(method), color=color)
                all_handles.append(line)
                all_labels.append(get_method_label(method))
        
        ax.set_title(row_title)
        ax.set_xlabel('Checkpoint Number')
        ax.set_ylabel(metric.capitalize())
        ax.grid(True)
        
        # # Set y-axis limits based on metric with extra headroom
        # if metric == 'accuracy':
        #     ax.set_ylim(0, 110)
        # else:
        #     ax.set_ylim(0, 1.1)
    
    # Add single legend outside the subplots to the right
    # Remove duplicates from legend handles/labels
    unique_pairs = []
    seen_labels = set()
    for handle, label in zip(all_handles, all_labels):
        if label not in seen_labels:
            unique_pairs.append((handle, label))
            seen_labels.add(label)
    
    if unique_pairs:
        handles, labels = zip(*unique_pairs)
        fig.legend(handles, labels, loc='center right', 
                   bbox_to_anchor=(1.02, 0.5), fontsize='small', 
                   title='Method', title_fontsize='small')
    
    plt.tight_layout()
    plt.subplots_adjust(right=0.88)
    
    # Save to dataset-specific directory
    plot_dir = PLOTS_DIR / "paper_plots" / dataset
    plot_dir.mkdir(exist_ok=True, parents=True)
    
    # Build filename based on options
    filename_parts = ['improvement', dataset, metric]
    if not with_meta:
        filename_parts.insert(1, 'no_meta')
    if include_20task:
        filename_parts.append('20task')
    if use_avg:
        filename_parts.append('avg')
    else:
        filename_parts.append('k' + '-'.join(str(k) for k in k_values))
    output_path = plot_dir / f'{"_".join(filename_parts)}.png'
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved improvement plot to {output_path}")


def plot_sauce(
    dataset: str,
    metric: str = 'accuracy',
    results_dir: Path = RESULTS_DIR,
    with_meta: bool = True,
    include_20task: bool = False
) -> None:
    """Plot SAUCE values (plasticity metric) for backward and forward directions.
    
    Creates a plot with 2 rows:
    - Top: Backward SAUCE (eval_task_id < checkpoint_num)
    - Bottom: Forward SAUCE (eval_task_id > checkpoint_num)
    
    SAUCE is a k-independent plasticity metric computed from evaluation results.
    
    Args:
        dataset: Dataset name (e.g., 'seq-cifar100', 'struct-cifar100')
        metric: Metric to plot ('accuracy' or 'loss') - used for SAUCE computation
        results_dir: Directory containing evaluation result CSVs
        with_meta: If True, include CSVs containing 'meta' in the name
        include_20task: If True, include CSVs with '20task' in the name
    """
    if not results_dir.exists():
        print(f"Error: Results directory {results_dir} does not exist.")
        return

    # Find all CSV files for this dataset
    csv_files = [f for f in results_dir.glob('*.csv') if dataset in f.name]
    
    if not csv_files:
        print(f"No CSV files found for dataset '{dataset}' in {results_dir}")
        return

    # Filter by with_meta flag
    if not with_meta:
        csv_files = [f for f in csv_files if not (
            f.name.startswith('evaluation_results_meta-') or 
            '_meta_' in f.name
        )]
    
    # Filter by include_20task flag
    if not include_20task:
        csv_files = [f for f in csv_files if '20task' not in f.name]
    else:
        csv_files = [f for f in csv_files if '20task' in f.name]
    
    if not csv_files:
        print(f"No CSV files found after filtering for dataset '{dataset}'")
        return

    print(f"Found {len(csv_files)} CSV files for dataset '{dataset}'")

    # Load all model results
    model_results = {}
    for csv_path in csv_files:
        model_dataset = csv_path.stem.replace('evaluation_results_', '')
        results = load_evaluation_results(csv_path)
        if results.empty:
            print(f"Skipping empty CSV for {model_dataset}")
            continue
        
        # Check if SAUCE column exists
        if 'SAUCE' not in results.columns:
            print(f"Warning: No SAUCE column in {model_dataset}. Run plasticity computation first.")
            continue
            
        results = results.copy()
        results['checkpoint_num'] = results['checkpoint_id'].apply(extract_checkpoint_num)
        model_results[model_dataset] = results
        print(f"  Loaded {model_dataset}: {len(results)} rows")

    if not model_results:
        print("No valid results to plot.")
        return

    # Get sorted list of methods
    all_methods = sorted(model_results.keys())
    non_meta = [m for m in all_methods if not m.startswith('meta')]
    meta = [m for m in all_methods if m.startswith('meta')]
    sorted_methods = non_meta + meta
    
    # Use Dark2 colormap
    cmap = plt.cm.Dark2
    
    # Create figure with 2 rows (backward, forward) and one column
    nrows = 2
    ncols = 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 10), squeeze=False)
    
    row_titles = ['Backward SAUCE\n(eval_task_id < checkpoint_num)', 
                  'Forward SAUCE\n(eval_task_id > checkpoint_num)']
    
    # Collect handles and labels for legend
    all_handles = []
    all_labels = []
    
    # Plot each row
    for row_idx, (direction, row_title) in enumerate(zip(
        ['backward', 'forward'], row_titles)):
        ax = axes[row_idx, 0]
        
        for method_idx, (method, results) in enumerate(model_results.items()):
            # Filter by direction
            if direction == 'forward':
                subset = results[results['eval_task_id'] > results['checkpoint_num']]
            else:  # backward
                subset = results[results['eval_task_id'] < results['checkpoint_num']]
            
            if subset.empty:
                continue
            
            # Group by checkpoint_num and compute mean SAUCE
            plot_data = subset.groupby('checkpoint_num')[['SAUCE']].mean().reset_index()
            if plot_data.empty:
                continue
            
            plot_data = plot_data.sort_values('checkpoint_num')
            color = cmap(method_idx / max(len(model_results) - 1, 1))
            line, = ax.plot(plot_data['checkpoint_num'], plot_data['SAUCE'], 
                   marker='o', label=get_method_label(method), color=color)
            all_handles.append(line)
            all_labels.append(get_method_label(method))
        
        ax.set_title(row_title)
        ax.set_xlabel('Checkpoint Number')
        ax.set_ylabel('SAUCE')
        ax.grid(True)
    
    # Add single legend outside the subplots to the right
    unique_pairs = []
    seen_labels = set()
    for handle, label in zip(all_handles, all_labels):
        if label not in seen_labels:
            unique_pairs.append((handle, label))
            seen_labels.add(label)
    
    if unique_pairs:
        handles, labels = zip(*unique_pairs)
        fig.legend(handles, labels, loc='center right', 
                   bbox_to_anchor=(1.02, 0.5), fontsize='small', 
                   title='Method', title_fontsize='small')
    
    plt.tight_layout()
    plt.subplots_adjust(right=0.88)
    
    # Save to dataset-specific directory
    plot_dir = PLOTS_DIR / "paper_plots" / dataset
    plot_dir.mkdir(exist_ok=True, parents=True)
    
    # Build filename based on options
    filename_parts = ['sauce', dataset, metric]
    if not with_meta:
        filename_parts.insert(1, 'no_meta')
    if include_20task:
        filename_parts.append('20task')
    output_path = plot_dir / f'{"_".join(filename_parts)}.png'
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved SAUCE plot to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description='Plot finalized results for paper')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset name to plot (e.g., seq-cifar100)')
    parser.add_argument('--metric', type=str, choices=['accuracy', 'loss'], default='accuracy',
                        help='Metric to plot (default: accuracy)')
    parser.add_argument('--k-values', type=str, default='1,2,5,10',
                        help='Comma-separated k-values to plot (default: 1,2,5,10)')
    parser.add_argument('--no-meta', action='store_true',
                        help='Exclude meta-learning methods')
    parser.add_argument('--include-20task', action='store_true',
                        help='Include 20-task variant CSVs')
    parser.add_argument('--plot-type', type=str, default='stability',
                        choices=['stability', 'forward_transfer', 'improvement', 'sauce'],
                        help='Type of plot to create (default: stability)')
    
    args = parser.parse_args()
    
    k_values = [k.strip() for k in args.k_values.split(',')]
    # Convert to int where possible, keep 'avg' as string
    k_values = [int(k) if k != 'avg' else k for k in k_values]
    
    if args.plot_type == 'stability':
        plot_k_shot_stability(
            dataset=args.dataset,
            k_values=k_values,
            metric=args.metric,
            with_meta=not args.no_meta,
            include_20task=args.include_20task
        )
    elif args.plot_type == 'forward_transfer':
        plot_forward_transfer(
            dataset=args.dataset,
            k_values=k_values,
            metric=args.metric,
            with_meta=not args.no_meta,
            include_20task=args.include_20task
        )
    elif args.plot_type == 'improvement':
        plot_improvement(
            dataset=args.dataset,
            k_values=k_values,
            metric=args.metric,
            with_meta=not args.no_meta,
            include_20task=args.include_20task
        )
    elif args.plot_type == 'sauce':
        plot_sauce(
            dataset=args.dataset,
            metric=args.metric,
            with_meta=not args.no_meta,
            include_20task=args.include_20task
        )


if __name__ == '__main__':
    main()