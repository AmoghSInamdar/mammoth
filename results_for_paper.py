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
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

RESULTS_DIR = Path('results/k_shot_evaluation')
PLOTS_DIR = Path('plots')

# Consistent color scheme for each method across all plots
METHOD_COLORS = {
    # Non-meta methods
    'sgd': '#1f77b4',       # blue
    'sgd_': '#1f77b4',      # blue variant
    'er': "#b4871f",        # gold
    'derpp': '#ff7f0e',     # orange
    'ewc_on': '#2ca02c',    # green
    'ewc': '#2ca02c',       # green variant
    'agem': '#d62728',      # red
    'mer': '#9467bd',       # purple
    'mas': '#8c564b',       # brown
    'rps': '#e377c2',       # pink
    'lwf': '#7f7f7f',       # gray
    'icarl': '#bcbd22',     # yellow-green
    'finetune': '#17becf',  # cyan
    # Meta methods
    'meta_sgd': '#1f77b4',       # blue (same as sgd)
    'meta_er': "#b4871f",        # gold (same as er)
    'meta_derpp': '#ff7f0e',    # orange (same as derpp)
    'meta_ewc': '#2ca02c',      # green (same as ewc)
    'meta_ewc_on': '#2ca02c',   # green (same as ewc_on)
    'meta_agem': '#d62728',     # red (same as agem)
    'meta_mer': '#9467bd',      # purple (same as mer)
    'meta_maml': '#8c564b',     # brown
    'meta_reptile': '#e377c2',  # pink
    'meta_no': '#7f7f7f',       # gray
}

LABEL_MAP = {
    'sgd': 'SGD',
    'derpp': 'DER++',
    'ewc-on': 'EWC',
    'ewc_on': 'EWC',
    'ewc': 'EWC',
    'agem': 'AGEM',
    'mer': 'MER',
    'lwf': 'LwF',
    'er': 'ER',
    'meta_sgd': 'Meta-SGD',
    'meta_derpp': 'Meta-DER++',
    'meta_ewc': 'Meta-EWC',
    'meta_ewc_on': 'Meta-EWC',
    'meta_agem': 'Meta-AGEM',
    'meta_mer': 'Meta-MER',
}

DATASET_MAP = {
    'seq-cifar100': 'SEQ-CIFAR100-10',
    'struct-cifar100': 'STRUCT-CIFAR100-20',
    'seq-mnist': 'SEQ-MNIST-5',
    'smooth-rot-mnist': 'ROT-MNIST-20',
}

def get_dataset_name(dataset: str, include_20task: bool = False) -> str:
    """Get consistent dataset name for labeling."""
    if include_20task and dataset == 'seq-cifar100':
        return 'SEQ-CIFAR100-20'
    return DATASET_MAP.get(dataset, dataset)

def get_method_color(method: str, cmap=plt.cm.Dark2) -> str:
    """Get consistent color for a method."""
    # Check for exact match first
    if method in METHOD_COLORS:
        return METHOD_COLORS[method]
    
    # Try prefix matching for meta methods (supports both meta_ and meta- naming)
    if method.startswith('meta_') or method.startswith('meta-'):
        base = method[5:]
        base = base.lstrip('-_')
        if base in METHOD_COLORS:
            return METHOD_COLORS[base]
        # Try first part of base using both separators
        base_part = base.split('_')[0].split('-')[0]
        if base_part in METHOD_COLORS:
            return METHOD_COLORS[base_part]
    
    # Try matching first part of method name
    method_part = method.split('_')[0].split('-')[0]
    if method_part in METHOD_COLORS:
        return METHOD_COLORS[method_part]
    
    # Default to Dark2 colormap - generate a consistent color based on hash
    import hashlib
    hash_val = int(hashlib.md5(method.encode()).hexdigest(), 16)
    return cmap(hash_val % cmap.N)


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
    include_20task: bool = False,
    do_avg: bool = False
) -> None:
    """Plot backward 0-shot and k-shot performance as side-by-side bars for each k > 0.
    
    Creates one plot per k-value (k > 0), showing backward performance comparison
    between 0-shot (k=0) and k-shot for each method.
    If do_avg=True and multiple k-values are supplied, averages across selected
    k-values and produces a single averaged subplot.
    
    Args:
        dataset: Dataset name (e.g., 'seq-cifar100', 'struct-cifar100')
        k_values: List of k-values to plot (default: [1, 2, 5, 10]). Use 'avg' for average across all k-values.
        metric: Metric to plot ('accuracy' or 'loss')
        results_dir: Directory containing evaluation result CSVs
        with_meta: If True, include CSVs containing 'meta' in the name
        include_20task: If True, include CSVs with '20task' in the name
        do_avg: If True, average metrics across all selected k-values instead of creating one subplot per k.
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

    average_over_k_values = use_avg or do_avg

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
    
    plot_k_values = [None] if average_over_k_values else k_values
    nrows = 1
    ncols = len(plot_k_values)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3.5) if dataset != 'seq-mnist' else (5 * ncols, 3.5), squeeze=False)
    axes_flat = axes.flatten()
    
    col_titles = ['avg'] if average_over_k_values else [f'k={k}' for k in k_values]
    
    # Get sorted list of methods
    all_methods = sorted(model_results.keys())
    non_meta = [m for m in all_methods if not m.startswith('meta')]
    meta = [m for m in all_methods if m.startswith('meta')]
    sorted_methods = non_meta + meta
    
    # Plot each k-value
    for col_idx, k_val in enumerate(plot_k_values):
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
            
            if average_over_k_values:
                all_kshot = []
                for k_v in k_values:
                    k_sub = results[results['k_value'] == k_v]
                    backward_k = k_sub[k_sub['eval_task_id'] < k_sub['checkpoint_num']]
                    if not backward_k.empty:
                        all_kshot.append(backward_k[metric].mean())
                if all_kshot:
                    method_kshot[method] = np.mean(all_kshot)
            else:
                # Get k-shot backward performance (k=k_val, eval_task_id < checkpoint_num)
                k_subset = results[results['k_value'] == k_val]
                backward_kshot = k_subset[k_subset['eval_task_id'] < k_subset['checkpoint_num']]
                if not backward_kshot.empty:
                    method_kshot[method] = backward_kshot[metric].mean()
        
        # Prepare data for side-by-side bars
        methods_with_data = [m for m in sorted_methods if m in method_0shot or m in method_kshot]
        
        x_positions = np.arange(len(methods_with_data))
        bar_width = 0.4
        
        # 0-shot bars (lighter version)
        values_0shot = [method_0shot.get(m, 0) for m in methods_with_data]
        colors_0shot = [get_method_color(m) for m in methods_with_data]
        bars_0shot = ax.bar(x_positions - bar_width/2, values_0shot, bar_width, 
                           label='0-shot', color=colors_0shot, alpha=0.4)
        
        # k-shot bars (darker version)
        values_kshot = [method_kshot.get(m, 0) for m in methods_with_data]
        colors_kshot = [get_method_color(m) for m in methods_with_data]
        bars_kshot = ax.bar(x_positions + bar_width/2, values_kshot, bar_width, 
                           label=f'{k_val}-shot', color=colors_kshot, alpha=0.9)
        
        # Add legend in top right of each subplot
        if dataset == 'seq-mnist':
            handles, labels = ax.get_legend_handles_labels()
            method_labels = sorted([LABEL_MAP.get(get_method_label(m), get_method_label(m)) for m in method_0shot])
            ax.legend(handles+list(bars_0shot), labels+method_labels, loc='center right', ncols=1,
                   bbox_to_anchor=(-0.2, 0.5), fontsize='small', labelspacing=1)
        
        # ax.set_xlabel('Method')
        if dataset == 'seq-mnist':
            ax.set_ylabel(metric.capitalize() if col_idx == 0 else '')
            ax.set_yticklabels([0,20,40,60,80,100], va='bottom')
        if dataset != 'seq-mnist':
            ax.yaxis.set_tick_params(length=0)
            ax.set_yticklabels([])
        # ax.set_xticks(x_positions)
        ax.set_xticks([])
        # ax.set_xticklabels([get_method_label(m) for m in methods_with_data], 
        #                   rotation=45, ha='right', fontsize=8)
        ax.grid(True, axis='y', alpha=0.3)
        ax.set_title(get_dataset_name(dataset, include_20task), fontsize=10)
        
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
    if do_avg and not use_avg:
        filename_parts.append('doavg')
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
    include_20task: bool = False,
    do_avg: bool = False
) -> None:
    """Plot 0-shot current vs k-shot forward transfer for each k-value.
    
    Creates a single plot with:
    - Left side: 0-shot 'current' accuracy (CL Plasticity)
    - Right side: k-shot 'forward' accuracy (k-shot Forward Transfer)
    - Separated by a vertical line
    - One column per k-value unless do_avg is True
    
    Args:
        dataset: Dataset name (e.g., 'seq-cifar100', 'struct-cifar100')
        k_values: List of k-values to plot (default: [1, 2, 5, 10]). Use 'avg' for average across all k-values.
        metric: Metric to plot ('accuracy' or 'loss')
        results_dir: Directory containing evaluation result CSVs
        with_meta: If True, include CSVs containing 'meta' in the name
        include_20task: If True, include CSVs with '20task' in the name
        do_avg: If True, average metrics across all selected k-values instead of creating one subplot per k.
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

    average_over_k_values = use_avg or do_avg

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
    
    plot_k_values = [None] if average_over_k_values else k_values
    # Create figure with one column per k-value
    nrows = 1
    ncols = len(plot_k_values)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3.5) if dataset != 'seq-mnist' else (5 * ncols, 3.5), squeeze=False)
    axes_flat = axes.flatten()
    
    # Plot each k-value
    for col_idx, k_val in enumerate(plot_k_values):
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
            
            if average_over_k_values:
                all_forward = []
                for k_v in k_values:
                    k_sub = results[results['k_value'] == k_v]
                    forward_k = k_sub[k_sub['eval_task_id'] > k_sub['checkpoint_num']]
                    if not forward_k.empty:
                        all_forward.append(forward_k[metric].mean())
                if all_forward:
                    method_forward[method] = np.mean(all_forward)
            else:
                # Get k-shot forward performance (k=k_val, eval_task_id > checkpoint_num)
                k_subset = results[results['k_value'] == k_val]
                forward = k_subset[k_subset['eval_task_id'] > k_subset['checkpoint_num']]
                if not forward.empty:
                    method_forward[method] = forward[metric].mean()
        
        # Prepare data for side-by-side bars
        methods_with_data = [m for m in sorted_methods if m in method_current or m in method_forward]
        bar_width = 0.5

        x_current = [x + bar_width + 0.15 for x in range(len(methods_with_data))]
        x_forward = [x_current[-1] + 1 + x + bar_width + 0.15 for x in range(len(methods_with_data))]

        current_values = [method_current.get(m, 0) for m in methods_with_data]
        forward_values = [method_forward.get(m, 0) for m in methods_with_data]
        
        # Use consistent method colors (lighter for current, darker for forward)
        colors_current = [get_method_color(m) for m in methods_with_data]
        colors_forward = [get_method_color(m) for m in methods_with_data]

        bars_0shot = ax.bar(x_current, current_values, width=bar_width, alpha=0.4, color=colors_current)
        bars_kshot = ax.bar(x_forward, forward_values, width=bar_width, alpha=0.9, color=colors_forward)

        # Add legend in top right of each subplot
        if dataset == 'seq-mnist':
            handles, labels = ax.get_legend_handles_labels()
            method_labels = sorted([LABEL_MAP.get(get_method_label(m), get_method_label(m)) for m in method_current])
            ax.legend(handles+list(bars_0shot), labels+method_labels, loc='center right', ncols=1,
                   bbox_to_anchor=(-0.2, 0.5), fontsize='small', labelspacing=1)
            
            ax.set_ylabel(metric.capitalize() if col_idx == 0 else '')
            ax.set_yticklabels([0,20,40,60,80,100], va='bottom')

        if dataset != 'seq-mnist':
            ax.yaxis.set_tick_params(length=0)
            ax.set_yticklabels([])

        all_labels = [get_method_label(m) for m in methods_with_data] * 2
        tick_positions = x_current + x_forward
        # ax.set_xticks(tick_positions)
        ax.set_xticks([])
        # ax.set_xticklabels(all_labels, rotation=45, ha='right', fontsize=8)
        secax = ax.secondary_xaxis('top')
        secax.set_xticks([sum(x_current)//len(x_current), 1+sum(x_forward)//len(x_forward)], labels=["0-shot Cur", f'{k_val}-shot Fwd'])
        ax.axvline(x=(x_current[-1] + x_forward[0]) / 2, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
        ax.set_title(get_dataset_name(dataset, include_20task), fontsize=10)
        
        # ax.set_ylabel(metric.capitalize() if col_idx == 0 else '')
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
    if do_avg and not use_avg:
        filename_parts.append('doavg')
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
    include_20task: bool = False,
    do_avg: bool = False
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
        do_avg: If True, average metrics across all selected k-values instead of using the last k-value.
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

    average_over_k_values = use_avg or do_avg

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
    fig, axes = plt.subplots(nrows, ncols, figsize=(6, 5), squeeze=False)
    
    dataset_name = get_dataset_name(dataset, include_20task)
    row_titles = [f'{dataset_name} (Backward)', f'{dataset_name} (Forward)']  #['Backward\n(eval_task_id < checkpoint_num)', 
                #   'Forward\n(eval_task_id > checkpoint_num)']
    
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
            
            # If averaging across k-values, compute average across all selected k-values for each checkpoint
            if average_over_k_values and method_data:
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
                color = get_method_color(method)
                # Use bold line for meta methods
                linewidth = 1.5 if method.startswith('meta') else 1.0
                linestyle = '-' if not with_meta or method.startswith('meta') else '--'
                alpha = 1.0 if not with_meta or method.startswith('meta') else 0.7
                line, = ax.plot(plot_data['checkpoint_num'], plot_data[metric], 
                       marker='o', markersize=5, label=get_method_label(method), color=color,
                       linewidth=linewidth, linestyle=linestyle, alpha=alpha)
                all_handles.append(line)
                all_labels.append(get_method_label(method))
        
        ax.set_title(row_title, fontsize=10)
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
            unique_pairs.append((handle, LABEL_MAP.get(label, label)))
            seen_labels.add(label)
    
    if unique_pairs:
        handles, labels = zip(*unique_pairs)
        fig.legend(handles, labels, loc='lower center', ncols=5,
                   bbox_to_anchor=(0.5, -0.1), fontsize='small') #, 
                #    title='Method', title_fontsize='small')
    
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
    if do_avg and not use_avg:
        filename_parts.append('doavg')
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
    meta = [m for m in all_methods if m.startswith('meta') and 'maml' in m and 'parallel' in m]
    sorted_methods = non_meta + meta
    model_results = {m: model_results[m] for m in sorted_methods}
    
    # Use Dark2 colormap
    cmap = plt.cm.Dark2
    
    # Create figure with 2 rows (backward, forward) and one column
    nrows = 2
    ncols = 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(6, 5), squeeze=False)
    
    dataset_name = get_dataset_name(dataset, include_20task)
    row_titles = [f'{dataset_name} (Backward)', f'{dataset_name} (Forward)'] 
    # row_titles = ['Backward SAUCE\n(eval_task_id < checkpoint_num)', 
    #               'Forward SAUCE\n(eval_task_id > checkpoint_num)']
    
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
            color = get_method_color(method)
            # Use bold line for meta methods
            linewidth = 1.5 if method.startswith('meta') else 1.0
            linestyle = '-' if not with_meta or method.startswith('meta') else '--'  
            alpha = 1.0 if not with_meta or method.startswith('meta') else 0.6         
            line, = ax.plot(plot_data['checkpoint_num'], plot_data['SAUCE'], 
                            marker='o',markersize=3, label=get_method_label(method), color=color,
                            linewidth=linewidth, linestyle=linestyle, alpha=alpha)
            all_handles.append(line)
            all_labels.append(get_method_label(method))
        
        ax.set_title(row_title, fontsize=10)
        ax.set_xlabel('Checkpoint Number')
        ax.set_ylabel('SAUCE')
        ax.grid(True)
        ax.set_yscale('log')
    
    # Add single legend outside the subplots to the right
    unique_pairs = []
    seen_labels = set()
    for handle, label in zip(all_handles, all_labels):
        if 'meta' in label.lower():
            label = '+Meta'
        if label not in seen_labels:
            unique_pairs.append((handle, LABEL_MAP.get(label, label)))
            seen_labels.add(label)
    
    if unique_pairs:
        handles, labels = zip(*unique_pairs)
        fig.legend(handles, labels, loc='lower center', ncols=5,
                   bbox_to_anchor=(0.5, -0.1), fontsize='small')
                #    title='Method', title_fontsize='small')
    
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


def plot_meta_improvement(
    dataset: str,
    k_values: List[int] = None,
    metric: str = 'accuracy',
    results_dir: Path = RESULTS_DIR,
    with_meta: bool = True,
    include_20task: bool = False,
    forward_only: bool = False,
    num_lookahead: int = 0,
    do_avg: bool = False,
    meta_method: str = 'maml',
    meta_strategy: str = 'parallel'
) -> None:
    """Plot meta vs non-meta method comparison for backward, current, and forward tasks.
    
    Creates a plot with 3 columns in a single row (or 1 column if forward_only=True):
    - Left: Backward tasks (eval_task_id < checkpoint_num)
    - Middle: Current tasks (eval_task_id == checkpoint_num)
    - Right: Forward tasks (eval_task_id > checkpoint_num + num_lookahead if num_lookahead > 0)
    
    If forward_only=True, only plots the forward transfer column.
    
    For each non-meta method that has a meta-equivalent, plots side-by-side bars.
    Uses plot_plasticity coloring with lower opacity for non-meta methods.
    
    Args:
        dataset: Dataset name (e.g., 'seq-cifar100', 'struct-cifar100')
        k_values: List of k-values to plot (default: [1, 2, 5, 10]). Use 'avg' for average.
        metric: Metric to plot ('accuracy' or 'loss')
        results_dir: Directory containing evaluation result CSVs
        with_meta: If True, include CSVs containing 'meta' in the name
        include_20task: If True, include CSVs with '20task' in the name
        forward_only: If True, only plot forward transfer column (default: False)
        num_lookahead: Number of tasks to skip ahead when computing forward accuracy (default: 0)
        do_avg: If True, average metrics across all selected k-values instead of using the last k-value.
        meta_method: Meta learning method to match (e.g., 'maml', 'reptile') (default: 'maml')
        meta_strategy: Meta learning strategy to match (e.g., 'parallel', 'sequential') (default: 'parallel')
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

    average_over_k_values = use_avg or do_avg

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

    # Identify non-meta methods and their meta equivalents
    # Map from non-meta base to meta equivalent
    non_meta_methods = []
    meta_methods = []
    
    for method in model_results.keys():
        if method.startswith('meta'):
            meta_methods.append(method)
        else:
            non_meta_methods.append(method)
    
    # Find pairs: non-meta method -> meta method
    # e.g., 'sgd' -> 'meta_sgd', 'derpp' -> 'meta_derpp', 'ewc_on' -> 'meta_ewc'
    method_pairs = {}
    for non_meta in non_meta_methods:
        # Try to find corresponding meta method
        base = non_meta.split('_')[0]  # e.g., 'sgd', 'derpp', 'ewc'
        for meta in meta_methods:
            base = 'ewc' if base.startswith('ewc') else base
            if meta_method in meta and meta_strategy in meta and (meta == f'meta-{base}' or meta.startswith(f'meta-{base}_')):
                method_pairs[non_meta] = meta
                break
    
    if not method_pairs:
        print("No matching non-meta/meta method pairs found.")
        return

    print(f"Found {len(method_pairs)} method pairs: {method_pairs}")

    # Use Dark2 colormap (same as plot_plasticity)
    cmap = plt.cm.Dark2
    
    # Determine columns to plot
    if forward_only:
        directions = ['forward']
        col_titles = [get_dataset_name(dataset, include_20task)]  #f'{dataset_name} (Forward)']
    else:
        directions = ['backward', 'current', 'forward']
        col_titles = ['Backward', 'Current', 'Forward']
    
    # Create figure with 1 row and N columns
    nrows = 1
    ncols = len(directions)
    fig, axes = plt.subplots(nrows, ncols, figsize=(2 * ncols, 3), squeeze=False)
    
    def get_xticklabel(non_meta: str, meta: str) -> str:
        """Create xticklabel: non-meta base + meta with + prefix."""
        # Non-meta: just the base technique
        non_meta_label = LABEL_MAP.get(non_meta.split('_')[0], non_meta.split('_')[0])
        # Meta: last 2 segments prefixed by '+'
        meta_parts = meta.split('_')
        meta_label = '+' + '_'.join(meta_parts[-2:]) if len(meta_parts) >= 2 else '+' + meta
        return f"{non_meta_label}" #/{meta_label}"
    
    # Plot each column
    for col_idx, (direction, col_title) in enumerate(zip(
        directions, col_titles)):
        ax = axes[0, col_idx]
        
        # Collect data for each method pair
        pair_data = []
        color_idx = 0
        
        for non_meta, meta in method_pairs.items():
            non_meta_results = model_results.get(non_meta)
            meta_results = model_results.get(meta)
            
            if non_meta_results is None or meta_results is None:
                continue
            
            # Get color for this pair
            color = get_method_color(non_meta)
            color_idx += 1
            
            # Compute performance for each k-value
            non_meta_values = []
            meta_values = []
            
            for k_val in k_values:
                k_subset_non = non_meta_results[non_meta_results['k_value'] == k_val]
                k_subset_meta = meta_results[meta_results['k_value'] == k_val]
                
                if direction == 'forward':
                    # Forward: eval_task_id > checkpoint_num + num_lookahead
                    forward_threshold = k_subset_non['checkpoint_num'] + num_lookahead
                    data_non = k_subset_non[k_subset_non['eval_task_id'] > forward_threshold]
                    forward_threshold = k_subset_meta['checkpoint_num'] + num_lookahead
                    data_meta = k_subset_meta[k_subset_meta['eval_task_id'] > forward_threshold]
                elif direction == 'backward':
                    # Backward: eval_task_id < checkpoint_num
                    data_non = k_subset_non[k_subset_non['eval_task_id'] < k_subset_non['checkpoint_num']]
                    data_meta = k_subset_meta[k_subset_meta['eval_task_id'] < k_subset_meta['checkpoint_num']]
                else:  # current
                    # Current: eval_task_id == checkpoint_num
                    # Evaluate 0-shot performance at the checkpoint task
                    k_subset_non_cur = non_meta_results[non_meta_results['k_value'] == 0]
                    k_subset_meta_cur = meta_results[meta_results['k_value'] == 0]
                    data_non = k_subset_non_cur[k_subset_non_cur['eval_task_id'] == k_subset_non_cur['checkpoint_num']]
                    data_meta = k_subset_meta_cur[k_subset_meta_cur['eval_task_id'] == k_subset_meta_cur['checkpoint_num']]
                
                if not data_non.empty:
                    non_meta_values.append(data_non[metric].mean())
                else:
                    non_meta_values.append(0)
                
                if not data_meta.empty:
                    meta_values.append(data_meta[metric].mean())
                else:
                    meta_values.append(0)
            
            # If averaging across k-values, compute average across all selected k-values
            if average_over_k_values:
                non_meta_val = np.mean(non_meta_values) if non_meta_values else 0
                meta_val = np.mean(meta_values) if meta_values else 0
            else:
                # Use the last k-value
                non_meta_val = non_meta_values[-1] if non_meta_values else 0
                meta_val = meta_values[-1] if meta_values else 0
            
            pair_data.append({
                'non_meta': non_meta,
                'meta': meta,
                'non_meta_val': non_meta_val,
                'meta_val': meta_val,
                'color': color
            })
        
        if not pair_data:
            print(f"No data found for {direction} direction.")
            continue
        
        # Plot side-by-side bars
        n_pairs = len(pair_data)
        x_positions = np.arange(n_pairs)
        bar_width = 0.35
        
        # Non-meta bars (lower opacity)
        non_meta_vals = [p['non_meta_val'] for p in pair_data]
        colors_non_meta = [p['color'] for p in pair_data]
        bars_non_meta = ax.bar(x_positions - bar_width/2, non_meta_vals, bar_width, 
                               label='Non-meta', color=colors_non_meta, alpha=0.5)
        
        # Meta bars (full opacity)
        meta_vals = [p['meta_val'] for p in pair_data]
        colors_meta = [p['color'] for p in pair_data]
        bars_meta = ax.bar(x_positions + bar_width/2, meta_vals, bar_width, 
                          label='Meta', color=colors_meta, alpha=0.9)
        
        # Add legend
        if direction == 'forward':
            ax.legend(loc='upper right', fontsize=8)
        
        # ax.set_xlabel('Method Pair')
        ax.set_ylabel(metric.capitalize() if col_idx == 0 else '')
        ax.set_title(col_title, fontsize=10)
        ax.set_xticks(x_positions)
        ax.set_xticklabels([get_xticklabel(p['non_meta'], p['meta']) for p in pair_data], 
                          rotation=45, ha='right', fontsize=8)
        ax.grid(True, axis='y', alpha=0.3)
        
        # Set y-axis limits based on metric
        if metric == 'accuracy':
            ax.set_ylim(0, 110)
        else:
            ax.set_ylim(0, 1.1)
        
        # Add value labels on bars
        for bar, val in zip(bars_non_meta, non_meta_vals):
            if val > 0:
                ax.annotate(f'{val:.1f}',
                           xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=6, rotation=90)
        for bar, val in zip(bars_meta, meta_vals):
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
    if forward_only:
        filename_parts = ['forward_meta_improvement', meta_method, meta_strategy, dataset, metric]
    else:
        filename_parts = ['meta_improvement', meta_method, meta_strategy, dataset, metric]
    if not with_meta:
        filename_parts.insert(1, 'no_meta')
    if include_20task:
        filename_parts.append('20task')
    if num_lookahead > 0:
        filename_parts.append(f'lookahead{num_lookahead}')
    if do_avg or use_avg:
        filename_parts.append('avg')
    else:
        filename_parts.append('k' + '-'.join(str(k) for k in k_values))
    output_path = plot_dir / f'{"_".join(filename_parts)}.png'
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved meta improvement plot to {output_path}")


def all_results_table(
    dataset: str,
    metric: str = 'accuracy',
    results_dir: Path = RESULTS_DIR,
    with_meta: bool = True,
    include_20task: bool = False
) -> None:
    """Create a comprehensive results table for all methods and metrics.
    
    Produces a CSV with rows for each method and columns for:
    - Average accuracy on backward, current, and forward tasks for each k-value > 0
    - Average forgetting on backward tasks for each k-value > 0
      (k-shot backward acc - 0-shot current acc for that task)
    - Backward and forward AUAC (average accuracy across all evaluations)
    - Backward and forward SAUCE
    - Backward and forward Clipped AUAC (accuracy clipped to [0, 100])
    - Backward and forward Clipped SAUCE (SAUCE clipped to [0, 1])
    
    Args:
        dataset: Dataset name (e.g., 'seq-cifar100', 'struct-cifar100')
        metric: Metric to use ('accuracy' or 'loss')
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
        results = results.copy()
        results['checkpoint_num'] = results['checkpoint_id'].apply(extract_checkpoint_num)
        model_results[model_dataset] = results
        print(f"  Loaded {model_dataset}: {len(results)} rows")

    if not model_results:
        print("No valid results to plot.")
        return

    # Get available k-values
    available_k_values = set()
    for results in model_results.values():
        available_k_values.update(results['k_value'].unique())
    available_k_values = sorted(available_k_values)
    k_values_for_table = available_k_values  #[k for k in available_k_values if k > 0]  # Only k > 0 for per-k metrics

    # Build column names
    columns = []
    for k in k_values_for_table:
        columns.extend([
            f'backward_avg_acc_k{k}',
            f'current_avg_acc_k{k}',
            f'forward_avg_acc_k{k}',
            f'forgetting_k{k}'
        ])
    columns.extend([
        'backward_AUAC',
        'forward_AUAC',
        'backward_SAUCE',
        'forward_SAUCE',
        'backward_Clipped_AUAC',
        'forward_Clipped_AUAC',
        'backward_Clipped_SAUCE',
        'forward_Clipped_SAUCE'
    ])

    # Compute data for each method
    data = {}
    for method, results in model_results.items():
        row = []
        
        # Per-k metrics
        for k in k_values_for_table:
            k_subset = results[results['k_value'] == k]
            if k_subset.empty:
                row.extend([np.nan, np.nan, np.nan, np.nan])
                continue
            
            backward = k_subset[k_subset['eval_task_id'] < k_subset['checkpoint_num']]
            current = k_subset[k_subset['eval_task_id'] == k_subset['checkpoint_num']]
            forward = k_subset[k_subset['eval_task_id'] > k_subset['checkpoint_num']]
            
            b_acc = backward[metric].mean() if not backward.empty else np.nan
            c_acc = current[metric].mean() if not current.empty else np.nan
            f_acc = forward[metric].mean() if not forward.empty else np.nan
            
            # Compute forgetting: average (k-shot backward acc - 0-shot current acc for that task)
            task_to_0_acc = {}
            zero_shot = results[results['k_value'] == 0]
            for _, z_row in zero_shot.iterrows():
                if z_row['eval_task_id'] == z_row['checkpoint_num']:
                    task_to_0_acc[z_row['eval_task_id']] = z_row[metric]
            
            diffs = []
            for _, b_row in backward.iterrows():
                task = b_row['eval_task_id']
                k_acc = b_row[metric]
                base_acc = task_to_0_acc.get(task, np.nan)
                if not np.isnan(base_acc):
                    diffs.append(k_acc - base_acc)
            
            forgetting = np.mean(diffs) if diffs else np.nan
            
            row.extend([b_acc, c_acc, f_acc, forgetting])
        
        # Overall backward/forward metrics
        backward_all = results[results['eval_task_id'] < results['checkpoint_num']]
        forward_all = results[results['eval_task_id'] > results['checkpoint_num']]
        
        if 'AUAC' in results.columns:
            b_auac= backward_all['AUAC'].mean() if not backward_all.empty else np.nan
            f_auac = forward_all['AUAC'].mean() if not forward_all.empty else np.nan
            b_clip_auac = backward_all['Clipped_AUAC'].mean() if not backward_all.empty else np.nan
            f_clip_auac = forward_all['Clipped_AUAC'].mean() if not forward_all.empty else np.nan
        else:
            b_sauce = f_sauce = b_clip_sauce = f_clip_sauce = np.nan
        
        if 'SAUCE' in results.columns:
            b_sauce = backward_all['SAUCE'].mean() if not backward_all.empty else np.nan
            f_sauce = forward_all['SAUCE'].mean() if not forward_all.empty else np.nan
            b_clip_sauce = backward_all['Clipped_SAUCE'].mean() if not backward_all.empty else np.nan
            f_clip_sauce = forward_all['Clipped_SAUCE'].mean() if not forward_all.empty else np.nan
        else:
            b_sauce = f_sauce = b_clip_sauce = f_clip_sauce = np.nan
        
        row.extend([b_auac, f_auac, b_sauce, f_sauce, b_clip_auac, f_clip_auac, b_clip_sauce, f_clip_sauce])
        
        data[method] = row

    # Create DataFrame
    df = pd.DataFrame.from_dict(data, orient='index', columns=columns)
    
    # Save to CSV
    plot_dir = PLOTS_DIR / "paper_plots" / dataset
    plot_dir.mkdir(exist_ok=True, parents=True)
    output_path = plot_dir / 'all_results_table.csv'
    df.to_csv(output_path)
    print(f"Saved results table to {output_path}")


def compare_meta_methods(
    dataset: str,
    k_values: List[int] = None,
    metric: str = 'accuracy',
    results_dir: Path = RESULTS_DIR,
    include_20task: bool = False
) -> None:
    """Compare meta methods with SAUCE and k-shot accuracy plots.
    
    Creates two figures:
    1. SAUCE values across checkpoints (backward and forward)
    2. Aggregated k-shot accuracy bar charts for each k > 0, with separate backward and forward values per subplot

    Args:
        dataset: Dataset name (e.g., 'seq-cifar100', 'struct-cifar100')
        k_values: List of k-values to plot (default: None means all available k > 0)
        metric: Metric to plot ('accuracy' or 'loss')
        results_dir: Directory containing evaluation result CSVs
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

    # Filter to only meta methods
    csv_files = [f for f in csv_files if (
        f.name.startswith('evaluation_results_meta-') or 
        '_meta_' in f.name
        and 'parallel' in f.name  # only parallel for now
    )]
    
    # Filter by include_20task flag
    if not include_20task:
        csv_files = [f for f in csv_files if '20task' not in f.name]
    else:
        csv_files = [f for f in csv_files if '20task' in f.name]
    
    if not csv_files:
        print(f"No meta method CSV files found after filtering for dataset '{dataset}'")
        return

    print(f"Found {len(csv_files)} meta method CSV files for dataset '{dataset}'")

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
        print("No valid meta method results to plot.")
        return

    # Get available k-values
    available_k_values = set()
    for results in model_results.values():
        available_k_values.update(results['k_value'].unique())
    available_k_values = sorted(available_k_values)
    if k_values is None:
        k_values_for_plot = [k for k in available_k_values if k > 0]
    else:
        requested_k_values = [k for k in k_values if k > 0]
        missing_k_values = [k for k in requested_k_values if k not in available_k_values]
        if missing_k_values:
            print(f"Warning: requested k-values {missing_k_values} not found in data. Available: {available_k_values}")
        k_values_for_plot = [k for k in requested_k_values if k in available_k_values]
        if not k_values_for_plot:
            print(f"None of the requested k-values {requested_k_values} found in data. Available: {available_k_values}")
            return

    # Figure 1: SAUCE plots
    nrows = 2
    ncols = 1
    fig1, axes1 = plt.subplots(nrows, ncols, figsize=(6, 5), squeeze=False)
    
    dataset_name = get_dataset_name(dataset, include_20task)
    row_titles = [f'{dataset_name} (Backward)', f'{dataset_name} (Forward)']
    
    # Plot SAUCE for each row
    for row_idx, (direction, row_title) in enumerate(zip(['backward', 'forward'], row_titles)):
        ax = axes1[row_idx, 0]
        
        for method_idx, (method, results) in enumerate(model_results.items()):
            # Filter by direction
            if direction == 'forward':
                subset = results[results['eval_task_id'] > results['checkpoint_num']]
            else:  # backward
                subset = results[results['eval_task_id'] < results['checkpoint_num']]
            
            if subset.empty:
                continue
            
            # Check if SAUCE column exists
            if 'SAUCE' not in results.columns:
                print(f"Warning: No SAUCE column in {method}. Skipping.")
                continue
            
            # Group by checkpoint_num and compute mean SAUCE
            plot_data = subset.groupby('checkpoint_num')[['SAUCE']].mean().reset_index()
            if plot_data.empty:
                continue
            
            plot_data = plot_data.sort_values('checkpoint_num')
            color = get_method_color(method)
            linewidth = 1
            if 'maml' in method:
                linewidth = 1
                linestyle = 'solid'
                alpha = 1
            elif 'reptile' in method:
                linewidth = 1
                linestyle = 'dashed'
                alpha = 0.8
            else:
                linewidth = 1
                linestyle = 'dotted'
                alpha = 0.6
            ax.plot(plot_data['checkpoint_num'], plot_data['SAUCE'], 
                    marker='o', markersize=2, label=get_method_label(method), 
                    color=color, linewidth=linewidth, linestyle=linestyle, alpha=alpha)
        
        ax.set_title(row_title, fontsize=10)
        ax.set_xlabel('Checkpoint Number')
        ax.set_ylabel('SAUCE')
        ax.grid(True)
        ax.set_yscale('log')
    
    # Add legend
    handles1, labels1 = axes1[0, 0].get_legend_handles_labels()
    label_set = set()
    plot_handles, plot_labels = [], []
    for handle, label in zip(handles1, labels1):
        label_print = '-'.join(label.split('-')[:2])
        if label_print not in label_set:
            label_set.add(label_print)
            plot_handles.append(handle)
            plot_labels.append(label_print)
    if handles1:
        fig1.legend(plot_handles, [LABEL_MAP.get(label, label) for label in plot_labels], 
                   loc='lower center', ncols=5, bbox_to_anchor=(0.5, -0.1), fontsize='small')
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    
    # Save SAUCE figure
    plot_dir = PLOTS_DIR / "paper_plots" / dataset
    plot_dir.mkdir(exist_ok=True, parents=True)
    
    filename_parts = ['meta_methods_sauce', dataset, metric]
    if include_20task:
        filename_parts.append('20task')
    output_path1 = plot_dir / f'{"_".join(filename_parts)}.png'
    fig1.savefig(output_path1, dpi=300, bbox_inches='tight')
    plt.close(fig1)
    print(f"Saved meta methods SAUCE plot to {output_path1}")

    # Figure 2: k-shot accuracy bar charts per k-value
    n_k = len(k_values_for_plot)
    if n_k == 0:
        print("No k-values > 0 available for accuracy plots.")
        return

    # All subplots in one row
    nrows = 1
    ncols = n_k
    fig2, axes2 = plt.subplots(nrows, ncols, figsize=(6 * ncols, 3), squeeze=False)

    # Get unique base methods and their corresponding meta methods
    base_to_methods = {}
    for method in model_results.keys():
        if method.startswith('meta-'):
            base = method.split('-')[1].split('_')[0]  # e.g., 'sgd', 'derpp', 'ewc'
            base_to_methods.setdefault(base, []).append(method)
    bases = sorted(base_to_methods.keys())
    for base in bases:
        base_to_methods[base].sort()

    for idx, k_val in enumerate(k_values_for_plot):
        ax = axes2[0, idx]

        backward_vals = []
        forward_vals = []
        colors = []
        labels = []
        hatches = []
        alphas = []
        backward_x = []
        base_centers_back = []

        x_pos = 0.0
        small_gap = 0.6
        large_gap = 1.0
        direction_gap = 1.5
        bar_width = 0.5

        edge_lws = []
        for base in bases:
            methods = base_to_methods[base]
            base_start = x_pos
            for method in methods:
                results = model_results[method]
                k_subset = results[results['k_value'] == k_val]
                if k_subset.empty:
                    backward_vals.append(np.nan)
                    forward_vals.append(np.nan)
                else:
                    backward_subset = k_subset[k_subset['eval_task_id'] < k_subset['checkpoint_num']]
                    forward_subset = k_subset[k_subset['eval_task_id'] > k_subset['checkpoint_num']]

                    backward_acc = backward_subset[metric].mean() if not backward_subset.empty else np.nan
                    forward_acc = forward_subset[metric].mean() if not forward_subset.empty else np.nan

                    backward_vals.append(backward_acc)
                    forward_vals.append(forward_acc)
                colors.append(get_method_color(base))
                labels.append(get_method_label(method))

                if 'maml' in method:
                    hatches.append(None)
                    alphas.append(1.0)
                elif 'reptile' in method:
                    hatches.append('///')
                    alphas.append(1.0)
                else:  # 'no' or others
                    hatches.append(None)
                    alphas.append(0.6)

                if 'parallel' in method:
                    edge_lws.append(1.5)
                else:
                    edge_lws.append(0.8)

                backward_x.append(x_pos)
                x_pos += small_gap

            if methods:
                base_center = base_start + small_gap * (len(methods) - 1) / 2.0
                base_centers_back.append(base_center)
            x_pos += large_gap - small_gap

        if not backward_x:
            continue

        forward_offset = backward_x[-1] + direction_gap
        forward_x = [x + forward_offset for x in backward_x]
        base_centers_fwd = [c + forward_offset for c in base_centers_back]

        for i, (x, val, color, hatch, alpha, lw) in enumerate(zip(backward_x, backward_vals, colors, hatches, alphas, edge_lws)):
            ax.bar(x, val, width=bar_width, label='Backward' if i == 0 else "", color=color, alpha=alpha, hatch=hatch, edgecolor='black', linewidth=lw)

        for i, (x, val, color, hatch, alpha, lw) in enumerate(zip(forward_x, forward_vals, colors, hatches, alphas, edge_lws)):
            ax.bar(x, val, width=bar_width, label='Forward' if i == 0 else "", color=color, alpha=alpha, hatch=hatch, edgecolor='black', linewidth=lw)

        ax.axvline(x=backward_x[-1] + direction_gap / 2.0, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)

        ax.set_xticks(base_centers_back + base_centers_fwd)
        ax.set_xticklabels(bases + bases, rotation=45, ha='right', fontsize=8)

        ax.set_title(dataset.upper())
        ax.set_ylabel(metric.capitalize())
        ax.grid(True, axis='y', alpha=0.3)
        ax.set_ylim(0, 110 if metric == 'accuracy' else 1.1)

        style_handles = [Patch(facecolor='white', edgecolor='black', label='MAML', hatch=''),
                         Patch(facecolor='white', edgecolor='black', label='Reptile', hatch='///'),
                         Patch(facecolor='white', edgecolor='black', label='No Meta', alpha=0.6)]
        parallel_handles = [Patch(facecolor='white', edgecolor='black', linewidth=1.5, label='Parallel'),
                            Patch(facecolor='white', edgecolor='black', linewidth=0.8, label='Sequential')]
        ax.legend(handles=style_handles + parallel_handles, fontsize=8, ncol=2)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)

    # Save accuracy figure
    filename_parts_acc = ['meta_methods_accuracy', dataset, metric]
    if include_20task:
        filename_parts_acc.append('20task')
    filename_parts_acc.append('k' + '-'.join(str(k) for k in k_values_for_plot))
    output_path2 = plot_dir / f'{"_".join(filename_parts_acc)}.png'
    fig2.savefig(output_path2, dpi=300, bbox_inches='tight')
    plt.close(fig2)
    print(f"Saved meta methods accuracy plot to {output_path2}")


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
    parser.add_argument('--forward-only', action='store_true',
                        help='For meta_improvement plot type: only plot forward transfer column')
    parser.add_argument('--num-lookahead', type=int, default=0,
                        help='For meta_improvement plot type: number of tasks to skip ahead when computing forward accuracy (default: 0)')
    parser.add_argument('--do-avg', action='store_true',
                        help='Average metrics across all selected k-values for applicable plot types')
    parser.add_argument('--meta-method', type=str, default='maml',
                        help='For meta_improvement plot type: meta learning method to match (default: maml)')
    parser.add_argument('--meta-strategy', type=str, default='parallel',
                        help='For meta_improvement plot type: meta learning strategy to match (default: parallel)')
    parser.add_argument('--plot-type', type=str, default='stability',
                        choices=['stability', 'forward_transfer', 'improvement', 'sauce', 'meta_improvement', 'all_results_table', 'compare_meta_methods'],
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
            include_20task=args.include_20task,
            do_avg=args.do_avg
        )
    elif args.plot_type == 'forward_transfer':
        plot_forward_transfer(
            dataset=args.dataset,
            k_values=k_values,
            metric=args.metric,
            with_meta=not args.no_meta,
            include_20task=args.include_20task,
            do_avg=args.do_avg
        )
    elif args.plot_type == 'improvement':
        plot_improvement(
            dataset=args.dataset,
            k_values=k_values,
            metric=args.metric,
            with_meta=not args.no_meta,
            include_20task=args.include_20task,
            do_avg=args.do_avg
        )
    elif args.plot_type == 'sauce':
        plot_sauce(
            dataset=args.dataset,
            metric=args.metric,
            with_meta=not args.no_meta,
            include_20task=args.include_20task
        )
    elif args.plot_type == 'meta_improvement':
        plot_meta_improvement(
            dataset=args.dataset,
            k_values=k_values,
            metric=args.metric,
            with_meta=not args.no_meta,
            include_20task=args.include_20task,
            forward_only=args.forward_only,
            num_lookahead=args.num_lookahead,
            meta_method=args.meta_method,
            meta_strategy=args.meta_strategy
        )
    elif args.plot_type == 'all_results_table':
        all_results_table(
            dataset=args.dataset,
            metric=args.metric,
            with_meta=not args.no_meta,
            include_20task=args.include_20task
        )
    elif args.plot_type == 'compare_meta_methods':
        compare_meta_methods(
            dataset=args.dataset,
            k_values=k_values,
            metric=args.metric,
            include_20task=args.include_20task
        )


if __name__ == '__main__':
    main()