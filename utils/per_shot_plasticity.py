# Copyright 2026-present, Amogh Inamdar, Vici Milenia, Richard Zemel.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import pandas as pd
import torch

RESULTS_DIR = Path('results/k_shot_evaluation')


# AUAC, SAUCE, Clipped-(S)AUAC
def compute_plasticity_score(
    strat_results: torch.Tensor,
    k_values: torch.Tensor,
    do_clip: bool = False,
    scale_losses: bool = False,
    higher_is_better: bool = True,
    upper_bound: float = 100.0,
    scale_k_values: bool = True,
    k_penalty: Optional[Callable] = None,
) -> float:
    """
    Compute a single plasticity score variant from evaluation data.

    Follows the same implementation logic as compute_plasticity_score but adapted
    for CSV data with trapezoidal integration over k values.

    Args:
        strat_results: Tensor of metric values (accuracies or accuracy-loss combinations)
                       in order of increasing k values
        k_values: Tensor of k values (number of adaptation examples) used for integration
        do_clip: If True, clip each result to be at most the max of all previous results
        scale_losses: If True, normalize results by min/max range
        higher_is_better: If True, assumes results are accuracies and reverses them for minimization
        upper_bound: If higher_is_better is True, the maximum possible accuracy value for reversal 
                     (e.g., 100 for percentage)
        scale_k_values: If True, scale the k value differences to sum to 1 for proper integration
        
    Returns:
        Plasticity score (area under curve via trapezoidal integration)
    """
    # Make a copy to avoid modifying the original
    results = strat_results.clone()

    # Ensure that lower is better
    if higher_is_better:
        results = upper_bound - results

    # Progressive clipping: each value is clipped to at most the max of all previous values
    if do_clip:
        for i in range(1, len(results)):
            results[i] = torch.min(results[i], results[i-1])

    # Scale by result range for meaningful aggregation
    if scale_losses:
        result_min = torch.min(results)
        result_max = torch.max(results)
        results = (results - result_min) / (result_max - result_min + 1.0e-9)

    # If a custom penalty function is provided, apply it to k values
    if k_penalty is not None:
        k_values = k_values.apply_(k_penalty)

    # Compute trapezoidal integration over k values
    # Area = sum of (k_i+1 - k_i) * (results_i + results_i+1) / 2
    k_diffs = k_values[1:] - k_values[:-1]
    if scale_k_values:
        k_diffs = k_diffs / torch.sum(k_diffs)

    result_pairs_avg = (results[:-1] + results[1:]) / 2
    strat_areas = k_diffs * result_pairs_avg

    # Plasticity score is the total area
    plasticity_score = torch.sum(strat_areas)

    return plasticity_score.item()


def compute_per_shot_plasticity_all_variants_csv(
    df_group: pd.DataFrame,
    metric_for_sauce: str = 'accuracy',
) -> Tuple[float, float, float, float]:
    """
    Compute all four plasticity score variants for a checkpoint/task group from CSV.

    Constructs the metric tensor combining accuracies and losses, then computes
    all four variants using different scale_losses and do_clip combinations.

    Args:
        df_group: DataFrame group with same checkpoint_id and eval_task_id,
                  but multiple k values

    Returns:
        Tuple of (AUAC, Clipped_AUAC, SAUCE, Clipped_SAUCE) scores
    """
    # Sort by k_value to ensure proper integration
    df_group = df_group.sort_values('k_value')
    k_values_np = df_group['k_value'].values.astype(np.float32)
    strat_results = df_group[metric_for_sauce].values.astype(np.float32)

    # Convert to PyTorch tensors
    k_values = torch.from_numpy(k_values_np)
    strat_results = torch.from_numpy(strat_results)
    higher_is_better = metric_for_sauce == 'accuracy'

    # Compute all four variants
    auac = compute_plasticity_score(strat_results, k_values, do_clip=False, scale_losses=False, 
                                    higher_is_better=higher_is_better)
    clipped_auac = compute_plasticity_score(strat_results, k_values, do_clip=True, scale_losses=False, 
                                            higher_is_better=higher_is_better)
    sauce = compute_plasticity_score(strat_results, k_values, do_clip=False, scale_losses=True, 
                                     higher_is_better=higher_is_better)
    clipped_sauce = compute_plasticity_score(strat_results, k_values, do_clip=True, scale_losses=True, 
                                            higher_is_better=higher_is_better)

    return auac, clipped_auac, sauce, clipped_sauce


def add_plasticity_scores_to_csv(csv_path: Path, metric_for_sauce: str = 'accuracy') -> None:
    """
    Compute per-shot plasticity scores for all groups in a CSV and save the modified file.

    For each unique (checkpoint_id, eval_task_id) group, computes four plasticity
    variants and adds them as new columns. The same score is repeated for all
    k values within a group since plasticity is defined per checkpoint/task.

    Args:
        csv_path: Path to the evaluation results CSV file
        metric_for_sauce: Metric to compute plasticity scores from
    """
    # Load CSV
    df = pd.read_csv(csv_path)

    # Initialize new columns for the four plasticity variants
    df['AUAC'] = 0.0
    df['Clipped_AUAC'] = 0.0
    df['SAUCE'] = 0.0
    df['Clipped_SAUCE'] = 0.0

    # Compute plasticity for each (checkpoint_id, eval_task_id) group
    for (ckpt_id, task_id), group_indices in df.groupby(['checkpoint_id', 'eval_task_id']).groups.items():
        df_group = df.loc[group_indices]
        auac, clipped_auac, sauce, clipped_sauce = compute_per_shot_plasticity_all_variants_csv(
            df_group, metric_for_sauce=metric_for_sauce)

        # Set plasticity scores for all rows in this group
        df.loc[group_indices, 'AUAC'] = auac
        df.loc[group_indices, 'Clipped_AUAC'] = clipped_auac
        df.loc[group_indices, 'SAUCE'] = sauce
        df.loc[group_indices, 'Clipped_SAUCE'] = clipped_sauce

    # Save modified CSV
    df.to_csv(csv_path, index=False)
    print(f"Plasticity scores added to {csv_path}")


def add_plasticity_scores_to_all_csvs(results_dir: Path = RESULTS_DIR, metric_for_sauce: str = 'accuracy') -> None:
    """
    Compute per-shot plasticity scores for all CSV files in RESULTS_DIR.

    Processes each CSV file, skipping any with errors and printing exceptions.

    Args:
        results_dir: Directory containing evaluation result CSV files
        metric_for_sauce: Metric to compute plasticity scores from
    """
    if not results_dir.exists():
        print(f"Error: Results directory {results_dir} does not exist.")
        return

    csv_files = list(results_dir.glob('*.csv'))
    if not csv_files:
        print(f"No CSV files found in {results_dir}")
        return

    print(f"Found {len(csv_files)} CSV files to process.")
    for csv_path in csv_files:
        try:
            print(f"Processing {csv_path.name}...", end=' ')
            add_plasticity_scores_to_csv(csv_path, metric_for_sauce=metric_for_sauce)
        except Exception as e:
            print(f"SKIPPED ({type(e).__name__}: {e})")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Add per-shot plasticity scores to evaluation results CSV')
    parser.add_argument('csv_file', type=str, nargs='?', help='Path to evaluation results CSV file')
    parser.add_argument('--metric', type=str, choices=['accuracy', 'loss'], default='accuracy',
                        help='Metric to compute plasticity scores from (default: accuracy)')
    parser.add_argument('--process-all', action='store_true',
                        help='Process all CSV files in RESULTS_DIR instead of a single file')
    args = parser.parse_args()

    if args.process_all:
        add_plasticity_scores_to_all_csvs(metric_for_sauce=args.metric)
    elif args.csv_file:
        csv_path = Path(args.csv_file)
        if not csv_path.exists():
            print(f"Error: CSV file {csv_path} does not exist")
            exit(1)
        add_plasticity_scores_to_csv(csv_path, metric_for_sauce=args.metric)
    else:
        parser.error("Either provide a CSV file or use --process-all")
