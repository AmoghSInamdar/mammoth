#!/usr/bin/env python3
"""
Gradient-only backbone comparison bar chart for the paper.

This is a trimmed copy of plot_backbone_comparison: left of the separator,
k-shot accuracy (0/5/10-shot) for SGD/A-GEM/EWC with the ResNet backbone;
right of it, the same methods with the VLM backbone using only the gradient
adaptation mode. The sub-group labels are dropped since there is one mode per
method, and a flag selects which accuracy column is plotted. Aggregation and
CSV-loading helpers come from plot_utils so they are not duplicated per figure.
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from plot_utils import (
    ACCURACY_COLUMNS,
    aggregate_metric,
    backbone_from_checkpoint_id,
    find_vlm_csv,
    k_alpha,
    load_with_checkpoint_num,
    select_method_rows,
    vlm_backbone_from_csv,
)
from results_for_paper import LABEL_MAP, get_method_color

import numpy as np


def plot_gradient_comparison(
    resnet_dir: Path,
    resnet_dataset: str,
    vlm_dir: Path,
    methods: List[str],
    k_values: List[int],
    accuracy: str = 'cil',
    direction: str = 'all',
    resnet_label: str = 'ResNet-18',
    vlm_csv: Optional[Path] = None,
) -> None:
    """Draw and save the two-backbone k-shot comparison bar chart, gradient
    only. Left of the separator each method shows one bar per k on the ResNet
    backbone; right of it each method shows one bar per k using the VLM's
    gradient adaptation mode.

    Args:
        resnet_dir: Directory with ResNet k-shot evaluation CSVs.
        resnet_dataset: Dataset name of the ResNet CSVs (e.g. 'seq-cifar100').
        vlm_dir: Directory with VLM adaptation results (one dataset).
        methods: Base method names (e.g. ['sgd', 'agem', 'ewc-on']).
        k_values: Shot counts to plot (e.g. [0, 5, 10]).
        accuracy: Accuracy variant key from ACCURACY_COLUMNS (default: 'cil').
        direction: Grid cells to average: 'all', 'backward' or 'forward' (default: 'all').
        resnet_label: Display label for the left backbone (default: 'ResNet-18').
        vlm_csv: One merged gradient eval CSV holding every method, split by
            checkpoint id. Used instead of searching vlm_dir (default: None).

    Returns:
        None: The figure is written to plotting/plots/<vlm dataset>/.
    """
    column, acc_tag, y_label = ACCURACY_COLUMNS[accuracy]

    resnet_results: Dict[str, "pd.DataFrame"] = {}
    for method in methods:
        csv_path = resnet_dir / f'evaluation_results_{method}_{resnet_dataset}.csv'
        if not csv_path.exists():
            print(f"Warning: missing ResNet CSV {csv_path}, skipping {method}")
            continue
        resnet_results[method] = load_with_checkpoint_num(csv_path)
        print(f"Loaded ResNet {method}: {len(resnet_results[method])} rows")

    vlm_results: Dict[str, "pd.DataFrame"] = {}
    vlm_backbone = 'VLM'
    if vlm_csv is not None:
        merged = load_with_checkpoint_num(vlm_csv)
        for method in methods:
            vlm_method = f'vlm-{method.replace("_", "-")}'
            rows = select_method_rows(merged, vlm_method)
            if rows.empty:
                print(f"Warning: no {vlm_method} rows in {vlm_csv.name}, skipping")
                continue
            vlm_results[method] = rows
            vlm_backbone = backbone_from_checkpoint_id(rows['checkpoint_id'].iloc[0])
            print(f"Loaded VLM {vlm_method}/gradient: {len(rows)} rows from {vlm_csv.name}")
    else:
        for method in methods:
            vlm_method = f'vlm-{method.replace("_", "-")}'
            csv_path = find_vlm_csv(vlm_dir, vlm_method, mode='gradient')
            if csv_path is None:
                print(f"Warning: missing VLM CSV for {vlm_method}/gradient, skipping")
                continue
            vlm_results[method] = load_with_checkpoint_num(csv_path)
            vlm_backbone = vlm_backbone_from_csv(csv_path, vlm_method)
            print(f"Loaded VLM {vlm_method}/gradient: {csv_path}")

    if not resnet_results and not vlm_results:
        print("No results to plot.")
        return

    fig, ax = plt.subplots(figsize=(12, 3.5))

    bar_width = 0.6
    small_gap = 0.8    # between k bars inside one method sub-group
    method_gap = 1.9   # between method groups
    direction_gap = 2.6  # across the backbone separator

    n_k = len(k_values)
    method_positions: List[float] = []
    method_labels: List[str] = []

    def draw_group(x_pos: float, results: "pd.DataFrame", color: str) -> float:
        """Draw one sub-group of k-shot bars starting at x_pos. Bars share a
        color and shade darker with increasing k.

        Args:
            x_pos: X coordinate of the first bar.
            results: Evaluation results to aggregate.
            color: Bar face color for the method.

        Returns:
            float: X coordinate just after the last bar.
        """
        for k_idx, k in enumerate(k_values):
            val, ste = aggregate_metric(results, k, column, direction)
            yerr = ste if np.isfinite(ste) else None
            ax.bar(x_pos, val, width=bar_width, color=color,
                   alpha=k_alpha(k_idx, n_k),
                   edgecolor='black', linewidth=0.6,
                   yerr=yerr, ecolor='black', capsize=2)
            x_pos += small_gap
        return x_pos

    x_pos = 0.0
    for method in methods:
        if method not in resnet_results:
            continue
        group_start = x_pos
        x_pos = draw_group(x_pos, resnet_results[method], get_method_color(method))
        group_center = (group_start + x_pos - small_gap) / 2.0
        method_positions.append(group_center)
        method_labels.append(LABEL_MAP.get(method, method.upper()))
        x_pos += method_gap - small_gap

    left_last_bar = x_pos - method_gap
    separator_x = left_last_bar + direction_gap / 2.0
    x_pos = left_last_bar + direction_gap

    right_start = x_pos
    for method in methods:
        if method not in vlm_results:
            continue
        group_start = x_pos
        x_pos = draw_group(x_pos, vlm_results[method], get_method_color(method))
        group_center = (group_start + x_pos - small_gap) / 2.0
        method_positions.append(group_center)
        method_labels.append(LABEL_MAP.get(method, method.upper()))
        x_pos += method_gap - small_gap
    right_end = x_pos - method_gap

    ax.axvline(x=separator_x, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)

    ax.set_xticks(method_positions)
    ax.set_xticklabels(method_labels, fontsize=9)
    ax.set_xlim(-1.0, right_end + 1.0)

    ax.text(left_last_bar / 2.0, 1.04, resnet_label,
            transform=ax.get_xaxis_transform(), ha='center', va='bottom',
            fontsize=11, fontweight='bold')
    ax.text((right_start + right_end) / 2.0, 1.04, vlm_backbone,
            transform=ax.get_xaxis_transform(), ha='center', va='bottom',
            fontsize=11, fontweight='bold')

    ax.set_ylabel(y_label)
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_ylim(0, 110)

    shot_handles = [Patch(facecolor='grey', edgecolor='black',
                          alpha=k_alpha(i, n_k), label=f'{k}-shot')
                    for i, k in enumerate(k_values)]
    ax.legend(handles=shot_handles, fontsize=8, ncol=n_k,
              loc='upper center', bbox_to_anchor=(0.5, -0.18))

    title = resnet_dataset.upper()
    ax.set_title(title, pad=28, fontweight='bold')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.22, top=0.85)

    dataset_dir = vlm_csv.parent.name if vlm_csv is not None else vlm_dir.name
    plot_dir = Path(__file__).resolve().parent / 'plots' / dataset_dir
    plot_dir.mkdir(exist_ok=True, parents=True)
    filename_parts = ['gradient_comparison', resnet_dataset, acc_tag, direction]
    filename_parts.append('k' + '-'.join(str(k) for k in k_values))
    output_path = plot_dir / f'{"_".join(filename_parts)}.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved gradient comparison plot to {output_path}")


def main() -> None:
    """Parse command-line arguments and draw the gradient comparison figure.
    Defaults reproduce the ResNet-vs-SmolVLM comparison on CIFAR-100.

    Returns:
        None: Runs the plotting routine for its side effects.
    """
    parser = argparse.ArgumentParser(description='Gradient-only backbone comparison bar chart')
    parser.add_argument('--resnet-dir', type=Path,
                        default=Path('results/k_shot_evaluation_multirun'),
                        help='Directory with ResNet k-shot evaluation CSVs')
    parser.add_argument('--resnet-dataset', type=str, default='seq-cifar100',
                        help='Dataset name of the ResNet CSVs')
    parser.add_argument('--resnet-label', type=str, default='ResNet-18',
                        help='Display label for the left backbone')
    parser.add_argument('--vlm-dir', type=Path,
                        default=Path('results/vlm_adaptation/seq-cifar100-224'),
                        help='Directory with VLM adaptation results')
    parser.add_argument('--vlm-csv', type=Path, default=None,
                        help='One merged gradient eval CSV holding every method; '
                             'used instead of searching --vlm-dir')
    parser.add_argument('--methods', type=str, default='sgd,agem,ewc-on',
                        help='Comma-separated base method names')
    parser.add_argument('--k-values', type=str, default='0,5,10',
                        help='Comma-separated shot counts to plot')
    parser.add_argument('--accuracy', type=str, choices=list(ACCURACY_COLUMNS),
                        default='cil',
                        help='Which accuracy column to plot; the choice is also '
                             'reflected in the output filename')
    parser.add_argument('--direction', type=str,
                        choices=['all', 'backward', 'forward'], default='all',
                        help='Checkpoint/task cells to average over')
    args = parser.parse_args()

    plot_gradient_comparison(
        resnet_dir=args.resnet_dir,
        resnet_dataset=args.resnet_dataset,
        vlm_dir=args.vlm_dir,
        methods=[m.strip() for m in args.methods.split(',') if m.strip()],
        k_values=[int(k) for k in args.k_values.split(',')],
        accuracy=args.accuracy,
        direction=args.direction,
        resnet_label=args.resnet_label,
        vlm_csv=args.vlm_csv,
    )


if __name__ == '__main__':
    main()
