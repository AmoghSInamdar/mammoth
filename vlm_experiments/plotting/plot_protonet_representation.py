#!/usr/bin/env python3
"""
ProtoNet-under-gradient-adaptation bar chart for the paper.

Shows how gradient adaptation moves the representation, read out by ProtoNet:
the x-axis is the gradient-adaptation shot count (gradeval-k), and for each
method two series are drawn per k -- ProtoNet (from the prototypical CSV) and
ProtoNet CIL (from the protonet_cil CSV), both the open-menu accuracy. One
figure per dataset. The per-method colours match plot_gradient_comparison.
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import numpy as np

from plot_utils import (
    ACCURACY_COLUMNS,
    aggregate_grad_adapted,
    aggregate_metric,
    find_grad_adapted_csvs,
    find_merged_eval_csv,
    find_readapt_csv,
    find_unadapted_csv,
    k_alpha,
    load_grad_adapted,
    load_readapt,
    load_with_checkpoint_num,
    readapt_backbone,
    select_method_rows,
)
from results_for_paper import LABEL_MAP, get_method_color

PROTO_SERIES = {
    'protonet': ('ProtoNet', 'prototypical', None),
    'protonet_cil': ('ProtoNet CIL', 'protonet_cil', '///'),
}


def plot_protonet_representation(
    vlm_dir: Path,
    dataset: str,
    methods: List[str],
    k_values: List[int],
    direction: str = 'all',
    backbone: str = 'smolvlm-256m',
    proto_k: Optional[List[int]] = None,
    unadapted_dir: Optional[Path] = None,
    readapt_dirs: Optional[List[Path]] = None,
    accuracy: str = 'cil',
    eval_dir: Optional[Path] = None,
) -> None:
    """Draw and save the ProtoNet-vs-gradeval-k bar chart for one dataset. Each
    group contributes a ProtoNet and a ProtoNet CIL bar at every gradient-
    adaptation shot count, shaded darker as k grows.

    Args:
        vlm_dir: Directory with VLM adaptation results (one dataset).
        dataset: Dataset name for the title and filename (e.g. 'seq-cifar100').
        methods: Base method names (e.g. ['sgd']); ignored with readapt_dirs.
        k_values: Gradient-adaptation shot counts to plot (e.g. [0, 1, 2, 5, 10]).
        direction: Grid cells to average: 'all', 'backward' or 'forward' (default: 'all').
        backbone: Backbone token to plot (e.g. 'smolvlm-256m', 'smolvlm-500m')
            (default: 'smolvlm-256m').
        proto_k: ProtoNet support sizes to average over; None keeps every
            support size present (default: None).
        unadapted_dir: Directory with protonet results evaluated without
            gradient adaptation, used for the k=0 bars; None omits them
            (default: None).
        readapt_dirs: Dataset directories of base-model readapt runs, one per
            backbone. When given, the bars are grouped by backbone instead of
            method (default: None).
        accuracy: Accuracy variant key from ACCURACY_COLUMNS (default: 'cil').
        eval_dir: Dataset directory of an eval run whose merged CSVs hold every
            method's checkpoints, with k in the k_value column. When given, the
            bars are grouped by method (default: None).

    Returns:
        None: The figure is written to plotting/plots/<dataset>/.
    """
    column, acc_tag, y_label = ACCURACY_COLUMNS[accuracy]
    loaded: Dict[str, Dict[str, "pd.DataFrame"]] = {}
    if eval_dir:
        for series, (_, mode, _) in PROTO_SERIES.items():
            csv_path = find_merged_eval_csv(eval_dir, mode)
            if csv_path is None:
                print(f"Warning: no merged eval CSV for {mode}, skipping")
                continue
            merged = load_with_checkpoint_num(csv_path)
            for method in methods:
                rows = select_method_rows(merged, f'vlm-{method.replace("_", "-")}')
                if rows.empty:
                    print(f"Warning: no {method} rows in {csv_path.name}, skipping")
                    continue
                loaded.setdefault(method, {})[series] = rows
            print(f"Loaded {mode}: {csv_path}")
    elif readapt_dirs:
        for readapt_dir in readapt_dirs:
            for series, (_, mode, _) in PROTO_SERIES.items():
                csv_path = find_readapt_csv(readapt_dir, mode)
                if csv_path is None:
                    print(f"Warning: no readapt CSV for {readapt_dir}/{mode}, skipping")
                    continue
                label = readapt_backbone(csv_path)
                loaded.setdefault(label, {})[series] = load_readapt(csv_path)
                print(f"Loaded {label}/{mode}: {csv_path}")
    else:
        for method in methods:
            vlm_method = f'vlm-{method.replace("_", "-")}'
            for series, (_, mode, _) in PROTO_SERIES.items():
                csvs = find_grad_adapted_csvs(vlm_dir, vlm_method, mode, backbone)
                if unadapted_dir is not None:
                    unadapted = find_unadapted_csv(unadapted_dir, vlm_method, mode,
                                                   backbone)
                    if unadapted is None:
                        print(f"Warning: no unadapted (k=0) CSV for "
                              f"{vlm_method}/{mode}; its k=0 bar will be empty")
                    else:
                        csvs = csvs + [unadapted]
                if not csvs:
                    print(f"Warning: no CSVs for {vlm_method}/{mode}, skipping")
                    continue
                loaded.setdefault(method, {})[series] = load_grad_adapted(csvs)
                print(f"Loaded {vlm_method}/{mode}: {len(csvs)} CSV(s)")

    if not loaded:
        print("No results to plot.")
        return

    fig, ax = plt.subplots(figsize=(12, 3.5))

    bar_width = 0.6
    series_gap = 0.8
    method_gap = 1.6
    k_gap = 2.4

    n_series = len(PROTO_SERIES)
    plotted_methods = ([m for m in methods if m in loaded] if not readapt_dirs
                       else list(loaded))
    k_positions: List[float] = []
    k_labels: List[str] = []

    x_pos = 0.0
    for k_idx, k in enumerate(k_values):
        block_start = x_pos
        for method in plotted_methods:
            color = get_method_color(method)
            for series, (_, _, hatch) in PROTO_SERIES.items():
                if series not in loaded[method]:
                    x_pos += series_gap
                    continue
                if eval_dir:
                    val, ste = aggregate_metric(loaded[method][series], k,
                                                column, direction)
                else:
                    val, ste = aggregate_grad_adapted(loaded[method][series], k,
                                                      column, direction, proto_k)
                yerr = ste if np.isfinite(ste) else None
                ax.bar(x_pos, val, width=bar_width, color=color,
                       alpha=k_alpha(k_idx, len(k_values)), hatch=hatch,
                       edgecolor='black', linewidth=0.6,
                       yerr=yerr, ecolor='black', capsize=2)
                x_pos += series_gap
            x_pos += method_gap - series_gap
        x_pos -= method_gap - series_gap
        k_positions.append((block_start + x_pos - series_gap) / 2.0)
        k_labels.append(f'k={k}')
        x_pos += k_gap

    ax.set_xticks(k_positions)
    ax.set_xticklabels(k_labels, fontsize=9)
    ax.set_xlim(-1.0, x_pos - k_gap + 1.0)

    ax.set_ylabel(f'ProtoNet {y_label.lower()}')
    ax.set_xlabel('Gradient-adaptation shots (k)')
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_ylim(0, 110)

    series_handles = [Patch(facecolor='grey', edgecolor='black',
                            hatch=hatch or '', label=label)
                      for label, _, hatch in PROTO_SERIES.values()]
    series_legend = ax.legend(handles=series_handles, fontsize=8, ncol=n_series,
                              loc='upper left', bbox_to_anchor=(-0.01, -0.28))
    ax.add_artist(series_legend)

    if len(plotted_methods) > 1:
        method_handles = [Patch(facecolor=get_method_color(m), edgecolor='black',
                                label=m if readapt_dirs
                                else LABEL_MAP.get(m, m.upper()))
                          for m in plotted_methods]
        ax.legend(handles=method_handles, fontsize=8, ncol=len(plotted_methods),
                  loc='upper right', bbox_to_anchor=(1.01, -0.28))

    ax.set_title(dataset.upper(), pad=12, fontweight='bold')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.34)

    plot_dir = Path(__file__).resolve().parent / 'plots' / dataset
    plot_dir.mkdir(exist_ok=True, parents=True)
    filename_parts = ['protonet_representation', dataset,
                      'readapt' if readapt_dirs else backbone, acc_tag, direction]
    if proto_k is not None:
        filename_parts.append('protok' + '-'.join(str(k) for k in proto_k))
    filename_parts.append('k' + '-'.join(str(k) for k in k_values))
    output_path = plot_dir / f'{"_".join(filename_parts)}.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved protonet representation plot to {output_path}")


def main() -> None:
    """Parse command-line arguments and draw the ProtoNet representation figure
    for one dataset. Defaults target the SmolVLM CIFAR-100 adaptation results.

    Returns:
        None: Runs the plotting routine for its side effects.
    """
    parser = argparse.ArgumentParser(
        description='ProtoNet accuracy vs gradient-adaptation shots bar chart')
    parser.add_argument('--vlm-dir', type=Path,
                        default=Path('results/vlm_adaptation/seq-cifar100-224'),
                        help='Directory with VLM adaptation results (one dataset)')
    parser.add_argument('--dataset', type=str, default='seq-cifar100',
                        help='Dataset name for the title and filename')
    parser.add_argument('--methods', type=str, default='sgd',
                        help='Comma-separated base method names')
    parser.add_argument('--k-values', type=str, default='0,1,2,5,10',
                        help='Comma-separated gradient-adaptation shot counts')
    parser.add_argument('--direction', type=str,
                        choices=['all', 'backward', 'forward'], default='all',
                        help='Checkpoint/task cells to average over')
    parser.add_argument('--backbone', type=str, default='smolvlm-256m',
                        help='Backbone token to plot (e.g. smolvlm-256m, smolvlm-500m)')
    parser.add_argument('--proto-k', type=str, default=None,
                        help='Comma-separated ProtoNet support sizes to average over '
                             '(e.g. 1,2,5,10); omit to keep every support size present')
    parser.add_argument('--unadapted-dir', type=Path, default=None,
                        help='Directory with protonet results evaluated without '
                             'gradient adaptation; supplies the k=0 bars')
    parser.add_argument('--accuracy', type=str, choices=list(ACCURACY_COLUMNS),
                        default='cil',
                        help='Which accuracy column to plot; the choice is also '
                             'reflected in the output filename')
    parser.add_argument('--readapt-dir', type=Path, action='append', default=None,
                        dest='readapt_dirs',
                        help='Dataset directory of a base-model readapt run; repeat '
                             'once per backbone to group the bars by backbone')
    parser.add_argument('--eval-dir', type=Path, default=None,
                        help='Dataset directory of an eval run whose merged CSVs '
                             'hold every method; groups the bars by method')
    args = parser.parse_args()

    plot_protonet_representation(
        vlm_dir=args.vlm_dir,
        dataset=args.dataset,
        methods=[m.strip() for m in args.methods.split(',') if m.strip()],
        k_values=[int(k) for k in args.k_values.split(',')],
        direction=args.direction,
        backbone=args.backbone,
        proto_k=([int(k) for k in args.proto_k.split(',')]
                 if args.proto_k else None),
        unadapted_dir=args.unadapted_dir,
        readapt_dirs=args.readapt_dirs,
        eval_dir=args.eval_dir,
        accuracy=args.accuracy,
    )


if __name__ == '__main__':
    main()
