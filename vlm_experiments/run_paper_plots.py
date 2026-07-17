#!/usr/bin/env python3
# Copyright 2026-present, Amogh Inamdar, Vici Milenia, Richard Zemel.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Thin wrapper: run the repo's results_for_paper.py plot functions on the staged
VLM adaptation CSVs without editing that file. Point its RESULTS_DIR at the
staging dir (symlinks named evaluation_results_<method>_seq-cifar100.csv, one
per model x mode), then call the base-method plot types. Each (model, mode)
shows up as one method bar; the meta plot types are skipped (there are no meta
runs here).

Usage:
    python vlm_experiments/run_paper_plots.py \
        --stage_dir results/vlm_adaptation/paper_stage/seq-cifar100 \
        --dataset seq-cifar100 --k-values 5,10
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import results_for_paper as rfp


def main():
    """Point RESULTS_DIR at the staged VLM CSVs and produce the base-method
    paper figures (stability / forward_transfer / improvement / sauce / table).

    Args: none (CLI).

    Outputs: PNGs + table under the paper script's PLOTS_DIR/<dataset>/.
    """
    p = argparse.ArgumentParser()
    p.add_argument('--stage_dir', required=True, help='dir of staged evaluation_results_*.csv')
    p.add_argument('--dataset', default='seq-cifar100')
    p.add_argument('--k-values', default='5,10')
    args = p.parse_args()

    stage = Path(args.stage_dir)
    assert stage.exists(), f"stage dir missing: {stage}"
    rfp.RESULTS_DIR = stage  # the global every plot function defaults to
    ks = [int(k) for k in args.k_values.split(',')]

    # Results table (mean values only; one seed here, so no standard error).
    rfp.all_results_table(dataset=args.dataset, results_dir=stage, with_meta=False)

    for k in ks:
        rfp.plot_k_shot_stability(dataset=args.dataset, k_values=[k], metric='accuracy',
                                  results_dir=stage, with_meta=False)
        rfp.plot_forward_transfer(dataset=args.dataset, k_values=[k], metric='accuracy',
                                  results_dir=stage, with_meta=False)
        rfp.plot_improvement(dataset=args.dataset, k_values=[k], metric='accuracy',
                             results_dir=stage, with_meta=False)
    rfp.plot_sauce(dataset=args.dataset, results_dir=stage, with_meta=False)
    print("done")


if __name__ == '__main__':
    main()
