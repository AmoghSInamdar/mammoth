#!/usr/bin/env python3
# Copyright 2026-present, Amogh Inamdar, Vici Milenia, Richard Zemel.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Merge per-task in-context SAUCE CSVs (produced by the per-task SLURM fan-out in
submit_icl_sauce_jobs.sh) into the single file the plotting/analysis expects.

Per-task jobs each write
    <tasks_root>/<tag>/task<t>/evaluation_results_<tag>.csv          (raw, per-seed)
    <tasks_root>/<tag>/task<t>/aggregated/evaluation_results_<tag>.csv (per-seed mean/std)

This script concatenates them into
    <out_dir>/evaluation_results_<tag>.csv                 (raw union)
    <out_dir>/aggregated/evaluation_results_<tag>.csv      (aggregated union + SAUCE)

SAUCE groups by (checkpoint_id, eval_task_id), and each task fan-out produced a
DISJOINT eval_task_id, so the union is just a row concatenation; we recompute the
four plasticity columns on the merged aggregated file so they are present and
consistent. (checkpoint_id is constant per (backend,dataset), so the curve x-axis
is unaffected by the split.)

Usage:
    python merge_icl_task_csvs.py --tag icl-qwen2vl_seq-cifar100 \
        --tasks_root results/k_shot_evaluation_multirun/_tasks \
        --out_dir results/k_shot_evaluation_multirun
"""

import argparse
from pathlib import Path

import pandas as pd

from utils.per_shot_plasticity import add_plasticity_scores_to_csv


def _concat(csvs):
    frames = [pd.read_csv(c) for c in csvs if Path(c).exists()]
    return pd.concat(frames, ignore_index=True) if frames else None


def main():
    ap = argparse.ArgumentParser(description='Merge per-task in-context SAUCE CSVs')
    ap.add_argument('--tag', required=True,
                    help='Group tag, e.g. icl-qwen2vl_seq-cifar100 or icl-llava-anon_rot-mnist')
    ap.add_argument('--tasks_root', required=True,
                    help='Directory containing <tag>/task<t>/ subdirs')
    ap.add_argument('--out_dir', required=True,
                    help='Destination dir for the merged evaluation_results_<tag>.csv')
    ap.add_argument('--metric', default='accuracy', choices=['accuracy', 'loss'])
    args = ap.parse_args()

    tag = args.tag
    fname = f'evaluation_results_{tag}.csv'
    group_dir = Path(args.tasks_root) / tag
    task_dirs = sorted(group_dir.glob('task*'))
    if not task_dirs:
        raise FileNotFoundError(f'No task* subdirs under {group_dir}')

    out_dir = Path(args.out_dir)
    (out_dir / 'aggregated').mkdir(parents=True, exist_ok=True)

    # Raw union (all seeds, all tasks).
    raw = _concat([d / fname for d in task_dirs])
    if raw is not None:
        raw = raw.sort_values(['eval_task_id', 'k_value', 'seed'] if 'seed' in raw.columns
                              else ['eval_task_id', 'k_value'])
        raw.to_csv(out_dir / fname, index=False)
        print(f'Wrote raw union ({len(raw)} rows, {raw["eval_task_id"].nunique()} tasks) '
              f'-> {out_dir / fname}')

    # Aggregated union + recomputed plasticity.
    agg = _concat([d / 'aggregated' / fname for d in task_dirs])
    if agg is None:
        print('No aggregated per-task CSVs found; skipping aggregated merge.')
        return
    agg = agg.sort_values(['eval_task_id', 'k_value'])
    agg_path = out_dir / 'aggregated' / fname
    agg.to_csv(agg_path, index=False)
    add_plasticity_scores_to_csv(agg_path, metric_for_sauce=args.metric)
    print(f'Wrote aggregated union + SAUCE ({len(agg)} rows, '
          f'{agg["eval_task_id"].nunique()} tasks) -> {agg_path}')


if __name__ == '__main__':
    main()
