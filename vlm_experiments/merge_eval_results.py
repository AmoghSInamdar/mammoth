#!/usr/bin/env python3
# Copyright 2026-present, Amogh Inamdar, Vici Milenia, Richard Zemel.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Merge the many split-out eval parts into the final CSVs (the CPU dependency job
of submit_eval.sh). Each SLURM part job wrote its own
``parts/<tag>/evaluation_results_<ckpt-stem>_<mode>.csv``; this joins them per
mode and writes the merged raw + aggregated (+ SAUCE) accuracy CSVs.

Read-only over parts/: never delete or rewrite the part CSVs. They are the
source of truth, so re-merges give the same result and later added seeds still
work. Keep it that way.

Usage:
    python vlm_experiments/merge_eval_results.py \
        --parts_dir results/vlm_adaptation/seq-cifar100-224/parts \
        --output_dir results/vlm_adaptation/seq-cifar100-224
"""

import argparse
import glob
import logging
import os
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import setup_logging
from vlm_experiments.adaptation import ADAPTATION_MODES
from vlm_experiments.writers import save_dataframe_csv


def parse_args() -> argparse.Namespace:
    """CLI for the merge step.

    Args: none (reads sys.argv).

    Returns:
        parsed namespace.
    """
    p = argparse.ArgumentParser(description='Merge the split-out adaptation eval parts')
    p.add_argument('--parts_dir', type=str, required=True,
                   help='directory holding the per-job part subdirs')
    p.add_argument('--output_dir', type=str, required=True,
                   help='where merged CSVs land')
    p.add_argument('--modes', type=str, default=','.join(ADAPTATION_MODES),
                   help='comma list of modes to merge')
    p.add_argument('--file_tag', type=str, default='',
                   help='extra tag added to merged CSV names before the mode '
                        'suffix (match the runner --file_tag, e.g. "icl200")')
    p.add_argument('--split_base', action='store_true',
                   help='write the base (before-adaptation) rows and the '
                        'grad-adapted rows to separate CSVs per mode')
    return p.parse_args()


def group_name_from(checkpoint_ids) -> str:
    """Build the group tag from the merged checkpoint ids: the shared prefix
    with the trailing task index removed (same convention as the runner).

    Args:
        checkpoint_ids: iterable of checkpoint stems.

    Returns:
        group name string.
    """
    stems = sorted(set(checkpoint_ids))
    prefix = os.path.commonprefix(stems) if len(stems) > 1 else stems[0]
    return re.sub(r'_\d+$', '', prefix).rstrip('_-')


def merge_mode(parts_dir: Path, mode: str, output_dir: Path,
               file_tag: str = '', split_base: bool = False) -> None:
    """Merge one mode: join every part CSV and write the merged raw +
    aggregated (+ SAUCE) accuracy CSVs.

    Args:
        parts_dir: directory with per-job part subdirs.
        mode: adaptation mode name (file suffix).
        output_dir: merged CSV destination.
        file_tag: optional tag added to output names before the mode suffix
            (keeps rerun CSVs apart from earlier merges).
        split_base: write base rows and grad-adapted rows to separate CSVs.

    Outputs: none — writes evaluation_results_<group>_<mode>.csv under
        output_dir.
    """
    files = sorted(glob.glob(str(parts_dir / '*' / f'evaluation_results_*_{mode}.csv')))
    if not files:
        logging.warning(f"[{mode}] no part CSVs under {parts_dir}; skipping")
        return
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    # A requeued part can re-run and produce the exact same cell twice; keep the
    # last one.
    df = df.drop_duplicates(subset=['seed', 'checkpoint_id', 'eval_task_id', 'k_value'],
                            keep='last')
    tag = f'__{file_tag}' if file_tag else ''

    if split_base:
        adapted = df['checkpoint_id'].str.contains('grad_adapted')
        # Same filename in separate base/ and adapted/ subdirs.
        stem = group_name_from(df['checkpoint_id'])
        parts = [(output_dir / 'base', df[~adapted]),
                 (output_dir / 'adapted', df[adapted])]
    else:
        stem = None
        parts = [(output_dir, df)]

    for out_sub, part_df in parts:
        if part_df.empty:
            continue
        base = stem if stem is not None else group_name_from(part_df['checkpoint_id'])
        group = f"{base}{tag}_{mode}"
        logging.info(f"[{mode}] merged {len(files)} parts -> {len(part_df)} rows ({out_sub}/{group})")
        save_dataframe_csv(part_df, out_sub, group)


def main():
    """Merge every requested mode and report what landed where.

    Args: none (CLI).

    Outputs: merged CSVs under --output_dir; log lines.
    """
    setup_logging()
    args = parse_args()
    for mode in [m.strip() for m in args.modes.split(',') if m.strip()]:
        merge_mode(Path(args.parts_dir), mode, Path(args.output_dir),
                   file_tag=args.file_tag, split_base=args.split_base)
    logging.info("Merge done")


if __name__ == '__main__':
    main()
