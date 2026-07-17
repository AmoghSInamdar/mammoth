#!/usr/bin/env python3
# Copyright 2026-present, Amogh Inamdar, Vici Milenia, Richard Zemel.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Dump the raw text the model generates for each test query in the grad_base arm,
next to the true label, across k. This uses the exact same grad_base setup —
base pretrained VLM, GradientAdapter with the same lr/steps/seed/query cap — so
the rows here are exactly what the accuracy_gen column was computed from, one
row per query. The raw generation is kept as-is (never turned into <unparsed>).

Columns: checkpoint, eval_task, k, sample_idx, pred (raw generation), pred_class
(the CIL-menu parse, or '' when it matches no class), label, label_id, marked
(how the generation-accuracy metric scores the row: correct / wrong / unparsed).

Usage:
    python vlm_experiments/dump_generations.py \
        --checkpoint_path <base_ckpt>.pt \
        --k_values 0,1,2,5,10 --max_queries 200 \
        --output_csv results/vlm_adaptation/<ds>/gradck/generations.csv
"""

import argparse
import csv
import logging
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import setup_logging
from utils.checkpoints import mammoth_load_checkpoint
from utils.conf import get_device

from vlm_experiments.adaptation import make_adapter
from vlm_experiments.adaptation.scoring import parse_generation
from vlm_experiments.adaptation.support import build_eval_state
from vlm_experiments.eval.evaluator import _task_loader
from vlm_experiments.model_loading import group_name_of, load_vlm_checkpoint


def parse_args() -> argparse.Namespace:
    """Parse the command line for the generation dump. Flags match the
    grad_base grid so the dumped predictions match the grad_base accuracy_gen
    rows.

    Args: none (reads sys.argv).

    Returns:
        parsed namespace.
    """
    p = argparse.ArgumentParser(description='Dump raw per-sample generations for grad_base')
    p.add_argument('--checkpoint_path', required=True,
                   help='a trained checkpoint of the target dataset (supplies args '
                        'and dataset only; the base weights are used)')
    p.add_argument('--k_values', type=str, default='0,1,2,5,10')
    p.add_argument('--seed', type=int, default=0,
                   help='k-shot sampling + query subsample seed (grad_base: 0)')
    p.add_argument('--adapt_lr', type=float, default=1e-5)
    p.add_argument('--num_adapt_steps', type=int, default=5)
    p.add_argument('--adapt_batch_size', type=int, default=32)
    p.add_argument('--max_queries', type=int, default=200,
                   help='queries per task (grad_base: 200; 0 = full test set)')
    p.add_argument('--max_new_tokens', type=int, default=8)
    p.add_argument('--eval_tasks', type=str, default='all')
    p.add_argument('--output_csv', required=True)
    p.add_argument('--device', type=str, default=None)
    args = p.parse_args()
    args.k_values = [int(k) for k in args.k_values.split(',')]
    return args


def main():
    """Load the base VLM once, adapt then generate for every (task, k) cell,
    and write one row per query to the output CSV.

    Args: none (CLI).

    Outputs: the generation CSV at --output_csv.
    """
    setup_logging()
    args = parse_args()
    device = get_device(args.device) if args.device is None else args.device

    ckpt_args = mammoth_load_checkpoint(args.checkpoint_path, return_only_args=True)
    ckpt_args.num_workers = 0
    state = build_eval_state(ckpt_args)
    eval_tasks = (list(range(state.n_tasks)) if args.eval_tasks == 'all'
                  else [int(t) for t in args.eval_tasks.split(',')])
    group = f'{group_name_of([args.checkpoint_path])}_base'

    model, _, _ = load_vlm_checkpoint(args.checkpoint_path, device, load_weights=False)
    adapter = make_adapter('gradient', num_steps=args.num_adapt_steps,
                           lr=args.adapt_lr, batch_size=args.adapt_batch_size,
                           all_classes=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
    with open(args.output_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['checkpoint', 'eval_task', 'k', 'sample_idx',
                    'pred', 'pred_class', 'label', 'label_id', 'marked'])
        for t in eval_tasks:
            cil_menu = list(range(state.n_classes))  # base model: CIL menu is all classes
            for k in args.k_values:
                adapter.prepare(model, state, t, k, args.seed)
                loader = _task_loader(state, t, args.max_queries, args.seed)
                logging.info(f"[dump] task={t} k={k} generating...")
                idx = 0
                for data in loader:
                    inputs = data[0].to(model.device)
                    labels = data[1].to(dtype=torch.long)
                    answers = adapter.generate(inputs, max_new_tokens=args.max_new_tokens)
                    for ans, lab in zip(answers, labels.tolist()):
                        pc = parse_generation(ans, state.class_names, cil_menu)
                        marked = ('unparsed' if pc < 0
                                  else 'correct' if pc == lab else 'wrong')
                        w.writerow([group, t, k, idx, ans,
                                    state.class_names[pc] if pc >= 0 else '',
                                    state.class_names[lab], lab, marked])
                        idx += 1
                f.flush()
                adapter._release()
    logging.info(f"[dump] wrote {args.output_csv}")


if __name__ == '__main__':
    main()
