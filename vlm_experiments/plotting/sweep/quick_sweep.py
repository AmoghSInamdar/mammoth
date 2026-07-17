# Copyright 2026-present, Amogh Inamdar, Vici Milenia, Richard Zemel.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Quick hyperparameter sweep runner: train the first N tasks with the normal
Mammoth training loop, and after every task eval the past, current, and unseen
tasks. Writes no checkpoints.

Eval uses vlm_experiments/eval/evaluator.evaluate_adapter with a k=0
GradientAdapter (score the live model as-is, no copy, no adaptation steps):
CIL + TIL menus x raw / PMI / generation measures, with the first few
prompt+answer pairs per task in the job log. The menu and measure definitions
live in that evaluator, not here.

Outputs under --sweep_out:
- summary.csv: one row per (after_task, eval_task), holding the evaluate_adapter
  result dict (accuracy = CIL raw primary, *_til, *_pmi, *_gen, gaps, unparsed).

Usage:
    python vlm_experiments/quick_sweep.py --sweep_tasks 3 \
        --sweep_out results/vlm_quick_sweep/<tag> -- <mammoth args...>
"""

import argparse
import csv
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_sweep_argv():
    """Split the runner flags from the mammoth args at '--'. Runner flags come
    before the '--'; everything after it is passed unchanged to the mammoth
    parser.

    Returns:
        (sweep_args, mammoth_argv): parsed runner namespace + argv list.
    """
    argv = sys.argv[1:]
    sep = argv.index('--') if '--' in argv else len(argv)
    p = argparse.ArgumentParser()
    p.add_argument('--sweep_tasks', type=int, default=3,
                   help='train (and eval) first N tasks')
    p.add_argument('--sweep_out', type=str, required=True,
                   help='output dir for summary.csv')
    p.add_argument('--sweep_max_queries', type=int, default=0,
                   help='cap eval queries per task (0 = full test set); debug knob')
    p.add_argument('--sweep_allow_cpu', action='store_true',
                   help='skip cuda assert (local debug only)')
    return p.parse_args(argv[:sep]), argv[sep + 1:]


def main():
    """Run one sweep config: start up mammoth, train task by task, eval every
    task via evaluate_adapter after each one, and write summary.csv. Prints one
    line per (after_task, eval_task) to stdout (the job log).

    Outputs: summary.csv under --sweep_out; exits non-zero if cuda is missing
    (guards against the cluster running on CPU by mistake).
    """
    sweep, margv = parse_sweep_argv()
    sys.argv = ['main.py'] + margv
    from main import initialize
    from vlm_experiments.adaptation import make_adapter
    from vlm_experiments.adaptation.support import build_eval_state
    from vlm_experiments.eval.evaluator import evaluate_adapter

    model, dataset, args = initialize()

    if not sweep.sweep_allow_cpu:
        assert 'cuda' in str(args.device), f"no cuda (got {args.device}); CPU-fallback node?"
    model.net.to(model.device)  # train() does this at utils/training.py:156, and we skip train()

    n_tasks = sweep.sweep_tasks
    os.makedirs(sweep.sweep_out, exist_ok=True)
    state = build_eval_state(args)
    measure_pmi = bool(getattr(model.net, 'generative', False))

    summary = []
    for t in range(n_tasks):
        train_loader, _ = dataset.get_data_loaders()
        model.meta_begin_task(dataset, t)
        model.net.train()
        for epoch in range(args.n_epochs):
            model.meta_begin_epoch(epoch, dataset)
            for i, data in enumerate(train_loader):
                if args.debug_mode and i > model.get_debug_iters():
                    break
                inputs = data[0].to(model.device)
                labels = data[1].to(model.device, dtype=torch.long)
                not_aug = data[2].to(model.device)
                loss = model.meta_observe(inputs, labels, not_aug, epoch=epoch)
            model.meta_end_epoch(epoch, dataset)
            print(f"[task {t}] epoch {epoch} done, last loss {loss:.4f}", flush=True)
        model.meta_end_task(dataset)

        # k=0 adapter scores the live model directly: no copy, no adaptation steps.
        model.net.eval()
        adapter = make_adapter('gradient').prepare(model, state, task_id=0, k=0, seed=0)
        for u in range(n_tasks):
            res = evaluate_adapter(adapter, state, u, ckpt_task=t,
                                   max_queries=sweep.sweep_max_queries,
                                   log_first=5, log_tag=f"[t={t}]",
                                   measure_pmi=measure_pmi, measure_gen=True)
            row = {'after_task': t, 'eval_task': u,
                   **{k: v for k, v in res.items() if k != 'per_class'}}
            summary.append(row)
            print(f"[task {t}] eval_task {u}: {row}", flush=True)
        model.net.train()

    with open(os.path.join(sweep.sweep_out, 'summary.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)
    print(f"[done] wrote {len(summary)} rows -> {sweep.sweep_out}/summary.csv", flush=True)


if __name__ == '__main__':
    main()
