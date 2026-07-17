#!/usr/bin/env python3
# Copyright 2026-present, Amogh Inamdar, Vici Milenia, Richard Zemel.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Score VLM checkpoints under each adaptation mode. One CSV per mode, named
``evaluation_results_<group>_<mode>.csv``.

Grid per mode: checkpoints x eval tasks x k values x seeds. Every cell is
scored under three menus (CIL, TIL, as-learned) and three measures (raw, PMI,
free generation) — see eval/evaluator.py. The accuracy/loss columns hold the
CIL raw numbers; the rest get their own columns.

Where each mode's k shots come from:

- prototypical:  the adapter holds them, so it adapts here.
- protonet_cil:  the adapter holds them, so it adapts here.
- icl:           the prompt holds them, so it adapts here.
- gradient, k>0: the weights hold them already. adapt_gradient.py makes those
                 checkpoints, and their sidecars give the task and k, so
                 --eval_tasks and --k_values are ignored.
- gradient, k=0: no shots, so the checkpoint is scored as given.

--readapt_adapted lets the other modes score a checkpoint whose weights already
hold the shots. They build their head on those same k shots, so the arm still
uses k shots in all.

Usage:
    python vlm_experiments/evaluate_checkpoint.py \
        --checkpoint_paths 'checkpoints/vlm-sgd_seq-cifar100-224_*.pt' \
        --modes gradient,prototypical,icl --k_values 0,1,2,5,10 \
        --seeds 0,1,2 --max_queries 100 --output_dir results/vlm_adaptation

Two prompts at once, each writing its own CSV:
    python vlm_experiments/evaluate_checkpoint.py \
        --checkpoint_paths 'checkpoints/vlm-sgd_seq-cifar100-224_*.pt' \
        --modes gradient,protonet_cil --k_values 0 \
        --vlm_prompts 'default;This is an image of' \
        --output_dir results/vlm_adaptation

Base-model arm: leave out --checkpoint_paths and the pretrained VLM is scored,
built from --dataset, --method, and --backbone. It is written as
``<group>_base``, and since it trained on no task its CIL menu covers all
classes.
"""

import argparse
import glob
import logging
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import setup_logging
from utils.checkpoints import mammoth_load_checkpoint
from utils.conf import get_device

from vlm_experiments.adaptation import (ADAPTATION_MODES, build_eval_state,
                                        make_adapter)
from vlm_experiments.eval import evaluate_adapter
from vlm_experiments.manifest import is_adapted, read_manifest
from vlm_experiments.manifest import subject_id as manifest_subject_id
from vlm_experiments.model_loading import (apply_backbone_override,
                                           build_base_args, group_name_of,
                                           load_vlm_checkpoint,
                                           task_of_checkpoint)
from vlm_experiments.writers import ResultRow, save_results_csv


def parse_args() -> argparse.Namespace:
    """Parse the command line for the adaptation grid. Flag names match
    eval_checkpoints.py where the meaning is the same.

    Args: none (reads sys.argv).

    Returns:
        parsed namespace with expanded checkpoint paths.
    """
    p = argparse.ArgumentParser(description='Inference-time adaptation of VLM checkpoints')
    p.add_argument('--checkpoint_paths', nargs='+', default=None,
                   help='checkpoint files (glob patterns ok), one per trained task. '
                        'Leave this out to score the pretrained model instead, '
                        'built from --dataset, --method, and --backbone')
    p.add_argument('--method', type=str, default='vlm-sgd',
                   help='model class used to build the pretrained model when no '
                        'checkpoint is given. Only sets how adaptation runs; the '
                        'pretrained weights were trained by no method here')
    p.add_argument('--modes', type=str, default='gradient,prototypical,icl',
                   help=f'comma list from {ADAPTATION_MODES}')
    p.add_argument('--readapt_adapted', action='store_true',
                   help='allow modes other than gradient to score checkpoints that '
                        'already carry k shots. The mode builds its head on the same '
                        'k shots the weights were trained on, so the arm still gets '
                        'k shots in total and stays comparable to the others')
    p.add_argument('--dataset', type=str, default=None,
                   help='dataset to run on. Required with no --checkpoint_paths, '
                        'where it is the only thing saying what to build. With a '
                        'checkpoint it overrides the dataset the checkpoint was '
                        'trained on, which is rarely what you want')
    p.add_argument('--eval_tasks', type=str, default='all',
                   help='"all" or comma list of task ids')
    p.add_argument('--k_values', type=str, default='0,1,2,5,10',
                   help='comma list of shots per class')
    p.add_argument('--seeds', type=str, default='0',
                   help='comma list of support-sampling seeds (>1 seed -> multirun '
                        'CSV with mean and standard error aggregation; one seed per '
                        'value lets SLURM run one job per seed)')
    p.add_argument('--num_adapt_steps', type=int, default=5,
                   help='gradient mode: passes over the k-shot loader')
    p.add_argument('--adapt_lr', type=float, default=0.01,
                   help='gradient mode: adaptation learning rate')
    p.add_argument('--adapt_batch_size', type=int, default=32,
                   help='gradient mode: k-shot loader batch cap')
    p.add_argument('--max_queries', type=int, default=0,
                   help='query cap per eval task (0 = full test set). ICL is one '
                        'prompt per query and slow, so pass a small cap (e.g. 100) '
                        'when --modes includes icl')
    p.add_argument('--log_first', type=int, default=5,
                   help='prompts+answers printed per task (first seed only)')
    p.add_argument('--vlm_prompts', type=str, default=None,
                   help='override the checkpoint training prompt at eval time, one '
                        'or more prompts in a semicolon-separated list (e.g. '
                        '"This is an image of" or "default;This is an image of"). '
                        'The literal "default" (or an empty entry) means the '
                        'checkpoint\'s own trained prompt; the default None keeps '
                        'that trained prompt. With more than one prompt each writes '
                        'its own evaluation_results_<group>__<prompt-tag>_<mode>.csv, '
                        'so prompts never clobber each other')
    p.add_argument('--backbone', type=str, default=None,
                   help='backbone to build the pretrained model with (e.g. '
                        '"smolvlm-500m"); it loads its own default base weights. '
                        'Only valid with no --checkpoint_paths, since a checkpoint '
                        'carries the backbone it was trained with')
    p.add_argument('--group_name', type=str, default=None,
                   help='override the output CSV group name (default: the '
                        'checkpoint stem). Set this when --backbone differs from '
                        'the checkpoint so the 500M CSVs never clobber the 256M ones')
    p.add_argument('--output_dir', type=str, default='results/vlm_adaptation',
                   help='where the CSVs land')
    p.add_argument('--file_tag', type=str, default='',
                   help='extra tag added to every output CSV name (e.g. "icl200"), '
                        'so setting sweeps never clobber earlier CSVs')
    args = p.parse_args()

    if args.checkpoint_paths is not None:
        paths = []
        for pattern in args.checkpoint_paths:
            matches = sorted(glob.glob(pattern))
            if not matches:
                logging.warning(f"No files match {pattern}")
            paths.extend(matches)
        assert paths, "no checkpoint files found"
        args.checkpoint_paths = paths
        if args.backbone is not None:
            raise SystemExit('--backbone is only for the pretrained model: a '
                             'checkpoint carries the backbone it was trained with, '
                             'and other weights would not load into it')
    elif not args.dataset:
        raise SystemExit('with no --checkpoint_paths the pretrained model is '
                         'scored, which needs --dataset to say on what data')

    args.modes = [m.strip() for m in args.modes.split(',') if m.strip()]
    for m in args.modes:
        assert m in ADAPTATION_MODES, f"unknown mode {m!r}"
    if (args.checkpoint_paths is not None and args.modes != ['gradient']
            and not args.readapt_adapted
            and any(is_adapted(p) for p in args.checkpoint_paths)):
        raise SystemExit('checkpoints that already carry k shots can only be scored '
                         'with --modes gradient: the shots are already baked into '
                         'the weights, and another mode would adapt a second time '
                         'on top. Pass --readapt_adapted to ask for that on purpose')
    args.k_values = [int(k) for k in args.k_values.split(',')]
    args.seeds = [int(s) for s in args.seeds.split(',')]

    # A None entry means the checkpoint's own trained prompt (no override).
    if args.vlm_prompts is not None:
        prompts = []
        for entry in args.vlm_prompts.split(';'):
            entry = entry.strip()
            prompts.append(None if entry == '' or entry.lower() == 'default' else entry)
        # Drop repeats but keep order: two identical prompts would write to the
        # same CSV name.
        seen = set()
        args.vlm_prompt_sweep = [p for p in prompts if not (p in seen or seen.add(p))]
    else:
        args.vlm_prompt_sweep = [None]
    return args


def prompt_tag(prompt) -> str:
    """Build a short filesystem-safe tag for a prompt, used to keep each
    prompt's CSVs apart. None (the checkpoint's own trained prompt) becomes
    'trainprompt'.

    Args:
        prompt: eval-time prompt string, or None for the trained prompt.

    Returns:
        a lowercase, underscore-joined tag (letters and digits only, <=32 chars).
    """
    if prompt is None:
        return 'trainprompt'
    import re
    tag = re.sub(r'[^0-9a-zA-Z]+', '_', prompt.strip().lower()).strip('_')
    return (tag or 'prompt')[:32]


def menu_spans_all_classes(path, ckpt_task) -> bool:
    """Say whether the menu and the prototype bank should cover every class.
    That is right for a model that never trained on a task, which has seen no
    class more than any other.

    Args:
        path: checkpoint file, or None for the pretrained model.
        ckpt_task: task from task_of_checkpoint.

    Returns:
        True when the menu should cover all classes.
    """
    if path is None:
        return True
    # Without a sidecar, no task means the task is unknown, not that there is none.
    return ckpt_task is None and read_manifest(path) is not None


def adapted_cell_of(path, seeds) -> dict:
    """Read the single cell a checkpoint that already carries k shots stands for
    off its sidecar: the task its shots came from, how many, and what made them.

    Args:
        path: the adapted checkpoint file.
        seeds: the seeds being scored, checked against the seed the shots were
            drawn with.

    Returns:
        the checkpoint's sidecar dict, or None when it carries no shots.
    """
    if path is None or not is_adapted(path):
        return None
    m = read_manifest(path)
    if m['seed'] not in seeds:
        logging.warning(f"{Path(path).stem} drew its shots with seed {m['seed']}, "
                        f"which is not among the seeds being scored ({seeds}). The "
                        f"row will report the seed it is scored under.")
    return m


def main():
    """Run the whole grid: load each checkpoint once, run every requested
    mode/task/k/seed cell on it, write one CSV per mode.

    Args: none (CLI).

    Outputs: CSV files under --output_dir; log lines with the first few prompts.
    """
    setup_logging()
    args = parse_args()
    device = get_device()

    # Build the eval state once from the first checkpoint's args (args only, no
    # model built here).
    pretrained_only = args.checkpoint_paths is None
    base_args = (build_base_args(args.dataset, args.method, args.backbone
                                 or 'smolvlm-256m') if pretrained_only else None)
    ckpt_args = (base_args if pretrained_only
                 else mammoth_load_checkpoint(args.checkpoint_paths[0],
                                              return_only_args=True))
    if args.dataset:
        ckpt_args.dataset = args.dataset
    ckpt_args.num_workers = 0  # cluster /tmp socket crash guard
    apply_backbone_override(ckpt_args, args.backbone)
    state = build_eval_state(ckpt_args)
    # Only main.py fills this in, so args parsed here still have it as None.
    if getattr(ckpt_args, 'num_classes', None) is None:
        ckpt_args.num_classes = state.n_classes
    eval_tasks = (list(range(state.n_tasks)) if args.eval_tasks == 'all'
                  else [int(t) for t in args.eval_tasks.split(',')])
    group = args.group_name or (f'{ckpt_args.dataset}_{ckpt_args.backbone}'
                                if pretrained_only
                                else group_name_of(args.checkpoint_paths))

    if pretrained_only:
        # No file to loop over: one pass builds the pretrained model directly.
        group = f'{group}_base'
        args.checkpoint_paths = [None]

    extra_keys = ('accuracy_til', 'loss_til',
                  'accuracy_as_learned', 'loss_as_learned',
                  'accuracy_pmi', 'loss_pmi',
                  'accuracy_til_pmi', 'loss_til_pmi',
                  'accuracy_as_learned_pmi', 'loss_as_learned_pmi',
                  'raw_pmi_gap', 'til_raw_pmi_gap', 'as_learned_raw_pmi_gap',
                  'accuracy_gen', 'accuracy_til_gen', 'accuracy_as_learned_gen',
                  'gen_unparsed_frac')

    sweep = args.vlm_prompt_sweep
    multi_prompt = len(sweep) > 1
    for vlm_prompt in sweep:
        pgroup = f'{group}__{prompt_tag(vlm_prompt)}' if multi_prompt else group
        if args.file_tag:
            pgroup = f'{pgroup}__{args.file_tag}'
        if multi_prompt:
            logging.info(f"===== prompt {vlm_prompt!r} -> group {pgroup} =====")

        rows = {m: [] for m in args.modes}
        for path in args.checkpoint_paths:
            model, model_args, ckpt_id = load_vlm_checkpoint(
                path, device, dataset_name=args.dataset, vlm_prompt=vlm_prompt,
                backbone=args.backbone, base_args=base_args)
            if path is None:
                ckpt_id = group
            ckpt_task = task_of_checkpoint(path, ckpt_id)
            all_classes = menu_spans_all_classes(path, ckpt_task)

            cell_m = adapted_cell_of(path, args.seeds)
            if cell_m is not None:
                ckpt_id = manifest_subject_id(cell_m)
                cells = [(cell_m['task'], cell_m['k'])]
            else:
                cells = [(t, k) for t in eval_tasks for k in args.k_values]

            for mode in args.modes:
                adapter = make_adapter(mode, num_steps=args.num_adapt_steps,
                                       lr=args.adapt_lr, batch_size=args.adapt_batch_size,
                                       ckpt_task=ckpt_task, all_classes=all_classes)
                max_q = args.max_queries
                for t, k in cells:
                    for seed in args.seeds:
                        n_shots = 0 if (cell_m and mode == 'gradient') else k
                        adapter.prepare(model, state, t, n_shots, seed)
                        tag = f"[{mode}][{ckpt_id}][k={k}][seed={seed}]"
                        res = evaluate_adapter(
                            adapter, state, t, ckpt_task=ckpt_task,
                            max_queries=max_q, query_seed=seed,
                            log_first=args.log_first if seed == args.seeds[0] else 0,
                            log_tag=tag, all_classes=all_classes)
                        hp = cell_m['hyperparams'] if cell_m else None
                        adapted_here = mode == 'gradient' and k > 0 and not cell_m
                        rows[mode].append(ResultRow(
                            seed=seed, checkpoint_id=ckpt_id, eval_task_id=t,
                            k_value=k, accuracy=res['accuracy'], loss=res['loss'],
                            num_adapt_steps=(hp['steps'] if hp else
                                             args.num_adapt_steps if adapted_here else 0),
                            adapt_lr=(hp['lr'] if hp else
                                      args.adapt_lr if adapted_here else None),
                            # An adapted gradient cell prepares no shots of its
                            # own, so only its weights know what they consumed.
                            num_examples_used=(k * (state.offsets[t][1] - state.offsets[t][0])
                                               if (cell_m and mode == 'gradient')
                                               else adapter.n_support),
                            extra={key: res[key] for key in extra_keys},
                            per_class_acc=res['per_class']))
                        adapter._release()
            del model
            torch.cuda.empty_cache()

        for mode in args.modes:
            save_results_csv(rows[mode], args.output_dir, f'{pgroup}_{mode}')

    logging.info("Adaptation evaluation done")


if __name__ == '__main__':
    main()
