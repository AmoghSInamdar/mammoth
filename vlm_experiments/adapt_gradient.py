#!/usr/bin/env python3
# Copyright 2026-present, Amogh Inamdar, Vici Milenia, Richard Zemel.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Train k labeled examples per class into a VLM checkpoint and save the result,
one checkpoint per (ckpt, task, k, seed) cell with a JSON sidecar saying what
made it. Nothing is scored here; evaluate_checkpoint.py reads these and scores
them.

Gradient is the only mode whose k shots land in weights, so it is the only one
here. The others (prototypical, protonet_cil, icl) keep their shots in the
adapter, leaving nothing to save, and adapt at scoring time instead.

Grid: checkpoints x tasks x k values x seeds. k=0 has no shots, so it saves
nothing.

Usage:
    python vlm_experiments/adapt_gradient.py \
        --checkpoint_paths 'checkpoints/vlm-sgd_seq-cifar100-224_*.pt' \
        --k_values 1,2,5,10 --seeds 0 --adapt_lr 1e-5 --num_adapt_steps 5 \
        --save_adapted_dir checkpoints/grad_adapted/seq-cifar100-224_vlm-sgd

Base-model arm: leave out --checkpoint_paths and the k shots are trained into
the pretrained VLM, built from --dataset, --method, and --backbone.
"""

import argparse
import glob
import logging
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import setup_logging
from utils.checkpoints import mammoth_load_checkpoint
from utils.conf import get_device

from vlm_experiments.adaptation import build_eval_state, make_adapter
from vlm_experiments.manifest import (adaptation_hyperparams, adapted_manifest,
                                      base_manifest, checkpoint_name,
                                      finetuned_manifest, read_manifest,
                                      write_manifest)
from vlm_experiments.model_loading import (apply_backbone_override,
                                           build_base_args, load_vlm_checkpoint,
                                           task_of_checkpoint)


def parse_args() -> argparse.Namespace:
    """Parse the command line for the adaptation grid. Flag names match
    evaluate_checkpoint.py where the meaning is the same.

    Args: none (reads sys.argv).

    Returns:
        parsed namespace with expanded checkpoint paths.
    """
    p = argparse.ArgumentParser(description='k-shot gradient adaptation of VLM checkpoints')
    p.add_argument('--checkpoint_paths', nargs='+', default=None,
                   help='checkpoint files (glob patterns ok), one per trained task. '
                        'Leave this out to adapt the pretrained model instead, '
                        'built from --dataset, --method, and --backbone')
    p.add_argument('--method', type=str, default='vlm-sgd',
                   help='model class used to build the pretrained model when no '
                        'checkpoint is given. Only sets how adaptation runs; the '
                        'pretrained weights were trained by no method here')
    p.add_argument('--dataset', type=str, default=None,
                   help='dataset to run on. Required with no --checkpoint_paths, '
                        'where it is the only thing saying what to build. With a '
                        'checkpoint it overrides the dataset the checkpoint was '
                        'trained on, which is rarely what you want')
    p.add_argument('--eval_tasks', type=str, default='all',
                   help='"all" or comma list of task ids to adapt on')
    p.add_argument('--k_values', type=str, default='0,1,2,5,10',
                   help='comma list of shots per class (k=0 saves nothing)')
    p.add_argument('--seeds', type=str, default='0',
                   help='comma list of support-sampling seeds (one seed per value '
                        'lets SLURM run one job per seed)')
    p.add_argument('--num_adapt_steps', type=int, default=5,
                   help='passes over the k-shot loader')
    p.add_argument('--adapt_lr', type=float, default=0.01,
                   help='adaptation learning rate')
    p.add_argument('--adapt_batch_size', type=int, default=32,
                   help='k-shot loader batch cap')
    p.add_argument('--vlm_prompts', type=str, default=None,
                   help='override the checkpoint training prompt while adapting, '
                        'one or more prompts in a semicolon-separated list (e.g. '
                        '"This is an image of" or "default;This is an image of"). '
                        'The literal "default" (or an empty entry) means the '
                        'checkpoint\'s own trained prompt; the default None keeps '
                        'that trained prompt')
    p.add_argument('--backbone', type=str, default=None,
                   help='backbone to build the pretrained model with (e.g. '
                        '"smolvlm-500m"); it loads its own default base weights. '
                        'Only valid with no --checkpoint_paths, since a checkpoint '
                        'carries the backbone it was trained with')
    p.add_argument('--save_adapted_dir', type=str, required=True,
                   help='where the adapted checkpoints and their sidecars land, '
                        'one .pt per (ckpt, task, k, seed) cell')
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
                         'adapted, which needs --dataset to say on what data')

    args.k_values = [int(k) for k in args.k_values.split(',')]
    args.seeds = [int(s) for s in args.seeds.split(',')]

    # A None entry means the checkpoint's own trained prompt (no override).
    if args.vlm_prompts is not None:
        prompts = []
        for entry in args.vlm_prompts.split(';'):
            entry = entry.strip()
            prompts.append(None if entry == '' or entry.lower() == 'default' else entry)
        seen = set()
        args.vlm_prompt_sweep = [p for p in prompts if not (p in seen or seen.add(p))]
    else:
        args.vlm_prompt_sweep = [None]
    return args


def save_adapted_checkpoint(model, save_dir: str, name: str) -> str:
    """Save a gradient-adapted model as a safe-format Mammoth checkpoint, using
    the same layout the training checkpoints use, buffer included for the
    methods that have one. The optimizer and scheduler are dropped, and a
    complete existing file is kept.

    Args:
        model: the adapted ContinualModel to save.
        save_dir: directory the .pt lands in (created if absent).
        name: filename stem, from manifest.checkpoint_name.

    Returns:
        the absolute path the checkpoint was written to.
    """
    from utils import to_parsable_obj
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.abspath(os.path.join(save_dir, name + '.pt'))
    if os.path.exists(path):
        try:
            torch.load(path, map_location='cpu', weights_only=True)
            logging.info(f"[save_adapted] exists, skip: {path}")
            return path
        except Exception:
            logging.warning(f"[save_adapted] unreadable, rewriting: {path}")
    save_obj = {
        'model': {kk: vv.detach().cpu() for kk, vv in model.state_dict().items()},
        'optimizer': None,
        'scheduler': None,
        'args': to_parsable_obj(vars(model.args)),
        'results': None,
    }
    if 'buffer_size' in model.args:
        save_obj['buffer'] = model.buffer.serialize()
    tmp = f'{path}.tmp.{os.getpid()}'  # atomic replace: no truncated .pt on crash
    torch.save(save_obj, tmp)
    os.replace(tmp, path)
    logging.info(f"[save_adapted] wrote {path}")
    return path


def base_manifest_of(path, ckpt_args, task) -> dict:
    """Describe the model an adaptation starts from. The pretrained model needs
    no file; a checkpoint uses its own sidecar, or its args when it has none.

    Args:
        path: checkpoint being adapted, or None for the pretrained model.
        ckpt_args: args the model was built with.
        task: the checkpoint's task, from task_of_checkpoint.

    Returns:
        the base sidecar dict.
    """
    if path is None:
        return base_manifest(ckpt_args.vlm_model_id, ckpt_args.backbone)
    m = read_manifest(path) or finetuned_manifest(ckpt_args, task)
    m['checkpoint'] = checkpoint_name(m)
    return m


def main():
    """Run the whole grid: load each checkpoint once, adapt it on every
    task/k/seed cell, and save each adapted model with its sidecar.

    Args: none (CLI).

    Outputs: one .pt plus one .json per cell under --save_adapted_dir.
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

    if pretrained_only:
        # No file to loop over: one pass builds the pretrained model directly.
        args.checkpoint_paths = [None]

    for vlm_prompt in args.vlm_prompt_sweep:

        def save_cell(adapted_model, base_m, t, k, seed):
            """Write one adapted cell's checkpoint and its sidecar.

            Args:
                adapted_model: the adapted ContinualModel for this cell.
                base_m: sidecar of the model this adapted.
                t: task adapted on.
                k: shots per class.
                seed: support-sampling seed.

            Outputs: none — writes the .pt and its sidecar.
            """
            hp = adaptation_hyperparams(args.adapt_lr, args.num_adapt_steps,
                                        args.adapt_batch_size,
                                        method=model_args.model,
                                        vlm_prompt=vlm_prompt)
            m = adapted_manifest(base_m, t, k, seed, hp, dataset=ckpt_args.dataset)
            apath = save_adapted_checkpoint(adapted_model, args.save_adapted_dir,
                                            checkpoint_name(m))
            write_manifest(apath, m)

        for path in args.checkpoint_paths:
            model, model_args, ckpt_id = load_vlm_checkpoint(
                path, device, dataset_name=args.dataset, vlm_prompt=vlm_prompt,
                backbone=args.backbone, base_args=base_args)
            ckpt_task = task_of_checkpoint(path, ckpt_id)
            base_m = base_manifest_of(path, model_args, ckpt_task)

            adapter = make_adapter('gradient', num_steps=args.num_adapt_steps,
                                   lr=args.adapt_lr, batch_size=args.adapt_batch_size)
            for t in eval_tasks:
                for k in args.k_values:
                    for seed in args.seeds:
                        adapter.prepare(model, state, t, k, seed)
                        if k > 0 and adapter.model is not None:
                            save_cell(adapter.model, base_m, t, k, seed)
                        adapter._release()
            del model
            torch.cuda.empty_cache()

    logging.info("Gradient adaptation done")


if __name__ == '__main__':
    main()
