# Copyright 2026-present, Amogh Inamdar, Vici Milenia, Richard Zemel.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Building the model a run works on, and reading what a checkpoint is. Shared by
adapt_gradient.py, which trains k shots into a checkpoint, and
evaluate_checkpoint.py, which scores checkpoints.
"""

import logging
import os
from pathlib import Path

from utils.checkpoints import mammoth_load_checkpoint

from vlm_experiments.eval.metrics import ckpt_task_index
from vlm_experiments.manifest import read_manifest
from vlm_experiments.manifest import ckpt_task as manifest_ckpt_task


def apply_backbone_override(ckpt_args, backbone=None):
    """Set the backbone choice on ckpt_args in place, so the pretrained model can
    be built at a chosen size (the new backbone loads its own default base
    weights).

    Args:
        ckpt_args: the Namespace read from the checkpoint.
        backbone: replacement backbone name (e.g. 'smolvlm-500m'), or None.

    Returns:
        None: ckpt_args is changed in place.
    """
    if backbone is not None:
        ckpt_args.backbone = backbone
        # get_backbone reads vlm_model_id from ckpt_args, so the donor's id must
        # be reset to the new backbone's default or it loads the wrong weights.
        from backbone import get_backbone_class
        _, bb_args = get_backbone_class(backbone, return_args=True)
        if 'vlm_model_id' in bb_args:
            ckpt_args.vlm_model_id = bb_args['vlm_model_id']['default']


def build_base_args(dataset: str, method: str, backbone: str):
    """Build a full set of args for the pretrained model without reading any
    checkpoint. This is what lets a base run stand on its own instead of
    borrowing an unrelated checkpoint just to find out its settings.

    Args:
        dataset: dataset to run on.
        method: model class used to build the model and run adaptation.
        backbone: backbone name, e.g. 'smolvlm-256m'.

    Returns:
        the args namespace, with every default filled in.
    """
    from main import parse_args as mammoth_parse_args
    return mammoth_parse_args(cmd=['--model', method, '--backbone', backbone,
                                   '--dataset', dataset, '--lr', '1e-5',
                                   # must match submit_checkpoints.sh
                                   '--batch_size', '32', '--vlm_dtype', 'bfloat16',
                                   '--num_workers', '0', '--seed', '0'])


def load_vlm_checkpoint(path: str, device, dataset_name=None,
                        vlm_prompt: str = None, backbone: str = None,
                        base_args=None, load_weights: bool = True):
    """Rebuild the training-time model around a checkpoint and load its weights,
    the same steps as eval_checkpoints.load_checkpoint_for_eval. With no
    checkpoint the pretrained VLM is built from base_args and kept as it is.

    Args:
        path: checkpoint file, or None for the pretrained model.
        device: torch device string.
        dataset_name: dataset name override.
        vlm_prompt: eval-time prompt override (None = training prompt).
        backbone: backbone to build the pretrained model with.
        base_args: args to build the model from when there is no checkpoint.
        load_weights: False builds the model around the checkpoint's args but
            keeps the pretrained weights.

    Returns:
        (model, ckpt_args, checkpoint_id).
    """
    from backbone import get_backbone
    from datasets import get_dataset_class
    from models import get_model

    ckpt_args = (base_args if path is None
                 else mammoth_load_checkpoint(path, return_only_args=True))
    if dataset_name:
        ckpt_args.dataset = dataset_name
    ckpt_args.num_workers = 0  # cluster /tmp socket crash guard
    apply_backbone_override(ckpt_args, backbone)
    if vlm_prompt is not None:
        # The prompt is stored in non-saved buffers (prompt_ids/prompt_mask), so
        # the trained weights still load cleanly under the new prompt.
        logging.info(f"Prompt override: {ckpt_args.vlm_prompt!r} -> {vlm_prompt!r}")
        ckpt_args.vlm_prompt = vlm_prompt

    dataset = get_dataset_class(ckpt_args)(ckpt_args)
    backbone = get_backbone(ckpt_args)
    model = get_model(ckpt_args, backbone, dataset.get_loss(),
                      dataset.get_transform(), dataset=dataset)
    if path is not None and load_weights:
        loaded = mammoth_load_checkpoint(path, model=model)
        model = loaded[0] if isinstance(loaded, tuple) else loaded
    model.to(device)
    model.device = device
    model.net.eval()
    return model, ckpt_args, (Path(path).stem if path is not None else 'pretrained')


def task_of_checkpoint(path, ckpt_id: str):
    """Say which task a checkpoint trained on, which sets the class-incremental
    menu. The checkpoint's sidecar answers this; checkpoints written before
    sidecars existed fall back to reading the trailing number off the filename,
    which is what the sidecar replaced.

    Args:
        path: checkpoint file, or None for the pretrained model.
        ckpt_id: checkpoint stem, used only by the fallback.

    Returns:
        the task index, or None when the model trained on no task.
    """
    if path is None:
        return None
    m = read_manifest(path)
    if m is not None:
        return manifest_ckpt_task(m)
    task = ckpt_task_index(ckpt_id)
    logging.warning(f"{ckpt_id} has no sidecar; read task {task} off its name. "
                    f"Run vlm_experiments/migrate_checkpoints.py to give it one.")
    return task


def group_name_of(paths) -> str:
    """Return the shared filename prefix of the checkpoints, used as the CSV
    group tag (same as eval_checkpoints.get_results_group_name).

    Args:
        paths: checkpoint path list.

    Returns:
        group name string.
    """
    stems = [Path(p).stem for p in paths]
    if len(stems) == 1:
        return stems[0]
    prefix = os.path.commonprefix(stems).rstrip('_-')
    return prefix or stems[0]
