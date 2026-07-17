# Copyright 2026-present, Amogh Inamdar, Vici Milenia, Richard Zemel.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
The JSON sidecar that describes a checkpoint, and the checkpoint naming built
from it. Every checkpoint is fully self-describing, so nothing has to load a
donor checkpoint just to find out how a model was built.

Each sidecar sits beside its .pt with the same stem and records what the model
came from (``base``), what data and settings made it, and which of the four
groups it belongs to:

- base:                       the pretrained VLM, untrained.
- finetuned:                  trained on a dataset through some task.
- gradient_adapted_base:      the pretrained VLM after k-shot adaptation.
- gradient_adapted_finetuned: a finetuned checkpoint after k-shot adaptation.

``base`` holds the parent's own sidecar, so a chain of builds nests: an adapted
checkpoint's base is the finetuned checkpoint it started from, whose base is the
pretrained model it started from.
"""

import json
import os
from typing import Optional

BASE = 'base'
FINETUNED = 'finetuned'
GRADIENT_ADAPTED_BASE = 'gradient_adapted_base'
GRADIENT_ADAPTED_FINETUNED = 'gradient_adapted_finetuned'

GROUPS = (BASE, FINETUNED, GRADIENT_ADAPTED_BASE, GRADIENT_ADAPTED_FINETUNED)

ADAPTED_GROUPS = (GRADIENT_ADAPTED_BASE, GRADIENT_ADAPTED_FINETUNED)

# '_' separates name fields, so groups use '-' inside a name.
_GROUP_IN_NAME = {g: g.replace('_', '-') for g in GROUPS}

_TRAIN_KEYS = ('lr', 'optimizer', 'optim_wd', 'optim_mom', 'n_epochs',
               'batch_size', 'vlm_prompt', 'vlm_dtype')

_METHOD_KEYS = ('buffer_size', 'minibatch_size', 'e_lambda', 'gamma')


def _as_dict(args) -> dict:
    """Read a checkpoint's args whether they were saved as a Namespace or a
    plain dict. Both layouts exist in this repo's checkpoints.

    Args:
        args: the args object read from a checkpoint.

    Returns:
        a plain dict of the args.
    """
    return args if isinstance(args, dict) else vars(args)


def training_hyperparams(args) -> dict:
    """Pull the training settings out of a checkpoint's args, keeping only the
    keys that are actually present. Method-specific settings are included when
    the method has them.

    Args:
        args: the args object read from a checkpoint.

    Returns:
        a dict of training settings.
    """
    a = _as_dict(args)
    keys = _TRAIN_KEYS + _METHOD_KEYS
    return {k: a[k] for k in keys if k in a and a[k] is not None}


def adaptation_hyperparams(lr: float, steps: int, batch_size: int,
                           method: str = None, vlm_prompt: str = None,
                           inferred=()) -> dict:
    """Collect the settings a k-shot gradient adaptation ran with. Adaptation
    takes a training step through the method's own code, so which method ran it
    changes what the k shots did and belongs here.

    Args:
        lr: adaptation learning rate.
        steps: passes over the k-shot support set.
        batch_size: support loader batch cap.
        method: model class whose training step ran the adaptation. A method
            that replays a buffer replays it here too.
        vlm_prompt: prompt in use during adaptation, if known.
        inferred: names of the settings that were assumed, not recorded.

    Returns:
        a dict of adaptation settings.
    """
    hp = {'lr': lr, 'steps': steps, 'batch_size': batch_size}
    if method is not None:
        hp['method'] = method
    if vlm_prompt is not None:
        hp['vlm_prompt'] = vlm_prompt
    if inferred:
        hp['inferred'] = list(inferred)
    return hp


def base_manifest(vlm_model_id: str, model: str) -> dict:
    """Describe the pretrained VLM the whole chain starts from. It has no
    dataset, method, or task, since it was never trained here.

    Args:
        vlm_model_id: full pretrained model name, e.g.
            'HuggingFaceTB/SmolVLM-256M-Base'.
        model: this repo's backbone name, e.g. 'smolvlm-256m'.

    Returns:
        a sidecar dict for the pretrained model.
    """
    return {'group': BASE, 'source': vlm_model_id, 'model': model}


def finetuned_manifest(args, task: int) -> dict:
    """Describe a checkpoint trained on a dataset through the given task. Its
    base is the pretrained model it started from.

    Args:
        args: the args object read from the checkpoint.
        task: last task the checkpoint trained on.

    Returns:
        a sidecar dict for the finetuned checkpoint.
    """
    a = _as_dict(args)
    return {
        'group': FINETUNED,
        'source': a['vlm_model_id'],
        'dataset': a['dataset'],
        'method': a['model'],
        'model': a['backbone'],
        'task': task,
        'seed': a.get('seed'),
        'hyperparams': training_hyperparams(args),
        'base': base_manifest(a['vlm_model_id'], a['backbone']),
    }


def adapted_manifest(base: dict, task: int, k: int, seed: int,
                     hyperparams: dict, dataset: str = None) -> dict:
    """Describe a checkpoint made by k-shot gradient adaptation on one task.
    The group depends on whether it started from the pretrained model or from a
    finetuned checkpoint.

    Args:
        base: the sidecar of the checkpoint this adapted, as returned by
            base_manifest or finetuned_manifest.
        task: task the adaptation ran on.
        k: shots per class used.
        seed: support sampling seed.
        hyperparams: adaptation settings, from adaptation_hyperparams.
        dataset: dataset the adaptation ran on; taken from base when absent.

    Returns:
        a sidecar dict for the adapted checkpoint.
    """
    from_base = base['group'] == BASE
    m = {
        'group': GRADIENT_ADAPTED_BASE if from_base else GRADIENT_ADAPTED_FINETUNED,
        'source': base['source'],
        'dataset': dataset or base.get('dataset'),
        'model': base['model'],
        'task': task,
        'k': k,
        'seed': seed,
        'hyperparams': hyperparams,
        'base': base,
    }
    if not from_base:
        m['method'] = base['method']
    return m


def checkpoint_name(m: dict) -> str:
    """Build the filename stem for a checkpoint from its sidecar. Fields the
    group does not have are left out, so a pretrained model carries no dataset
    or task and only an adapted checkpoint carries a k.

    Args:
        m: the sidecar dict.

    Returns:
        the filename stem, with no extension.
    """
    group = m['group']
    tag = _GROUP_IN_NAME[group]
    if group == BASE:
        return f"{m['model']}_{tag}"
    if group == FINETUNED:
        # mammoth appends _<task> to the name it is given.
        return f"{m['dataset']}_{m['method']}_{m['model']}_{tag}_{m['task']}"
    if group == GRADIENT_ADAPTED_BASE:
        return f"{m['dataset']}_{m['model']}_{tag}_a{m['task']}_k{m['k']}"
    return (f"{m['dataset']}_{m['method']}_{m['model']}_{tag}"
            f"_t{m['base']['task']}_a{m['task']}_k{m['k']}")


def sidecar_path(checkpoint_path: str) -> str:
    """Give the sidecar path for a checkpoint path. The two always share a stem.

    Args:
        checkpoint_path: path to the .pt file.

    Returns:
        path to the .json file.
    """
    return os.path.splitext(checkpoint_path)[0] + '.json'


def write_manifest(checkpoint_path: str, m: dict) -> str:
    """Write a sidecar beside its checkpoint. The write is atomic, so a crash
    cannot leave a half-written sidecar next to a good checkpoint.

    Args:
        checkpoint_path: path to the .pt the sidecar describes.
        m: the sidecar dict.

    Returns:
        the path the sidecar was written to.
    """
    path = sidecar_path(checkpoint_path)
    tmp = f'{path}.tmp.{os.getpid()}'
    with open(tmp, 'w') as f:
        json.dump(m, f, indent=2, sort_keys=False)
    os.replace(tmp, path)
    return path


def read_manifest(checkpoint_path: str) -> Optional[dict]:
    """Read the sidecar beside a checkpoint, if there is one.

    Args:
        checkpoint_path: path to the .pt file.

    Returns:
        the sidecar dict, or None when no readable sidecar exists.
    """
    path = sidecar_path(checkpoint_path)
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def is_adapted(checkpoint_path: str) -> bool:
    """Say whether a checkpoint already carries k shots, read off its sidecar.
    A checkpoint with no sidecar cannot say it does, so it is taken as not
    adapted.

    Args:
        checkpoint_path: path to the .pt file.

    Returns:
        True when the sidecar puts the checkpoint in one of the adapted groups.
    """
    m = read_manifest(checkpoint_path)
    return m is not None and m['group'] in ADAPTED_GROUPS


def ckpt_task(m: dict) -> Optional[int]:
    """Give the last task the model behind a sidecar trained on, which sets the
    class-incremental menu. Adapting on k shots is not training a task, so an
    adapted checkpoint reports the task its parent trained through, found by
    following ``base``.

    Args:
        m: the sidecar dict.

    Returns:
        the task index, or None when the model trained on no task at all, which
        means every class is equally seen and the menu spans all of them.
    """
    if m['group'] == BASE:
        return None
    if m['group'] in ADAPTED_GROUPS:
        return ckpt_task(m['base'])
    return m['task']


def subject_id(m: dict) -> str:
    """Name the model that was scored, ignoring any k-shot adaptation on top of
    it. An adapted checkpoint reports the checkpoint it adapted from, so its row
    lands on the same CSV key as that checkpoint's other rows and the k values
    line up as one curve.

    Args:
        m: the sidecar dict.

    Returns:
        the name of the model the k shots were applied to.
    """
    if m['group'] in ADAPTED_GROUPS:
        base = dict(m['base'])
        base.setdefault('dataset', m.get('dataset'))  # the pretrained model has none
        return subject_id(base)
    if m['group'] == BASE:
        dataset = m.get('dataset')
        return f"{dataset}_{m['model']}_base" if dataset else f"{m['model']}_base"
    return m.get('checkpoint') or checkpoint_name(m)
