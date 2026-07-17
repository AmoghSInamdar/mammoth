# Copyright 2026-present, Amogh Inamdar, Vici Milenia, Richard Zemel.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Ways to adapt a VLM checkpoint at test time using k labeled examples per class,
all sharing one interface (prepare / set_width / scores / explain / _release):

- gradient:      repo k-shot fine-tuning (utils/few_shot.py), text-generation loss.
- prototypical:  Snell et al. 2017 class-mean prototypes on frozen features,
                 over the eval task's own classes (task identity known).
- protonet_cil:  the same prototype head over all classes seen by the checkpoint
                 (0..ckpt_task), matching the CIL menu.
- icl:           labeled example images and answers placed in the prompt, model
                 kept frozen (Brown et al. 2020, Alayrac et al. 2022).
"""

from vlm_experiments.adaptation.gradient import GradientAdapter
from vlm_experiments.adaptation.icl import ICLAdapter
from vlm_experiments.adaptation.prototypical import TaskLocalPrototypicalAdapter
from vlm_experiments.adaptation.prototypical_cil import \
    ClassIncrementalPrototypicalAdapter
from vlm_experiments.adaptation.support import EvalState, build_eval_state

ADAPTATION_MODES = ('gradient', 'prototypical', 'protonet_cil', 'icl')


def make_adapter(mode: str, num_steps: int = 5, lr: float = 0.01,
                 batch_size: int = 32, ckpt_task=None, all_classes: bool = False):
    """Build an adapter instance from a mode name. The gradient settings are
    ignored by modes that use no gradients; ckpt_task and all_classes are only
    used by protonet_cil, where they set the seen class range.

    Args:
        mode: one of ADAPTATION_MODES.
        num_steps: gradient adaptation steps.
        lr: gradient adaptation learning rate.
        batch_size: gradient k-shot loader batch cap.
        ckpt_task: checkpoint's task index (protonet_cil seen class range;
            None falls back to the eval task's own classes).
        all_classes: the model trained on no task, so its prototype bank covers
            every class.

    Returns:
        adapter with the shared prepare/scores/explain interface.
    """
    if mode == 'gradient':
        return GradientAdapter(num_steps=num_steps, lr=lr, batch_size=batch_size)
    if mode == 'prototypical':
        return TaskLocalPrototypicalAdapter()
    if mode == 'protonet_cil':
        return ClassIncrementalPrototypicalAdapter(ckpt_task=ckpt_task,
                                                   all_classes=all_classes)
    if mode == 'icl':
        return ICLAdapter()
    raise ValueError(f"mode must be one of {ADAPTATION_MODES}, got {mode!r}")
