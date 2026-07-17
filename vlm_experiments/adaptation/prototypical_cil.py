# Copyright 2026-present, Amogh Inamdar, Vici Milenia, Richard Zemel.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Class-incremental prototypical network mode (Snell et al. 2017).

The shared prototype head lives in ``prototypical_base.py``; this mode only fixes
the support pool. Prototypes are built for every class the checkpoint has seen
(tasks 0..ckpt_task), so a query competes against every seen class at once. This
matches the evaluator's CIL menu ([0, class_hi(ckpt_task))): with the task-local
head (``TaskLocalPrototypicalAdapter``) that menu would only have prototypes for
the eval task's classes, whereas here it covers all seen classes.

Eval tasks the checkpoint has not reached (task > ckpt_task) get no prototype,
so their true label is not among the candidates. That is the honest measure of
how well the model handles classes it was never told about. When ``ckpt_task``
is None (base model, no checkpoint task), the seen range falls back to the eval
task's own classes, matching the evaluator's base-model fallback.
"""

from vlm_experiments.adaptation.prototypical_base import PrototypicalBase
from vlm_experiments.adaptation.support import (EvalState, sample_k_shot,
                                                sample_k_shot_cil,
                                                sample_k_shot_range)


class ClassIncrementalPrototypicalAdapter(PrototypicalBase):
    """Snell et al. 2017 prototype head over all classes seen by the
    checkpoint."""

    name = 'protonet_cil'

    def __init__(self, ckpt_task=None, all_classes: bool = False,
                 embed_batch: int = 32):
        """Store the checkpoint's task index, which sets the seen class range, on
        top of the base embedding-batch setup.

        Args:
            ckpt_task: last task the checkpoint trained on. With all_classes
                False, None means the checkpoint's task is unknown, so it falls
                back to the eval task's own classes.
            all_classes: the model trained on no task, so it has seen no class
                more than any other and the support pool covers every class.
                Overrides ckpt_task.
            embed_batch: support and query images per feature forward.

        Outputs: none.
        """
        super().__init__(embed_batch=embed_batch)
        self.ckpt_task = ckpt_task
        self.all_classes = all_classes

    def _select_support(self, state: EvalState, task_id: int, k: int, seed: int):
        """Support is k examples per class of every seen class (0..ckpt_task).
        ``task_id`` is ignored for the support pool, since the pool is set by the
        checkpoint and not the eval task, except in the all-classes and
        unknown-task fallbacks.

        Args:
            state: EvalState with raw train pool and TEST transform.
            task_id: eval task (only used when all_classes or ckpt_task is None).
            k: shots per class.
            seed: support sampling seed.

        Returns:
            tuple (span, imgs, labels): span is the seen class range.
        """
        if self.all_classes:
            # Pairs with the full-width menu the evaluator uses for it.
            imgs, labels = sample_k_shot_range(state, 0, state.n_classes, k, seed)
            return (0, state.n_classes), imgs, labels
        if self.ckpt_task is None:
            # Unknown checkpoint task: fall back to the eval task's own classes.
            imgs, labels = sample_k_shot(state, task_id, k, seed)
            return state.offsets[task_id], imgs, labels
        imgs, labels = sample_k_shot_cil(state, self.ckpt_task, k, seed)
        return (0, state.offsets[self.ckpt_task][1]), imgs, labels
