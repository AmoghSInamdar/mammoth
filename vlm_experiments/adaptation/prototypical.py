# Copyright 2026-present, Amogh Inamdar, Vici Milenia, Richard Zemel.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Task-local prototypical network mode (Snell et al. 2017).

The shared prototype head lives in ``prototypical_base.py``; this mode only fixes
the support pool. Prototypes are built over the eval task's own classes, so a
query competes against just that task's classes -- the model is effectively told
which task the query belongs to. Contrast
``ClassIncrementalPrototypicalAdapter``, which covers every class seen so far.
"""

from vlm_experiments.adaptation.prototypical_base import PrototypicalBase
from vlm_experiments.adaptation.support import EvalState, sample_k_shot


class TaskLocalPrototypicalAdapter(PrototypicalBase):
    """Snell et al. 2017 prototype head over the eval task's own classes, with
    the task identity effectively known."""

    name = 'prototypical'

    def _select_support(self, state: EvalState, task_id: int, k: int, seed: int):
        """Support is k examples per class of the eval task's own classes, so the
        query is scored only against that task's classes.

        Args:
            state: EvalState with raw train pool and TEST transform.
            task_id: eval task supplying support classes.
            k: shots per class.
            seed: support sampling seed.

        Returns:
            tuple (span, imgs, labels): span is the eval task's class range.
        """
        imgs, labels = sample_k_shot(state, task_id, k, seed)
        return state.offsets[task_id], imgs, labels
