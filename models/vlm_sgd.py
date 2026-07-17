# Copyright 2026-present, Amogh Inamdar, Vici Milenia, Richard Zemel.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Version of ``models/sgd.py`` for a vision-language model: plain incremental
fine-tuning with no forgetting countermeasures, but the model is trained to
generate text instead of using a classifier head. ``observe`` trains the model
to produce the correct class name after the prompt; ``self.net(x)`` returns a
[B, N_CLASSES] score per candidate class name, so ``utils.evaluate``,
checkpointing, and the k-shot/SAUCE pipeline treat it like any classifier.
Because the model still works in terms of text tokens, checkpoints stay usable
as text-generating VLMs, which the in-context-vs-gradient adaptation comparison
depends on.

Requires a ``HuggingFaceVLM`` backbone (``--backbone smolvlm-256m`` /
``smolvlm-500m`` / ``qwen2vl-2b`` / ``openflamingo-9b`` / ...); it switches that
backbone into text-generation mode at init.
"""

import torch

from models.utils.continual_model import ContinualModel
from utils import binary_to_boolean_type
from utils.args import ArgumentParser


class VLMSgd(ContinualModel):
    """Incremental generative fine-tuning of a VLM (no forgetting countermeasures)."""
    NAME = 'vlm-sgd'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        """Add the text-generation training options. Backbone options
        (``--vlm_dtype``, ``--vlm_freeze``, ...) come from the backbone
        registry, not from here.

        Args:
            parser: the base experiment parser to extend.

        Returns:
            the parser with the vlm-sgd arguments added.
        """
        # best config from vlm_experiments/quick_sweep_results.csv (3-task
        # seq-cifar100-224): adamw 1e-5 beat 3e-5/1e-4 and momentum-SGD on
        # current, past, and unseen classes.
        parser.set_defaults(optimizer='adamw', lr=1e-5)
        parser.add_argument('--vlm_length_norm', type=binary_to_boolean_type, default=1,
                            help='Score candidates by average log-prob per token (1, avoids '
                                 'favoring short names) or by total log-prob (0). Used only at eval.')
        parser.add_argument('--vlm_score_chunk', type=int, default=16,
                            help='Candidate names scored per forward during evaluation.')
        parser.add_argument('--vlm_clip_grad', type=float, default=1.0,
                            help='Max grad-norm over trainable params; 0 disables.')
        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        """Wrap the registry backbone and switch it into text-generation mode
        with the dataset's class names. ``loss`` is unused: the training loss
        is computed inside the net.

        Args:
            backbone: HuggingFaceVLM from the registry (asserted).
            loss / args / transform / dataset: standard ContinualModel plumbing.

        Outputs: none — configures ``self.net`` in place.
        """
        super().__init__(backbone, loss, args, transform, dataset=dataset)
        assert hasattr(self.net, 'enable_generative'), (
            "vlm-sgd needs a HuggingFaceVLM backbone: use --backbone smolvlm-256m, "
            "smolvlm-500m, qwen2vl-2b, openflamingo-9b, ...")

        names = self.dataset.get_class_names() if hasattr(self.dataset, 'get_class_names') else None
        names = list(names) if names is not None else [str(c) for c in range(self.N_CLASSES)]
        self.net.enable_generative(names, length_norm=args.vlm_length_norm,
                                   score_chunk=args.vlm_score_chunk)

    def _trainables(self):
        """Parameters the optimizer actually updates; the part of the VLM
        frozen by ``--vlm_freeze`` never carries gradients. Shared by the
        vlm-* trainers for gradient clipping, gradient projection, and Fisher
        vectors.

        Args: none.

        Returns:
            generator over the trainable parameters of ``self.net``.
        """
        return (p for p in self.net.parameters() if p.requires_grad)

    def begin_task(self, dataset):
        """Sync the candidate set with the classes seen so far
        (``meta_begin_task`` updates ``n_seen_classes`` before this runs).

        Args:
            dataset: current task's dataset (unused; counters suffice).

        Outputs: none — limits ``self.net`` scoring to seen classes.
        """
        self.net.set_active_classes(self.n_seen_classes)

    def observe(self, inputs, labels, not_aug_inputs, epoch=None, **kwargs):
        """One fine-tuning step: train the model to produce the true class name
        for the batch images. Matches how candidates are scored at eval time.

        Args:
            inputs: augmented, normalized images [B, C, H, W] (the net undoes
                the normalization internally).
            labels: class ids [B].
            not_aug_inputs: unused (the net handles the transformed inputs).
            epoch: unused (observe contract).

        Returns:
            scalar loss value (cross-entropy over the answer tokens only).
        """
        self.opt.zero_grad()
        loss = self.net.lm_loss(inputs, labels)
        loss.backward()
        if self.args.vlm_clip_grad:
            torch.nn.utils.clip_grad_norm_(list(self._trainables()),
                                           self.args.vlm_clip_grad)
        self.opt.step()

        return loss.item()
