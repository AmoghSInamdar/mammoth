# Copyright 2026-present, Amogh Inamdar, Vici Milenia, Richard Zemel.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Version of ``models/agem.py`` for a vision-language model: A-GEM on top of the
vlm-sgd training objective. The task gradient comes from the training loss;
when it points against the gradient on a replay-buffer minibatch (their dot
product is negative), it is adjusted so it no longer increases the buffer loss.

Two differences from ``models/agem.py``, both on purpose: (1) the gradient
vectors cover the TRAINABLE parameters only — under ``--vlm_freeze`` the frozen
part never has grads, and full-model vectors on a 256M/500M VLM would cost
gigabytes on zeros; (2) ``end_task`` reads the loader until the per-task quota
(``buffer_size // N_TASKS``) is filled instead of storing one batch, because
VLM batch sizes are single-digit and one batch would leave the buffer nearly
empty.

Requires a ``HuggingFaceVLM`` backbone (``--backbone smolvlm-256m`` /
``smolvlm-500m``), same as vlm-sgd.
"""

import torch

from models.agem import project
from models.gem import overwrite_grad, store_grad
from models.vlm_sgd import VLMSgd
from utils.args import add_rehearsal_args, ArgumentParser
from utils.buffer import Buffer


class VLMAGem(VLMSgd):
    """A-GEM with the generative VLM objective."""
    NAME = 'vlm-agem'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il']

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        """Extend the vlm-sgd parser with the rehearsal arguments
        (``--buffer_size``, ``--minibatch_size``).

        Args:
            parser: the base experiment parser to extend.

        Returns:
            the parser with the vlm-agem arguments added.
        """
        parser = VLMSgd.get_parser(parser)
        add_rehearsal_args(parser)
        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        """Set up the replay buffer and the two flat scratch vectors for
        gradients, sized over the trainable parameters. ``VLMSgd.__init__``
        switches the backbone into text-generation mode first.

        Args:
            backbone: HuggingFaceVLM from the registry (asserted upstream).
            loss / args / transform / dataset: standard ContinualModel plumbing.

        Outputs: none — sets ``self.buffer``, ``self.grad_dims``,
            ``self.grad_xy``, ``self.grad_er``.
        """
        super().__init__(backbone, loss, args, transform, dataset=dataset)
        self.buffer = Buffer(self.args.buffer_size)
        self.grad_dims = [p.numel() for p in self._trainables()]
        self.grad_xy = torch.zeros(sum(self.grad_dims), device=self.device)
        self.grad_er = torch.zeros(sum(self.grad_dims), device=self.device)

    def end_task(self, dataset):
        """Store this task's quota (``buffer_size // N_TASKS``) of raw images
        and labels, reading the loader until the quota is met. Replay applies
        ``self.transform`` when sampling, matching the other rehearsal models.

        Args:
            dataset: current task's dataset; ``train_loader`` yields
                (aug, label, not-aug) batches.

        Outputs: none — appends to ``self.buffer``.
        """
        quota = max(1, self.args.buffer_size // dataset.N_TASKS)
        xs, ys, n = [], [], 0
        for data in dataset.train_loader:
            ys.append(data[1])
            xs.append(data[2])
            n += data[1].shape[0]
            if n >= quota:
                break
        self.buffer.add_data(examples=torch.cat(xs)[:quota].to(self.device),
                             labels=torch.cat(ys)[:quota].to(self.device))

    def observe(self, inputs, labels, not_aug_inputs, epoch=None, **kwargs):
        """One A-GEM step on the training loss: backward on the current batch
        and, if that gradient points against a buffer-batch gradient, adjust it
        before stepping.

        Args:
            inputs: augmented, normalized images [B, C, H, W].
            labels: class ids [B].
            not_aug_inputs: unused (raw images enter the buffer in end_task).
            epoch: unused (observe contract).

        Returns:
            scalar loss value on the current batch (buffer loss excluded).
        """
        self.opt.zero_grad()
        loss = self.net.lm_loss(inputs, labels)
        loss.backward()

        if not self.buffer.is_empty():
            store_grad(self._trainables, self.grad_xy, self.grad_dims)

            buf_inputs, buf_labels = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform, device=self.device)
            self.net.zero_grad()
            penalty = self.net.lm_loss(buf_inputs, buf_labels)
            penalty.backward()
            store_grad(self._trainables, self.grad_er, self.grad_dims)

            dot_prod = torch.dot(self.grad_xy, self.grad_er)
            if dot_prod.item() < 0:
                g_tilde = project(gxy=self.grad_xy, ger=self.grad_er)
                overwrite_grad(self._trainables, g_tilde, self.grad_dims)
            else:
                overwrite_grad(self._trainables, self.grad_xy, self.grad_dims)

        if self.args.vlm_clip_grad:
            torch.nn.utils.clip_grad_norm_(list(self._trainables()),
                                           self.args.vlm_clip_grad)
        self.opt.step()

        return loss.item()
