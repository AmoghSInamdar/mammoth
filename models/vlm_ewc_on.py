# Copyright 2026-present, Amogh Inamdar, Vici Milenia, Richard Zemel.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Version of ``models/ewc_on.py`` for a vision-language model: online EWC on top
of the vlm-sgd training objective. After each task, a Fisher value per weight
is estimated from the training loss; later tasks pay a squared penalty for
moving the weights the Fisher marks as important, and the running Fisher total
is scaled down by ``--gamma`` each time (online EWC).

Two differences from ``models/ewc_on.py``, both on purpose: (1) the Fisher is
estimated per batch — the squared gradient of each minibatch, averaged over
``--ewc_fisher_batches`` batches — because ewc_on's per-example version would
mean one backward pass per image through the VLM; (2) the Fisher, checkpoint,
and penalty vectors cover the TRAINABLE parameters only, since parts frozen by
``--vlm_freeze`` never get gradients, and full-model vectors on a 256M/500M VLM
would waste gigabytes on zeros. Even so, full fine-tuning keeps two extra
float32 vectors the size of the trainable set on the GPU (~1 GB each for
smolvlm-256m) — budget for it.

Requires a ``HuggingFaceVLM`` backbone (``--backbone smolvlm-256m`` /
``smolvlm-500m``), same as vlm-sgd.
"""

import torch

from models.vlm_sgd import VLMSgd
from utils.args import ArgumentParser


class VLMEwcOn(VLMSgd):
    """Online EWC with the generative VLM objective."""
    NAME = 'vlm-ewc-on'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il']

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        """Extend the vlm-sgd parser with the EWC knobs; ``--e_lambda`` and
        ``--gamma`` stay required, mirroring ewc_on.

        Args:
            parser: the base experiment parser to extend.

        Returns:
            the parser with the vlm-ewc-on arguments added.
        """
        parser = VLMSgd.get_parser(parser)
        parser.add_argument('--e_lambda', type=float, required=True,
                            help='lambda weight for EWC')
        parser.add_argument('--gamma', type=float, required=True,
                            help='gamma parameter for EWC online')
        parser.add_argument('--ewc_fisher_batches', type=int, default=50,
                            help='Train-loader batches used for the Fisher estimate; 0 = full loader.')
        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        """Initialize with an empty Fisher and checkpoint; both stay ``None``
        until the first ``end_task``. ``VLMSgd.__init__`` switches the backbone
        into text-generation mode.

        Args:
            backbone: HuggingFaceVLM from the registry (asserted upstream).
            loss / args / transform / dataset: standard ContinualModel plumbing.

        Outputs: none — sets ``self.checkpoint``, ``self.fish``.
        """
        super().__init__(backbone, loss, args, transform, dataset=dataset)
        self.checkpoint = None
        self.fish = None

    def _params_vector(self):
        """Flatten the current trainable parameters into one float32 vector,
        in ``self._trainables()`` order (shared by Fisher, checkpoint,
        penalty).

        Args: none.

        Returns:
            detached float32 tensor of shape [n_trainable].
        """
        return torch.cat([p.detach().float().reshape(-1) for p in self._trainables()])

    def _grads_vector(self):
        """Flatten the current gradients over trainable parameters; params
        that got no gradient (e.g. the unused classifier head in
        text-generation mode) count as zeros.

        Args: none.

        Returns:
            detached float32 tensor of shape [n_trainable].
        """
        return torch.cat([
            p.grad.detach().float().reshape(-1) if p.grad is not None
            else torch.zeros(p.numel(), device=self.device)
            for p in self._trainables()])

    def penalty(self):
        """Squared EWC penalty at the current weights (for monitoring and to
        match ewc_on; ``observe`` adds the penalty's gradient directly instead
        of backpropagating through this).

        Args: none.

        Returns:
            scalar tensor: lambda * sum(F * (theta - theta*)^2), 0 before the
            first end_task.
        """
        if self.checkpoint is None:
            return torch.tensor(0.0, device=self.device)
        return self.args.e_lambda * (
            self.fish * (self._params_vector() - self.checkpoint) ** 2).sum()

    def end_task(self, dataset):
        """Estimate a Fisher value per weight from the training loss on the
        ending task, add it into the running total (after scaling the old total
        by gamma), and checkpoint the weights.

        Args:
            dataset: current task's dataset; ``train_loader`` yields
                (aug, label, ...) batches.

        Outputs: none — updates ``self.fish``, ``self.checkpoint``.
        """
        fish = torch.zeros(sum(p.numel() for p in self._trainables()), device=self.device)
        n_batches = 0
        for j, data in enumerate(dataset.train_loader):
            if self.args.ewc_fisher_batches and j >= self.args.ewc_fisher_batches:
                break
            inputs, labels = data[0].to(self.device), data[1].to(self.device)
            self.opt.zero_grad()
            loss = self.net.lm_loss(inputs, labels)
            loss.backward()
            fish += self._grads_vector() ** 2
            n_batches += 1
        self.opt.zero_grad()  # clear Fisher grads so they don't leak into the next observe
        fish /= max(n_batches, 1)

        if self.fish is None:
            self.fish = fish
        else:
            self.fish *= self.args.gamma
            self.fish += fish
        self.checkpoint = self._params_vector()

    def _add_penalty_grads(self):
        """Add the penalty's gradient 2*lambda*F*(theta - theta*) into each
        trainable param's ``.grad``. Params that got no grad from the task
        backward are skipped — their Fisher is zero anyway, and giving them a
        grad would let AdamW apply weight decay to them.

        Args: none.

        Outputs: none — mutates ``p.grad`` in place.
        """
        pen = self.args.e_lambda * 2 * self.fish * (self._params_vector() - self.checkpoint)
        offset = 0
        for p in self._trainables():
            n = p.numel()
            if p.grad is not None:
                p.grad += pen[offset:offset + n].view_as(p).to(p.dtype)
            offset += n

    def observe(self, inputs, labels, not_aug_inputs, epoch=None, **kwargs):
        """One EWC step on the training loss: backward on the current batch,
        add the penalty's gradient, clip, step.

        Args:
            inputs: augmented, normalized images [B, C, H, W].
            labels: class ids [B].
            not_aug_inputs: unused (the net handles the transformed inputs).
            epoch: unused (observe contract).

        Returns:
            scalar task loss value (the penalty affects grads only, as in ewc_on).
        """
        self.opt.zero_grad()
        loss = self.net.lm_loss(inputs, labels)
        assert not torch.isnan(loss)
        loss.backward()
        if self.checkpoint is not None:
            self._add_penalty_grads()
        if self.args.vlm_clip_grad:
            torch.nn.utils.clip_grad_norm_(list(self._trainables()),
                                           self.args.vlm_clip_grad)
        self.opt.step()

        return loss.item()
