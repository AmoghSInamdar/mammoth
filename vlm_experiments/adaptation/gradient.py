# Copyright 2026-present, Amogh Inamdar, Vici Milenia, Richard Zemel.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Gradient k-shot adaptation, the generative version of
utils/few_shot.adapt_model.

Same steps as the repo ResNet pipeline: deepcopy the model, make a fresh
optimizer (same class, adapt lr, weight_decay 0), and run ``num_steps`` passes
over the shuffled k-shot loader calling ``model.observe``. For vlm-sgd that
observe step trains the model to generate `` <classname><eos>``
(models/vlm_sgd.py) instead of a classifier cross-entropy loss; that one swap
is the whole "generative version". With k=0 we skip adaptation and score the
checkpoint as-is (the no-shot point of the accuracy-vs-k curve).
"""

import copy
import logging

import torch

from vlm_experiments.adaptation.support import (EvalState, make_adapt_loader,
                                                sample_k_shot)


class GradientAdapter:
    """Repo k-shot gradient adaptation on a generative VLM checkpoint."""

    name = 'gradient'

    def __init__(self, num_steps: int = 5, lr: float = 0.01, batch_size: int = 32):
        """Stash the adaptation knobs. Same defaults family as
        run_k_shot_evaluation.

        Args:
            num_steps: passes over the k-shot loader.
            lr: adaptation learning rate (weight_decay forced 0).
            batch_size: k-shot loader batch cap.

        Outputs: none — knobs stored.
        """
        self.num_steps = num_steps
        self.lr = lr
        self.batch_size = batch_size
        self.model = None
        self._own_copy = False
        self.n_support = 0

    def prepare(self, model, state: EvalState, task_id: int, k: int, seed: int) -> 'GradientAdapter':
        """Sample k-shot support of the eval task and fine-tune a COPY on it.
        Original checkpoint untouched; k<=0 reuse it directly (no copy, no
        steps).

        Args:
            model: loaded ContinualModel (vlm-sgd) checkpoint.
            state: EvalState with raw train pool.
            task_id: eval task whose classes supply the support.
            k: shots per class.
            seed: support sampling seed.

        Returns:
            self, with ``self.model`` ready to score.
        """
        self._release()
        imgs, labels = sample_k_shot(state, task_id, k, seed)
        self.n_support = len(labels)
        loader = make_adapt_loader(imgs, labels, state.train_transform, self.batch_size)
        if loader is None:
            self.model, self._own_copy = model, False
            return self

        adapted = copy.deepcopy(model)
        self._own_copy = True
        opt_class = type(adapted.opt) if hasattr(adapted, 'opt') else torch.optim.SGD
        adapted.opt = opt_class(adapted.net.parameters(), lr=self.lr, weight_decay=0.0)

        adapted.net.train()
        logging.info(f"[gradient] adapting {self.num_steps} steps, lr={self.lr}, "
                     f"support={self.n_support}")
        for step in range(self.num_steps):
            total, batches = 0.0, 0
            for inputs, labels_b, not_aug in loader:
                inputs = inputs.to(adapted.device)
                labels_b = labels_b.to(adapted.device, dtype=torch.long)
                not_aug = not_aug.to(adapted.device)
                total += adapted.observe(inputs, labels_b, not_aug, epoch=0)
                batches += 1
            logging.debug(f"[gradient] step {step + 1}/{self.num_steps} "
                          f"loss={total / max(batches, 1):.4f}")
        adapted.net.eval()
        self.model = adapted
        return self

    @torch.no_grad()
    def scores(self, x: torch.Tensor) -> torch.Tensor:
        """Candidate scores from the (adapted) net: the log-probability the
        model gives to generating each class name, [B, N_CLASSES]. Active width
        set by the evaluator.

        Args:
            x: transformed query batch [B, C, H, W].

        Returns:
            scores [B, N_CLASSES] float32.
        """
        return self.model.net(x)

    def set_width(self, width: int) -> None:
        """Restrict candidate scoring to the first ``width`` classes (same
        convention as quick_sweep).

        Args:
            width: class_hi of the eval task.

        Outputs: none — flips net.n_active.
        """
        if hasattr(self.model.net, 'set_active_classes'):
            self.model.net.set_active_classes(width)

    @torch.no_grad()
    def prior_scores(self, width: int) -> torch.Tensor:
        """PMI prior: candidate scores under the SAME prompt but with a plain
        mid-gray image (no class information) in the query slot, from the
        ADAPTED weights. This measures the menu bias of the current prompt so
        it can be subtracted out (Holtzman et al. 2021, Zhao et al. 2021). It
        does not depend on the query, so it is computed once per task eval.

        Args:
            width: score candidates [0, width).

        Returns:
            prior scores [N_CLASSES] float32.
        """
        from vlm_experiments.adaptation.scoring import (encode_context,
                                                        gray_pil,
                                                        teacher_forced_scores)
        net = self.model.net
        ids, mask, img = encode_context(net, net.prompt, [gray_pil(net)],
                                        self.model.device)
        return teacher_forced_scores(net, ids, mask, net.image_features(img), width)

    @torch.no_grad()
    def generate(self, x: torch.Tensor, max_new_tokens: int = 8):
        """Let the adapted VLM generate an answer freely under the fixed
        prompt for every query — the generate-and-parse measure. Batched.

        Args:
            x: transformed query batch [B, C, H, W].
            max_new_tokens: generation cap.

        Returns:
            list of generated answer strings, length B.
        """
        net = self.model.net
        bsz = x.shape[0]
        out = net.generate_ids(net.prompt_ids.expand(bsz, -1),
                               net.prompt_mask.expand(bsz, -1),
                               net.image_inputs(x), max_new_tokens)
        return [a.strip() for a in net.processor.tokenizer.batch_decode(
            out[:, net.prompt_ids.shape[1]:], skip_special_tokens=True)]

    @torch.no_grad()
    def explain(self, x: torch.Tensor, state: EvalState, max_new_tokens: int = 8):
        """Prompt + generated answer pairs for the first-5 log lines.

        Args:
            x: transformed query batch [B, C, H, W].
            state: EvalState (unused here, uniform adapter interface).
            max_new_tokens: generation cap.

        Returns:
            list of (prompt_text, answer_text) per query.
        """
        net = self.model.net
        return [(net.prompt, a) for a in self.generate(x, max_new_tokens)]

    def _release(self) -> None:
        """Drop the adapted deepcopy so the next (task, k, seed) start clean.
        No-op when we only borrowed the original.

        Args: none.

        Outputs: none — frees GPU memory.
        """
        if self._own_copy and self.model is not None:
            del self.model
            torch.cuda.empty_cache()
        self.model, self._own_copy = None, False
