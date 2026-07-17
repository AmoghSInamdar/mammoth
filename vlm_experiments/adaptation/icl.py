# Copyright 2026-present, Amogh Inamdar, Vici Milenia, Richard Zemel.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
In-context learning adaptation, with no weight updates (in-context learning:
GPT-3, Brown et al. 2020; image-and-text form: Flamingo, Alayrac et al. 2022).

The weights are frozen. The k labeled examples go IN THE PROMPT as (image,
caption) pairs, the query image goes last, and the model reads the whole
context to answer. To classify, compare the log-probability the model assigns
to generating each candidate name, divided by name length — the same rule the
backbone's own score_classes uses, so k=0 ICL matches plain checkpoint scoring.

PMI prior (Holtzman et al. 2021): the same demo context but with the query
image replaced by a plain gray image carrying no class information (Zhao et al.
2021) — what the model would answer with no evidence, with the same token
layout. Raw score minus prior is what the model actually read off the image.

The demo format reuses the checkpoint's own training template so the
fine-tuned model sees the same kind of input it was trained on:

    <image>This is a photo of a apple.\n<image>This is a photo of a dog.\n...
    <image>This is a photo of a            <- score candidates / generate here

Demos are shuffled (seeded) so classes are mixed together rather than blocked
by class. Candidate answers reuse net._answer_ids (`` <name><eos>``) — the
same tokenization as training and as the gradient path's scoring.
"""

import logging

import torch

from vlm_experiments.adaptation.scoring import (encode_context, gray_pil,
                                                teacher_forced_scores)
from vlm_experiments.adaptation.support import (EvalState, batch_to_pil,
                                                sample_k_shot, to_pil)


class ICLAdapter:
    """Few-shot prompting with images and text on a frozen VLM checkpoint."""

    name = 'icl'

    def __init__(self):
        """Nothing to configure — k / seed arrive in prepare, scoring knobs
        (length_norm, score_chunk) come from the checkpoint's backbone.

        Args: none.

        Outputs: none.
        """
        self.model = None
        self.demo_pils = []
        self.demo_text = ''
        self.n_support = 0
        self.width = 0
        self._ctx_warned = False

    def prepare(self, model, state: EvalState, task_id: int, k: int,
                seed: int) -> 'ICLAdapter':
        """Build the demo context: k raw train images per eval-task class,
        shuffled with the seed, one `` <prompt> <name>.\\n`` block each. Model
        weights untouched.

        Args:
            model: loaded ContinualModel (vlm-sgd) checkpoint, frozen.
            state: EvalState with raw train pool + names.
            task_id: eval task supplying demo classes.
            k: shots per class in the prompt.
            seed: sampling + shuffle seed.

        Returns:
            self, demo context ready.
        """
        import numpy as np
        self.model = model
        model.net.eval()

        imgs, labels = sample_k_shot(state, task_id, k, seed)
        self.n_support = len(labels)

        # Seeded shuffle so demos are mixed across classes, not blocked by class.
        order = np.random.RandomState(seed).permutation(self.n_support)
        pils = to_pil(imgs)
        net = model.net
        self.demo_pils = [pils[i] for i in order]
        self.demo_text = ''.join(
            f"{net.prompt} {state.class_names[int(labels[i])].replace('_', ' ')}.\n"
            for i in order)
        logging.info(f"[icl] prompt carry {self.n_support} demos "
                     f"({k} per class, task {task_id})")
        return self

    def set_width(self, width: int) -> None:
        """Candidate set = first ``width`` classes (the evaluator slices menus
        out of the full score matrix later).

        Args:
            width: widest menu the evaluator needs.

        Outputs: none — stored for scores().
        """
        self.width = width

    def _encode_query(self, query_pil):
        """Run the processor on the demos + one query: token ids with every
        image inline, pixel_values for all images. query_pil None = PMI prior
        context: SAME prompt, query slot filled with a plain gray image with no
        class information (token layout identical to a real query).

        Args:
            query_pil: PIL image of the query, or None for the prior.

        Returns:
            (input_ids, attention_mask, image_inputs) on model device.
        """
        net = self.model.net
        if query_pil is None:
            query_pil = gray_pil(net)
        text = self.demo_text + net.prompt       # end exactly where answer starts
        pils = self.demo_pils + [query_pil]
        ids, mask, img = encode_context(net, text, pils, self.model.device)
        # Past the model's context limit scores become unreliable but there is
        # no crash; k=10 x 10 classes = 7.7k of 8.2k positions, so warn early.
        limit = net.context_limit()
        if limit and ids.shape[1] > limit and not self._ctx_warned:
            logging.warning(f"[icl] prompt {ids.shape[1]} tokens exceeds model "
                            f"context {limit}; scores unreliable (reduce k)")
            self._ctx_warned = True
        return ids, mask, img

    def _score_context(self, query_pil) -> torch.Tensor:
        """Candidate scoring for one context: the log-probability the model
        gives to generating each `` <name><eos>`` after it, via the shared
        scorer. Vision tower runs once per context.

        Args:
            query_pil: PIL query image, or None for the prior context.

        Returns:
            scores [N_CLASSES] float32; classes >= width get -1e9.
        """
        net = self.model.net
        ids, mask, img = self._encode_query(query_pil)
        return teacher_forced_scores(net, ids, mask, net.image_features(img), width=self.width)

    @torch.no_grad()
    def scores(self, x: torch.Tensor) -> torch.Tensor:
        """Score a transformed query batch one prompt at a time (each query
        gets the full demo context, and the prompts are too big to batch).

        Args:
            x: transformed query batch [B, C, H, W].

        Returns:
            scores [B, N_CLASSES] float32.
        """
        net = self.model.net
        pils = batch_to_pil(x, net.data_mean, net.data_std)
        return torch.stack([self._score_context(p) for p in pils])

    @torch.no_grad()
    def prior_scores(self, width: int) -> torch.Tensor:
        """PMI prior: candidate scores under the demo context with a plain gray
        query image carrying no class information (Holtzman et al. 2021). One
        context per (task, k, seed) — it does not depend on the query.

        Args:
            width: score candidates [0, width).

        Returns:
            prior scores [N_CLASSES] float32.
        """
        self.set_width(width)
        return self._score_context(None)

    @torch.no_grad()
    def generate(self, x: torch.Tensor, max_new_tokens: int = 8):
        """Let the model generate an answer freely after the demo prompt for
        every query — the generate-and-parse measure.

        Args:
            x: transformed query batch [B, C, H, W].
            max_new_tokens: generation cap.

        Returns:
            list of generated answer strings, length B.
        """
        net = self.model.net
        answers = []
        for pil in batch_to_pil(x, net.data_mean, net.data_std):
            ids, mask, img = self._encode_query(pil)
            gen = net.generate_ids(ids, mask, img, max_new_tokens)
            answers.append(net.processor.tokenizer.decode(
                gen[0, ids.shape[1]:], skip_special_tokens=True).strip())
        return answers

    @torch.no_grad()
    def explain(self, x: torch.Tensor, state: EvalState, max_new_tokens: int = 8):
        """Prompt + generated answer pairs for the first-5 log lines.

        Args:
            x: transformed query batch [B, C, H, W].
            state: EvalState (unused, uniform adapter interface).
            max_new_tokens: generation cap.

        Returns:
            list of (prompt_text, answer_text) per query.
        """
        prompt = self.demo_text + self.model.net.prompt
        return [(prompt, a) for a in self.generate(x, max_new_tokens)]

    def _release(self) -> None:
        """Drop the demo context. Weights were never copied.

        Args: none.

        Outputs: none.
        """
        self.demo_pils, self.demo_text = [], ''
