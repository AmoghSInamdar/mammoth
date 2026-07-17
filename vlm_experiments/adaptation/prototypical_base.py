# Copyright 2026-present, Amogh Inamdar, Vici Milenia, Richard Zemel.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Shared prototypical network machinery (Snell et al. 2017), no gradients.

The recipe: turn each support image into a feature vector, build one prototype
per class as the mean of that class's support feature vectors, and score a query
by its squared Euclidean distance to each prototype (closer is better). No
fine-tuning, no temperature, no feature normalization.

The feature vectors come from the frozen VLM checkpoint (forward with
returnt='features': fixed prompt, last-token hidden state), so the model doing
the embedding is the continually-trained model itself. Support images go through
the dataset TEST transform, so they are in the same input domain as queries with
no train augmentation. Classes with no support get a score of -1e9 so they are
never predicted.

The one thing that varies between prototypical modes is which classes supply
support -- the eval task's own classes, or every class the checkpoint has seen.
That choice lives in the ``_select_support`` hook; embedding, prototype build,
scoring, and explain are shared here. Subclass and override ``_select_support``
to define a mode.
"""

import logging

import torch

from vlm_experiments.adaptation.support import EvalState, to_test_tensors


class PrototypicalBase:
    """Snell et al. 2017 prototype head on frozen VLM features. Subclasses
    override ``_select_support`` to fix which classes get a prototype."""

    name = 'prototypical_base'

    def __init__(self, embed_batch: int = 32):
        """Store the embedding batch size. The model is set later in prepare.

        Args:
            embed_batch: support and query images per feature forward.

        Outputs: none.
        """
        self.embed_batch = embed_batch
        self.model = None
        self.protos = None       # [N_CLASSES, D] prototype per class (zeros if none)
        self.has_proto = None    # [N_CLASSES] bool mask
        self.n_support = 0
        self._span = (0, 0)      # seen class range, for explain()

    def _select_support(self, state: EvalState, task_id: int, k: int, seed: int):
        """Pick the support pool for this mode: the class range it covers and the
        k-shot support examples drawn from it. The only method a mode subclass
        must define.

        Args:
            state: EvalState with raw train pool and TEST transform.
            task_id: eval task supplying support classes (used differently per mode).
            k: shots per class.
            seed: support sampling seed.

        Returns:
            tuple (span, imgs, labels): span is the (lo, hi) class range for
            explain(); imgs and labels are the sampled support arrays.
        """
        raise NotImplementedError

    @torch.no_grad()
    def _embed(self, x: torch.Tensor) -> torch.Tensor:
        """Feature vectors from the frozen model, computed in chunks. The
        feature is the VLM last-token hidden state.

        Args:
            x: image batch [B, C, H, W] (any device, moved inside).

        Returns:
            features [B, D] float32 on model device.
        """
        net, device = self.model.net, self.model.device
        out = []
        for lo in range(0, x.shape[0], self.embed_batch):
            out.append(net(x[lo:lo + self.embed_batch].to(device), returnt='features'))
        return torch.cat(out, dim=0)

    def prepare(self, model, state: EvalState, task_id: int, k: int,
                seed: int) -> 'PrototypicalBase':
        """Build one prototype per class as the mean of that class's support
        feature vectors. Which classes contribute comes from the subclass's
        ``_select_support`` call. With k<=0 there is no support, so no prototypes
        are built and all scores come out flat.

        Args:
            model: loaded ContinualModel (frozen, never touched).
            state: EvalState with raw train pool and TEST transform.
            task_id: eval task (used differently per mode in _select_support).
            k: shots per class.
            seed: support sampling seed.

        Returns:
            self, prototypes ready.
        """
        self.model = model
        model.net.eval()
        n_classes = state.n_classes

        self._span, imgs, labels = self._select_support(state, task_id, k, seed)
        self.n_support = len(labels)

        dim = self.model.net.embed_dim
        device = self.model.device
        self.protos = torch.zeros(n_classes, dim, device=device)
        self.has_proto = torch.zeros(n_classes, dtype=torch.bool, device=device)
        if self.n_support == 0:
            logging.info(f"[{self.name}] k=0 -> no prototypes, uniform scores")
            return self

        z = self._embed(to_test_tensors(imgs, state.test_transform))
        labels_t = torch.as_tensor(labels, device=device)
        for c in labels_t.unique().tolist():
            self.protos[c] = z[labels_t == c].mean(dim=0)   # prototype = class mean
            self.has_proto[c] = True
        lo, hi = self._span
        logging.info(f"[{self.name}] {int(self.has_proto.sum())} prototypes "
                     f"(classes {lo}-{hi - 1}) from {self.n_support} support examples")
        return self

    @torch.no_grad()
    def scores(self, x: torch.Tensor) -> torch.Tensor:
        """Score a query by the negative squared Euclidean distance to each
        prototype, so a closer prototype gets a higher score. Classes with no
        prototype get -1e9 so they are never predicted.

        Args:
            x: transformed query batch [B, C, H, W].

        Returns:
            scores [B, N_CLASSES] float32.
        """
        if self.n_support == 0:
            # No support: nothing to compare against, return flat scores.
            return torch.zeros(x.shape[0], self.protos.shape[0], device=self.model.device)
        z = self._embed(x)                                        # [B, D]
        d2 = torch.cdist(z, self.protos, p=2).pow(2)              # [B, N_CLASSES]
        scores = -d2
        scores[:, ~self.has_proto] = -1e9
        return scores

    def set_width(self, width: int) -> None:
        """No-op: the prototype mask already limits which classes can be
        predicted, and the evaluator slices [:, :width] anyway.

        Args:
            width: class_hi of the eval task (unused).

        Outputs: none.
        """
        pass

    def prior_scores(self, width: int):
        """No PMI correction for protonet -- distances have no class prior to
        cancel out. The evaluator skips the PMI columns when this returns None.

        Args:
            width: unused.

        Returns:
            None.
        """
        return None

    def generate(self, x: torch.Tensor, max_new_tokens: int = 8):
        """Protonet never produces text. The evaluator skips the generation
        columns when this returns None.

        Args:
            x: unused.
            max_new_tokens: unused.

        Returns:
            None.
        """
        return None

    @torch.no_grad()
    def explain(self, x: torch.Tensor, state: EvalState, max_new_tokens: int = 8):
        """Protonet has no text prompt, so describe the setup and report the
        nearest-prototype class name as the answer.

        Args:
            x: transformed query batch [B, C, H, W].
            state: EvalState for class names.
            max_new_tokens: unused (kept for the shared adapter interface).

        Returns:
            list of (description, predicted_name) per query.
        """
        lo, hi = self._span
        desc = (f"[{self.name}] query embedded by frozen VLM, nearest of "
                f"{int(self.has_proto.sum())} class prototypes (classes {lo}-{hi - 1}, "
                f"{self.n_support} support)")
        preds = self.scores(x).argmax(dim=1)
        return [(desc, state.class_names[int(p)]) for p in preds]

    def _release(self) -> None:
        """Drop prototypes. Model was never copied, nothing else to free.

        Args: none.

        Outputs: none.
        """
        self.protos, self.has_proto = None, None
