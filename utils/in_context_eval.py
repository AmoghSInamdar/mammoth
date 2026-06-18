# Copyright 2026-present, Amogh Inamdar, Vici Milenia, Richard Zemel.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Gradient-free few-shot evaluation for measuring SAUCE.

Instead of adapting a ResNet via gradient descent on k-shot examples
(see ``utils/few_shot.py``), this module measures plasticity by conditioning a
frozen model on the ``k`` labeled support examples and reading off a prediction
--- *no gradient updates*. ``k`` plays the role that gradient steps play in the
ResNet path, so the accuracy-vs-k curves feed the *same* SAUCE computation
(``utils/per_shot_plasticity.py``) and plotting.

Two distinct adaptation regimes, behind one ``--icl_backend`` toggle. They are
NOT all "in-context learning" --- only the generative VLMs are:

REPRESENTATION-PROBE / frozen-feature few-shot (the support set defines a
non-parametric head; the model never attends over the demonstrations):
- ``clip``   : HuggingFace CLIP. Query scored by cosine similarity to a
               per-class prototype that fuses the class-name *text* embedding
               (the k=0 zero-shot prior) with the mean of the ``k`` support
               *image* embeddings.
- ``dinov2`` : DINOv2 ViT image features + nearest-centroid over the ``k``
               support embeddings per class. No text.
- ``vit``    : the repo's own ``backbone/vit.py`` (ImageNet-pretrained ViT-B/16)
               features, same nearest-centroid head as ``dinov2``.

TRUE IN-CONTEXT LEARNING (the ``k`` labeled images are interleaved with the
query in one multimodal prompt; the transformer attends over the demonstrations
to produce the answer):
- ``qwen2vl`` : Qwen2-VL-2B-Instruct, local/free. Single-letter labels scored by
                next-token log-prob -> real softmax distribution.
- ``llava``   : LLaVA-Interleave-0.5B, local/free, same scoring.
- ``vlm``     : Claude (API). Free-form generation parsed to a class; loss is a
                hard-label proxy and NOT comparable to the local VLMs' CE.

The named/anonymized ablation (``--icl_anonymize``) isolates true ICL from
zero-shot recall: labels are single-token letters assigned by a SEEDED RANDOM
PERMUTATION per episode, so the emitted label is decoupled from the image's
visual identity (anonymization bites even on MNIST, where positional indexing
would not). Named runs additionally show the class name in the demo text;
anon runs show only the letter, so the mapping must be learned in context.

Faithfulness to ``utils/evaluate.evaluate`` (so accuracy/loss are comparable to
the ResNet curves):

- Accuracy is a PERCENT in [0, 100] = correct / total * 100.
- The candidate label set is ``[0 : dataset.get_offsets(task_id)[1]]`` --- the
  cumulative "classes seen so far" for class-il datasets, and the fixed
  ``[0:N_CLASSES_PER_TASK]`` set for domain-il datasets (rot/smooth-mnist),
  exactly as ``get_offsets`` resolves it.
- Feature backends emit a logits-like score vector over that range; generative
  VLMs emit a softmax over candidate label tokens. argmax == prediction, and CE
  over that distribution is the ``loss`` column.

All backends consume the *already-transformed* tensors produced by the
dataset's own train/test loaders. This means per-dataset normalization,
channel handling, and the per-task rotation of rot/smooth-mnist are applied by
the dataset itself --- support and query always share the same input domain,
with no manual re-normalization or re-rotation.
"""

from __future__ import annotations

import base64
import io
import logging
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from datasets.utils.continual_dataset import ContinualDataset


ICL_BACKENDS = ('clip', 'dinov2', 'vit', 'vlm', 'qwen2vl', 'llava')

# Backends that score the whole test set cheaply (local feature extractors) vs.
# generative VLMs whose queries are sampled.
_LOCAL_BACKENDS = ('clip', 'dinov2', 'vit')

# Open-source generative VLMs run locally and perform TRUE in-context learning:
# the k labeled support images are interleaved with the query image in one
# multimodal prompt and the model generates the predicted label. Like the Claude
# `vlm` backend they are sampled (generation is the cost), but they are free and
# local. Maps backend name -> default HuggingFace model id (smallest of each
# family). Override with the ICL_VLM_MODEL env var.
_HF_VLM_MODELS = {
    'qwen2vl': 'Qwen/Qwen2-VL-2B-Instruct',
    'llava':   'llava-hf/llava-interleave-qwen-0.5b-hf',
}
_GENERATIVE_VLM_BACKENDS = tuple(_HF_VLM_MODELS)  # ('qwen2vl', 'llava')


# --------------------------------------------------------------------------- #
# Per-task data access (mirrors utils/few_shot.create_k_shot_eval_dataset but
# also keeps the per-task *train* loader, which is the support pool).
# --------------------------------------------------------------------------- #

@dataclass
class _TaskData:
    """Already-transformed support pool + query loader for one task."""
    support_inputs: torch.Tensor   # [N_support, C, H, W] (augmented/transformed)
    support_labels: torch.Tensor   # [N_support]
    test_loader: torch.utils.data.DataLoader
    n_classes: int                 # get_offsets(task_id)[1]: candidate label-set width
    class_lo: int                  # get_offsets(task_id)[0]
    class_hi: int                  # get_offsets(task_id)[1]


def _build_task_data(dataset: ContinualDataset, task_id: int) -> _TaskData:
    """
    Build a fresh dataset advanced to ``task_id`` and collect the task's support
    pool (its train loader, materialized once) and query loader (its test
    loader). Cached on the parent ``dataset`` so repeated k/seed calls are cheap.

    The per-task transform (including rot/smooth-mnist rotation) is applied by
    the loaders, so the returned tensors are in the exact input domain that
    ``utils.evaluate.evaluate`` would feed the model.
    """
    cache = getattr(dataset, '_icl_task_cache', None)
    if cache is None:
        cache = {}
        dataset._icl_task_cache = cache
    if task_id in cache:
        return cache[task_id]

    from datasets import get_dataset_class
    dataset_class = get_dataset_class(dataset.args)
    fresh = dataset_class(dataset.args)

    train_loader = None
    for _ in range(task_id + 1):
        train_loader, _ = fresh.get_data_loaders()  # advances one task per call

    # Materialize the support pool (this task's training examples, transformed).
    # Some Mammoth loaders are unbatched (batch_size=None), yielding one sample
    # per step with a 0-dim label and a [C,H,W] image; _stack_batches normalizes
    # both the batched and unbatched cases to [N,C,H,W] / [N].
    support_inputs, support_labels = _stack_batches(train_loader)

    class_lo, class_hi = fresh.get_offsets(task_id)
    task_data = _TaskData(
        support_inputs=support_inputs,
        support_labels=support_labels,
        test_loader=fresh.test_loaders[task_id],
        n_classes=class_hi,
        class_lo=class_lo,
        class_hi=class_hi,
    )
    cache[task_id] = task_data
    return task_data


def _stack_batches(loader) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collect (inputs, labels) from a loader into [N, C, H, W] / [N], tolerating
    unbatched loaders (batch_size=None) whose elements are [C,H,W] images with
    0-dim scalar labels. Returns the first two elements of each yielded tuple.
    """
    inputs, labels = [], []
    for batch in loader:
        x, y = batch[0], batch[1]
        if not torch.is_tensor(x):
            x = torch.as_tensor(np.asarray(x))
        if x.dim() == 3:          # [C,H,W] unbatched -> [1,C,H,W]
            x = x.unsqueeze(0)
        # Labels may be a batched tensor, a 0-dim tensor, a python int, or ndarray.
        if torch.is_tensor(y):
            y = y.reshape(-1) if y.dim() > 0 else y.reshape(1)
        else:
            y = torch.as_tensor(np.atleast_1d(np.asarray(y)), dtype=torch.long)
        inputs.append(x)
        labels.append(y)
    if not inputs:
        return torch.empty(0), torch.empty(0, dtype=torch.long)
    return torch.cat(inputs, dim=0), torch.cat(labels, dim=0).long()


def _iter_test_batches(loader, batch_size: int = 128):
    """
    Yield (inputs[B,C,H,W], labels[B]) from a (possibly unbatched) test loader,
    re-chunking into batches of ``batch_size`` for efficient embedding.
    """
    inputs, labels = _stack_batches(loader)
    for i in range(0, len(labels), batch_size):
        yield inputs[i:i + batch_size], labels[i:i + batch_size]


def _sample_support(task: _TaskData, k: int, seed: int) -> Dict[int, torch.Tensor]:
    """
    Sample ``k`` support examples *per class* from the task's candidate label
    set ``[class_lo, class_hi)``. Returns {class_id: [n<=k, C, H, W]}.

    Seeding matches the spirit of ``utils/few_shot.create_k_shot_loader``
    (per-class ``np.random.choice`` under a fixed seed) so the support set is
    reproducible across backends and across runs.
    """
    if k <= 0:
        return {}
    labels = task.support_labels.numpy()
    per_class: Dict[int, torch.Tensor] = {}
    for class_id in range(task.class_lo, task.class_hi):
        idx = np.where(labels == class_id)[0]
        if len(idx) == 0:
            continue
        n = min(k, len(idx))
        rng = np.random.RandomState(seed + class_id)  # decorrelate classes
        chosen = rng.choice(idx, size=n, replace=False)
        per_class[class_id] = task.support_inputs[chosen]
    return per_class


# --------------------------------------------------------------------------- #
# Image helpers: dataset tensors -> uint8 RGB images for CLIP/DINOv2/VLM.
# The loader tensors are normalized/augmented for the *original* backbone; for
# the pretrained foundation models we want clean RGB, so we de-normalize back to
# [0,1] heuristically (clamp) and let each backend's own preprocessing run.
# --------------------------------------------------------------------------- #

def _to_uint8_rgb(batch: torch.Tensor) -> np.ndarray:
    """
    [B, C, H, W] float tensor (possibly normalized) -> [B, H, W, 3] uint8.

    We min-max rescale per image to [0,1] (robust to whatever normalization the
    dataset applied) then expand grayscale to 3 channels. This is intentionally
    backend-agnostic: CLIP/DINOv2 apply their own ImageNet normalization on top.
    """
    x = batch.detach().float().cpu()
    if x.dim() == 3:
        x = x.unsqueeze(1)
    b, c, h, w = x.shape
    flat = x.view(b, c, -1)
    mn = flat.min(dim=2, keepdim=True).values.unsqueeze(-1)
    mx = flat.max(dim=2, keepdim=True).values.unsqueeze(-1)
    x = (x - mn) / (mx - mn + 1e-6)
    if c == 1:
        x = x.repeat(1, 3, 1, 1)
    elif c > 3:
        x = x[:, :3]
    x = (x.clamp(0, 1) * 255).round().to(torch.uint8)
    return x.permute(0, 2, 3, 1).numpy()  # [B, H, W, 3]


def _scores_to_acc_loss(
    scores: torch.Tensor,      # [N_query, n_classes] logits-like
    labels: torch.Tensor,      # [N_query]
    n_classes: int,
) -> Tuple[float, float, Dict[int, Tuple[int, int]]]:
    """
    Reproduce evaluate()'s accuracy (argmax over [:, :n_classes], percent) and a
    cross-entropy loss over the same scores. Also returns per-class (correct,
    total) for the per-digit metadata columns.
    """
    scores = scores[:, :n_classes]
    preds = scores.argmax(dim=1)
    correct = (preds == labels).sum().item()
    total = labels.numel()
    accuracy = correct / total * 100.0 if total else 0.0
    loss = F.cross_entropy(scores, labels).item() if total else 0.0

    per_class: Dict[int, Tuple[int, int]] = {}
    for c in labels.unique().tolist():
        m = labels == c
        per_class[int(c)] = (int((preds[m] == c).sum().item()), int(m.sum().item()))
    return accuracy, loss, per_class


# --------------------------------------------------------------------------- #
# Local feature-extractor backends (clip / dinov2 / vit): a shared
# nearest-centroid / prototype head over support embeddings, with an optional
# text prior (CLIP) that supplies the k=0 zero-shot prediction.
# --------------------------------------------------------------------------- #

class _FeatureBackend:
    """Embeds images to L2-normalized vectors and (optionally) class-name text."""

    has_text_prior: bool = False

    def __init__(self, device: str):
        self.device = device

    def embed_images(self, batch_uint8: np.ndarray) -> torch.Tensor:
        raise NotImplementedError

    def embed_text(self, class_names: List[str]) -> Optional[torch.Tensor]:
        return None


class _ClipBackend(_FeatureBackend):
    has_text_prior = True

    def __init__(self, device: str, model_name: str = 'openai/clip-vit-base-patch32'):
        super().__init__(device)
        from transformers import CLIPModel, CLIPProcessor
        self.model = CLIPModel.from_pretrained(model_name).to(device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)

    @torch.no_grad()
    def embed_images(self, batch_uint8: np.ndarray) -> torch.Tensor:
        from PIL import Image
        imgs = [Image.fromarray(im) for im in batch_uint8]
        inp = self.processor(images=imgs, return_tensors='pt').to(self.device)
        feats = self.model.get_image_features(**inp)
        return F.normalize(feats, dim=-1)

    @torch.no_grad()
    def embed_text(self, class_names: List[str]) -> torch.Tensor:
        prompts = [f"a photo of a {name}" for name in class_names]
        inp = self.processor(text=prompts, return_tensors='pt', padding=True).to(self.device)
        feats = self.model.get_text_features(**inp)
        return F.normalize(feats, dim=-1)


class _Dinov2Backend(_FeatureBackend):
    def __init__(self, device: str, model_name: str = 'facebook/dinov2-base'):
        super().__init__(device)
        from transformers import AutoImageProcessor, AutoModel
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()

    @torch.no_grad()
    def embed_images(self, batch_uint8: np.ndarray) -> torch.Tensor:
        from PIL import Image
        imgs = [Image.fromarray(im) for im in batch_uint8]
        inp = self.processor(images=imgs, return_tensors='pt').to(self.device)
        out = self.model(**inp)
        feats = out.last_hidden_state[:, 0]  # CLS token
        return F.normalize(feats, dim=-1)


class _RepoVitBackend(_FeatureBackend):
    """The repo's own ImageNet-pretrained ViT-B/16 (backbone/vit.py)."""

    def __init__(self, device: str):
        super().__init__(device)
        from backbone.vit import vit_backbone
        import torchvision.transforms as T
        # num_classes>0 only builds an (unused) head; returnt='features' bypasses it.
        self.model = vit_backbone(num_classes=1, pretrained=True).to(device).eval()
        # in21k-ft-in1k variant -> Inception mean/std, 224x224 RGB.
        self.tf = T.Compose([
            T.Resize((224, 224), antialias=True),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    @torch.no_grad()
    def embed_images(self, batch_uint8: np.ndarray) -> torch.Tensor:
        x = torch.from_numpy(batch_uint8).permute(0, 3, 1, 2).float() / 255.0
        x = self.tf(x).to(self.device)
        feats = self.model(x, returnt='features')
        return F.normalize(feats, dim=-1)


def _make_feature_backend(backend: str, device: str) -> _FeatureBackend:
    if backend == 'clip':
        return _ClipBackend(device)
    if backend == 'dinov2':
        return _Dinov2Backend(device)
    if backend == 'vit':
        return _RepoVitBackend(device)
    raise ValueError(f"Unknown feature backend: {backend}")


def _embed_in_batches(be: _FeatureBackend, batch_uint8: np.ndarray, bs: int = 64) -> torch.Tensor:
    out = []
    for i in range(0, len(batch_uint8), bs):
        out.append(be.embed_images(batch_uint8[i:i + bs]))
    return torch.cat(out, dim=0) if out else torch.empty(0)


def _evaluate_feature_backend(
    be: _FeatureBackend,
    dataset: ContinualDataset,
    task: _TaskData,
    k: int,
    seed: int,
    class_names: Optional[List[str]],
    temperature: float = 100.0,
) -> Tuple[float, float, Dict[int, Tuple[int, int]]]:
    """
    Prototype head. Each class ``c`` in [class_lo, class_hi) gets a prototype:
      - the mean of its ``k`` support image embeddings (if k>0 and any exist),
      - falling back to / fused with the class-name text embedding (CLIP only),
      - or, when there is neither support nor text (k=0, no text), a zero vector
        so the class contributes a neutral score.
    Query score = temperature * cosine(query, prototype), i.e. a logits-like
    vector over [0, class_hi). Slicing to [:, :n_classes] then matches evaluate.
    """
    n_classes = task.n_classes
    dim = None

    # Text prior (CLIP): prototypes for the candidate classes from their names.
    text_proto = None
    if be.has_text_prior and class_names is not None:
        names = class_names[task.class_lo:task.class_hi]
        text_emb = be.embed_text(names)  # [n_task_classes, D]
        dim = text_emb.shape[1]
        text_proto = {task.class_lo + i: text_emb[i] for i in range(len(names))}

    # Image prototypes from the k support examples.
    support = _sample_support(task, k, seed)
    img_proto: Dict[int, torch.Tensor] = {}
    for class_id, imgs in support.items():
        emb = _embed_in_batches(be, _to_uint8_rgb(imgs))
        if emb.numel():
            img_proto[class_id] = emb.mean(dim=0)
            dim = emb.shape[1]

    if dim is None:
        # k=0 and no text prior (dinov2/vit): nothing to predict from -> chance.
        # Build uniform scores so accuracy ~= 1/n_classes and loss is finite.
        return _score_uniform(be, task, n_classes)

    # Fuse text + image prototypes per class.
    protos = torch.zeros(n_classes, dim, device=be.device)
    for c in range(task.class_lo, task.class_hi):
        parts = []
        if text_proto is not None and c in text_proto:
            parts.append(text_proto[c])
        if c in img_proto:
            parts.append(img_proto[c].to(be.device))
        if parts:
            protos[c] = F.normalize(torch.stack(parts, 0).mean(0), dim=-1)

    # Score every query in the test loader.
    all_scores, all_labels = [], []
    for inputs, labels in _iter_test_batches(task.test_loader):
        q = _embed_in_batches(be, _to_uint8_rgb(inputs)).to(be.device)  # [B, D]
        scores = temperature * (q @ protos.t())                        # [B, n_classes]
        all_scores.append(scores.cpu())
        all_labels.append(labels)
    scores = torch.cat(all_scores, 0)
    labels = torch.cat(all_labels, 0)
    return _scores_to_acc_loss(scores, labels, n_classes)


def _score_uniform(be, task: _TaskData, n_classes: int):
    _, labels = _stack_batches(task.test_loader)
    scores = torch.zeros(labels.numel(), n_classes)  # argmax -> class 0; CE finite
    return _scores_to_acc_loss(scores, labels, n_classes)


# --------------------------------------------------------------------------- #
# VLM backend (Claude): true in-context examples. Sampled subset of queries.
# --------------------------------------------------------------------------- #

def _png_b64(img_uint8: np.ndarray) -> str:
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(img_uint8).resize((96, 96)).save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('ascii')


def _evaluate_vlm_backend(
    dataset: ContinualDataset,
    task: _TaskData,
    k: int,
    seed: int,
    class_names: Optional[List[str]],
    max_queries: int = 50,
    model: str = 'claude-opus-4-8',
) -> Tuple[float, float, Dict[int, Tuple[int, int]]]:
    """
    Place the k labeled support images (per candidate class) into the prompt as
    in-context examples, then ask the model to classify each query image. The
    candidate label set is the task's [class_lo, class_hi). We score a random
    subset of queries (API cost) and report percent accuracy; loss is left as a
    one-hot cross-entropy proxy (the VLM gives a hard label, not logits).
    """
    import anthropic

    if class_names is None:
        raise RuntimeError("VLM backend requires dataset.get_class_names()")
    candidate = list(range(task.class_lo, task.class_hi))
    names = {c: class_names[c] for c in candidate}

    support = _sample_support(task, k, seed)

    client = anthropic.Anthropic(api_key=os.environ.get('ANTHROPIC_API_KEY'))

    # Gather a query subset.
    q_inputs, q_labels = _stack_batches(task.test_loader)
    rng = np.random.RandomState(seed)
    sel = rng.choice(len(q_labels), size=min(max_queries, len(q_labels)), replace=False)

    label_list = ", ".join(f"{c}={names[c]}" for c in candidate)
    correct = 0
    per_class: Dict[int, List[int]] = {c: [0, 0] for c in candidate}

    for i in sel:
        content = [{
            "type": "text",
            "text": (
                f"You are a few-shot image classifier. Valid labels: {label_list}.\n"
                + ("Here are labeled example images:\n" if support else "No examples provided.\n")
            ),
        }]
        for c, imgs in support.items():
            for j in range(imgs.shape[0]):
                img = _to_uint8_rgb(imgs[j:j + 1])[0]
                content.append({"type": "text", "text": f"Example label {c} ({names[c]}):"})
                content.append({"type": "image", "source": {
                    "type": "base64", "media_type": "image/png", "data": _png_b64(img)}})
        qimg = _to_uint8_rgb(q_inputs[int(i):int(i) + 1])[0]
        content.append({"type": "text", "text": "Classify this image. Reply with ONLY the integer label:"})
        content.append({"type": "image", "source": {
            "type": "base64", "media_type": "image/png", "data": _png_b64(qimg)}})

        try:
            resp = client.messages.create(
                model=model, max_tokens=8,
                messages=[{"role": "user", "content": content}],
            )
            txt = resp.content[0].text if resp.content else ""
            m = re.search(r'-?\d+', txt)
            pred = int(m.group()) if m else -1
        except Exception as e:  # network / rate / parse -> count as wrong
            logging.warning(f"VLM query failed: {e}")
            pred = -1

        true = int(q_labels[int(i)])
        per_class[true][1] += 1
        if pred == true:
            correct += 1
            per_class[true][0] += 1

    n = len(sel)
    accuracy = correct / n * 100.0 if n else 0.0
    loss = float(-np.log(max(correct / n, 1e-6))) if n else 0.0  # proxy
    per_class_out = {c: (v[0], v[1]) for c, v in per_class.items() if v[1] > 0}
    return accuracy, loss, per_class_out


# --------------------------------------------------------------------------- #
# Open-source generative VLM backends (Qwen2-VL, LLaVA-Interleave): TRUE
# in-context learning, run locally, free. The k labeled support images are
# interleaved with the query image in one multimodal prompt; the model generates
# a label, which we parse. Sampled (generation cost), like the Claude path.
#
# `anonymize=True` replaces semantic class names with abstract symbols (the i-th
# candidate class shown in this task becomes label "0","1",... regardless of its
# true CIFAR/MNIST name). This forces the model to learn the support->label
# mapping purely IN CONTEXT rather than from prior knowledge of the class name —
# the canonical ablation distinguishing real ICL from zero-shot recall.
# --------------------------------------------------------------------------- #

# Process-level cache: (model, processor) per backend, loaded once per run.
_VLM_MODEL_CACHE: Dict[str, tuple] = {}


def _pil_rgb(img_uint8: np.ndarray, size: int = 224):
    from PIL import Image
    return Image.fromarray(img_uint8).convert('RGB').resize((size, size))


def _load_hf_vlm(backend: str, device: str):
    """Load (and cache) the HF model + processor for a generative-VLM backend."""
    if backend in _VLM_MODEL_CACHE:
        return _VLM_MODEL_CACHE[backend]
    import torch
    from transformers import AutoProcessor

    model_id = os.environ.get('ICL_VLM_MODEL') or _HF_VLM_MODELS[backend]
    logging.info(f"Loading generative VLM '{backend}' ({model_id}) on {device}...")
    dtype = torch.float16 if device != 'cpu' else torch.float32

    if backend == 'qwen2vl':
        from transformers import Qwen2VLForConditionalGeneration
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=dtype, low_cpu_mem_usage=True).to(device).eval()
    elif backend == 'llava':
        from transformers import LlavaForConditionalGeneration
        model = LlavaForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=dtype, low_cpu_mem_usage=True).to(device).eval()
    else:
        raise ValueError(f"Not a generative-VLM backend: {backend}")
    processor = AutoProcessor.from_pretrained(model_id)
    _VLM_MODEL_CACHE[backend] = (model, processor)
    return model, processor


# Single-token symbolic labels. Each candidate class is assigned ONE letter via
# a seeded random permutation per episode. This (a) makes every scored answer a
# single token -> no length bias and O(1) scoring; (b) decouples the emitted
# label from the image's visual identity, so even on MNIST "looks like a 3"
# tells the model nothing about which label to emit (the no-op that positional
# indexing had on domain-il datasets). Letters chosen to be single-token in the
# Qwen/LLaMA-family tokenizers used by both backends.
_LABEL_LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def _build_label_map(candidate: List[int], seed: int) -> Dict[int, str]:
    """Seeded random permutation: class_id -> single-letter label.

    Keyed ONLY on ``seed`` (not k), so the class->letter mapping is CONSTANT
    across all k-values within one seed -- the accuracy-vs-k curve is then one
    coherent task and its SAUCE area is well-defined. Different seeds give
    different permutations, so averaging over ``--n_seeds`` averages over label
    assignments and the ICL result is not a one-permutation fluke. Named and
    anon runs at the same seed share the same map (only the demo text differs).

    Falls back to two-char labels only if there are more candidates than letters
    (>26, e.g. full Split-CIFAR late tasks); those are scored at the FIRST token,
    so still single-token-cost (see _score_candidates)."""
    rng = np.random.RandomState(seed * 100003 + 7)  # decorrelate from sampling seed
    perm = list(range(len(candidate)))
    rng.shuffle(perm)
    out = {}
    for pos, c in zip(perm, candidate):
        out[c] = _LABEL_LETTERS[pos] if pos < len(_LABEL_LETTERS) else \
            _LABEL_LETTERS[pos // 26 - 1] + _LABEL_LETTERS[pos % 26]
    return out


def _demo_label_text(c: int, label_map: Dict[int, str], names: Dict[int, str],
                     anonymize: bool) -> str:
    """Text shown for a demo's label. Named: 'A (deer)' so the model may also use
    prior knowledge; Anon: 'A' only (mapping must be learned in-context). The
    SCORED continuation is always just the letter (label_map[c])."""
    if anonymize:
        return label_map[c]
    return f"{label_map[c]} ({names[c]})"


def _support_pairs(support, label_map, names, anonymize):
    """Flatten support into [(demo_label_text, PIL_image), ...] in fixed order."""
    pairs = []
    for c, imgs in support.items():
        txt = _demo_label_text(c, label_map, names, anonymize)
        for j in range(imgs.shape[0]):
            pairs.append((txt, _pil_rgb(_to_uint8_rgb(imgs[j:j + 1])[0])))
    return pairs


def _valid_labels_str(candidate, label_map):
    return ", ".join(label_map[c] for c in candidate)


def _build_llava_prompt(support, q_img, candidate, label_map, names, anonymize, processor):
    """LLaVA-Interleave prompt built MANUALLY so each <image> stays inline next
    to its label (apply_chat_template hoists all <image> tokens to the front,
    which destroys the demo->label binding ICL needs). Returns (prompt, images).
    """
    valid = _valid_labels_str(candidate, label_map)
    parts = [(f"You are a few-shot image classifier. Valid labels: {valid}. "
              + ("Each example below is an image followed by its label. "
                 "Use them to classify the final image. " if support else "")
              + "Answer with ONLY the label letter.\n")]
    images = []
    for txt, img in _support_pairs(support, label_map, names, anonymize):
        parts.append("<image>")            # image first ...
        parts.append(f" label = {txt}\n")  # ... then its label, inline
        images.append(img)
    parts.append("<image> label =")
    images.append(_pil_rgb(q_img))
    body = "".join(parts)
    prompt = f"<|im_start|>user\n{body}<|im_end|>\n<|im_start|>assistant\n"
    return prompt, images


def _build_qwen_messages(support, q_img, candidate, label_map, names, anonymize):
    """Qwen2-VL message list; its chat template DOES interleave images inline, so
    we can use the structured content form. Returns (messages, images)."""
    valid = _valid_labels_str(candidate, label_map)
    content = [{"type": "text", "text":
                f"You are a few-shot image classifier. Valid labels: {valid}. "
                + ("Each example is an image followed by its label; use them to "
                   "classify the final image. " if support else "")
                + "Answer with ONLY the label letter."}]
    images = []
    for txt, img in _support_pairs(support, label_map, names, anonymize):
        content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": f"label = {txt}"})
        images.append(img)
    q_img_pil = _pil_rgb(q_img)
    content.append({"type": "image", "image": q_img_pil})
    content.append({"type": "text", "text": "label ="})
    images.append(q_img_pil)
    return [{"role": "user", "content": content}], images


def _build_vlm_query(backend, support, q_img, candidate, label_map, names,
                     anonymize, processor):
    """Return (base_prompt, images) for one query, where base_prompt ends exactly
    at the point a single-letter label answer would begin (after 'label =')."""
    if backend == 'qwen2vl':
        messages, images = _build_qwen_messages(
            support, q_img, candidate, label_map, names, anonymize)
        base = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    else:  # llava — manual inline-<image> prompt
        base, images = _build_llava_prompt(
            support, q_img, candidate, label_map, names, anonymize, processor)
    return base, images


def _first_token_ids(processor, label_map, candidate):
    """The single (first) token id of each candidate's label letter, as it
    appears after 'label = '. Scoring the FIRST token only keeps cost O(1) per
    query and removes any length bias (all labels score one token at one
    position). The leading space matches the ' <letter>' continuation."""
    tok = processor.tokenizer
    ids = []
    for c in candidate:
        enc = tok.encode(' ' + label_map[c], add_special_tokens=False)
        ids.append(enc[0] if enc else tok.unk_token_id)
    return ids


def _score_candidates(model, processor, base_prompt, images, cand_token_ids, device):
    """ONE forward pass. Read the next-token log-prob at the final position of
    base_prompt for each candidate's first label token. Returns a numpy array of
    log-probs (one per candidate). No per-candidate forward, no length bias, no
    prefix-tokenization assumption (we never append candidate text)."""
    import torch
    import torch.nn.functional as F

    inp = processor(images=images, text=base_prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        logits = model(**inp).logits[0, -1]  # next-token logits at the last position
    logprobs = F.log_softmax(logits, dim=-1)
    return np.array([logprobs[tid].item() for tid in cand_token_ids], dtype=np.float64)


def _score_candidates_batch(model, processor, prompts, per_images, cand_token_ids, device):
    """Batched analogue of _score_candidates: B queries (identical demo prefix,
    different query image) in one padded forward. Reads each row's last
    non-padded position. Returns [B, n_candidates] log-probs.

    Numerically identical to looping _score_candidates (verified: 0.0 logit
    diff) but amortizes the demo-token compute across the batch on CUDA/A100
    (no speedup on MPS, where batch_size=1 is the right setting). Robust to
    left/right padding via the attention mask."""
    import torch
    import torch.nn.functional as F

    inp = processor(text=prompts, images=per_images, return_tensors='pt', padding=True).to(device)
    with torch.no_grad():
        logits = model(**inp).logits  # [B, T, vocab]
    mask = inp['attention_mask']
    out = np.empty((len(prompts), len(cand_token_ids)), dtype=np.float64)
    for b in range(len(prompts)):
        nz = mask[b].nonzero()
        last = int(nz[-1]) if len(nz) else logits.shape[1] - 1  # last real token
        lp = F.log_softmax(logits[b, last], dim=-1)
        out[b] = [lp[tid].item() for tid in cand_token_ids]
    return out


def _evaluate_generative_vlm_backend(
    backend: str,
    dataset: ContinualDataset,
    task: _TaskData,
    k: int,
    seed: int,
    class_names: Optional[List[str]],
    device: str,
    max_queries: int = 50,
    anonymize: bool = False,
    batch_size: int = 1,
) -> Tuple[float, float, Dict[int, Tuple[int, int]]]:
    """Score each query by the model's next-token log-prob over the candidate
    single-letter labels, giving a real softmax distribution. Accuracy = argmax;
    loss = mean cross-entropy = mean(-log p(true_class)).

    Labels are single-token letters assigned by a seeded random permutation
    (see _build_label_map), so the emitted label is decoupled from the image's
    visual identity (anonymization bites even on MNIST) and every candidate is
    one token at one position (no length bias, O(1) scoring per query).

    ``batch_size`` queries are scored per forward. >1 amortizes the (dominant)
    demo-token compute across the batch on CUDA/A100; on MPS keep it at 1.
    Results are numerically identical regardless of batch_size."""
    candidate = list(range(task.class_lo, task.class_hi))
    names = {c: (class_names[c] if class_names and c < len(class_names) else str(c))
             for c in candidate}
    label_map = _build_label_map(candidate, seed)

    model, processor = _load_hf_vlm(backend, device)
    cand_token_ids = _first_token_ids(processor, label_map, candidate)
    support = _sample_support(task, k, seed)

    q_inputs, q_labels = _stack_batches(task.test_loader)
    rng = np.random.RandomState(seed)
    sel = rng.choice(len(q_labels), size=min(max_queries, len(q_labels)), replace=False)

    correct = 0
    ce_sum = 0.0
    per_class: Dict[int, List[int]] = {c: [0, 0] for c in candidate}

    def _record(true, logps):
        nonlocal correct, ce_sum
        probs = np.exp(logps - logps.max())
        probs /= probs.sum()
        pred = candidate[int(probs.argmax())]
        p_true = float(probs[candidate.index(true)]) if true in candidate else 1e-9
        ce_sum += -np.log(max(p_true, 1e-9))
        if true in per_class:
            per_class[true][1] += 1
            if pred == true:
                correct += 1

    bs = max(1, int(batch_size))
    for start in range(0, len(sel), bs):
        chunk = sel[start:start + bs]
        prompts, per_images, trues = [], [], []
        for i in chunk:
            q_img = _to_uint8_rgb(q_inputs[int(i):int(i) + 1])[0]
            base, images = _build_vlm_query(
                backend, support, q_img, candidate, label_map, names, anonymize, processor)
            prompts.append(base); per_images.append(images); trues.append(int(q_labels[int(i)]))
        try:
            if len(prompts) == 1:
                logps_b = _score_candidates(
                    model, processor, prompts[0], per_images[0], cand_token_ids, device)[None]
            else:
                logps_b = _score_candidates_batch(
                    model, processor, prompts, per_images, cand_token_ids, device)
        except Exception as e:
            logging.warning(f"[{backend}] scoring failed: {e}")
            logps_b = np.zeros((len(prompts), len(candidate)))  # uniform -> chance, finite CE
        for true, logps in zip(trues, logps_b):
            _record(true, logps)

    n = len(sel)
    accuracy = correct / n * 100.0 if n else 0.0
    loss = ce_sum / n if n else 0.0  # mean candidate-set cross-entropy
    per_class_out = {c: (v[0], v[1]) for c, v in per_class.items() if v[1] > 0}
    return accuracy, loss, per_class_out


# --------------------------------------------------------------------------- #
# Public entry point.
# --------------------------------------------------------------------------- #

# Process-level cache so the (expensive) foundation model is loaded once per run.
_BACKEND_CACHE: Dict[str, _FeatureBackend] = {}


def _get_class_names(dataset: ContinualDataset) -> Optional[List[str]]:
    try:
        names = dataset.get_class_names()
        return list(names) if names is not None else None
    except Exception as e:
        logging.warning(f"Could not get class names: {e}")
        return None


def evaluate_in_context(
    dataset: ContinualDataset,
    task_id: int,
    k: int,
    backend: str,
    seed: int = 0,
    device: str = 'cpu',
    vlm_max_queries: int = 50,
    anonymize: bool = False,
    vlm_batch_size: int = 1,
) -> Tuple[float, float, Dict[int, float]]:
    """
    Evaluate one (task_id, k, seed) in-context and return
    ``(accuracy_percent, loss, per_class_accuracy_fraction)``.

    ``per_class_accuracy_fraction`` mirrors ``utils.few_shot.evaluate_per_digit``
    ({class_id: acc in [0,1]}) for the per-digit metadata columns.

    ``anonymize`` (generative-VLM backends only) replaces semantic class names
    with abstract indices, forcing the model to learn the label mapping purely
    from the in-context examples — the ablation that isolates true ICL from
    zero-shot recall. Ignored by the feature backends (they never see names as
    answer targets).
    """
    if backend not in ICL_BACKENDS:
        raise ValueError(f"--icl_backend must be one of {ICL_BACKENDS}, got {backend!r}")

    task = _build_task_data(dataset, task_id)
    class_names = _get_class_names(dataset)

    if backend in _LOCAL_BACKENDS:
        be = _BACKEND_CACHE.get(backend)
        if be is None:
            logging.info(f"Loading in-context backend '{backend}' on {device}...")
            be = _make_feature_backend(backend, device)
            _BACKEND_CACHE[backend] = be
        accuracy, loss, per_class = _evaluate_feature_backend(
            be, dataset, task, k, seed, class_names)
    elif backend in _GENERATIVE_VLM_BACKENDS:  # qwen2vl / llava — true ICL, local
        accuracy, loss, per_class = _evaluate_generative_vlm_backend(
            backend, dataset, task, k, seed, class_names, device,
            max_queries=vlm_max_queries, anonymize=anonymize, batch_size=vlm_batch_size)
    else:  # vlm (Claude API)
        accuracy, loss, per_class = _evaluate_vlm_backend(
            dataset, task, k, seed, class_names, max_queries=vlm_max_queries)

    per_class_acc = {c: (corr / tot if tot else 0.0) for c, (corr, tot) in per_class.items()}
    tag = f"{backend}{'-anon' if anonymize else ''}"
    logging.info(f"[icl:{tag}] task={task_id} k={k} seed={seed}: "
                 f"acc={accuracy:.2f}% loss={loss:.4f}")
    return accuracy, loss, per_class_acc
