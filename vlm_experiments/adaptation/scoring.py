# Copyright 2026-present, Amogh Inamdar, Vici Milenia, Richard Zemel.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Shared candidate-scoring helpers for the adaptation modes.

One scorer for every context shape: fixed prompt (gradient), demo prompt (ICL),
and the PRIOR contexts that PMI needs (Holtzman et al. 2021). For the PMI prior
we score each name under the same context but with the query image replaced by
a plain gray image carrying no class information (the image slot stays, the
class evidence is removed; cf. Zhao et al. 2021), then subtract that from the
real score. This cancels out the menu bias of the prompt, and both scores share
the exact same token layout. Also holds the parser for the free-generation
measure.
"""

from typing import List, Optional

import torch
import torch.nn.functional as F


def encode_context(net, text: str, pils: List, device):
    """Run the processor on an arbitrary context: text with zero or more
    inline image tokens + matching PIL list. Empty list = pure text (the
    PMI prior context).

    Args:
        net: HuggingFaceVLM backbone.
        text: prompt text (image tokens inline where images go).
        pils: PIL images matching the image tokens (may be empty).
        device: torch device.

    Returns:
        (input_ids, attention_mask, image_inputs): image_inputs is the backbone's
        image arguments, None when no images.
    """
    if pils:
        enc = net.processor(images=pils, text=text, return_tensors='pt')
        img = {k: v.to(device) for k, v in net.processor_image_inputs(enc).items()}
    else:
        enc = net.processor(text=text, return_tensors='pt')
        img = None
    return enc['input_ids'].to(device), enc['attention_mask'].to(device), img


def gray_pil(net):
    """Query image for the PMI prior: mid-gray at the processor's native tile
    size. Carries no class information, but keeps the token layout of a real
    query.

    Args:
        net: HuggingFaceVLM backbone (supplies the tile size).

    Returns:
        PIL RGB image, uniform (128, 128, 128).
    """
    from PIL import Image
    side = max(net._pixel_hw)
    return Image.new('RGB', (side, side), (128, 128, 128))


def teacher_forced_scores(net, prefix_ids: torch.Tensor, prefix_mask: torch.Tensor,
                          image_hidden: Optional[torch.Tensor], width: int) -> torch.Tensor:
    """Log-probability the model gives to generating each candidate
    `` <name><eos>`` appended to an arbitrary prefix — the classification
    scorer, same math as backbone.score_classes (mean per token when
    net.length_norm). Vision features are computed by the CALLER once and
    reused across candidate chunks; None = text-only context (PMI prior).

    Args:
        net: HuggingFaceVLM backbone (answer ids, pad id, chunk, norm).
        prefix_ids: [1, P] context token ids.
        prefix_mask: [1, P] context attention mask.
        image_hidden: [n_img, S, D] features or None.
        width: score candidates [0, width).

    Returns:
        scores [num_classes] float32; classes >= width get -1e9.
    """
    device = prefix_ids.device
    plen = prefix_ids.shape[1]
    scores = torch.full((net.num_classes,), -1e9, device=device, dtype=torch.float32)
    for lo in range(0, width, net.score_chunk):
        cand = list(range(lo, min(lo + net.score_chunk, width)))
        answers = [net._answer_ids[c] for c in cand]
        lmax = max(a.numel() for a in answers)
        n = len(cand)

        ids = torch.full((n, plen + lmax), net.pad_id, dtype=torch.long, device=device)
        attn = torch.zeros((n, plen + lmax), dtype=torch.long, device=device)
        span = torch.zeros((n, plen + lmax), dtype=torch.bool, device=device)
        ids[:, :plen] = prefix_ids
        attn[:, :plen] = prefix_mask
        for i, a in enumerate(answers):
            a = a.to(device)
            ids[i, plen:plen + a.numel()] = a
            attn[i, plen:plen + a.numel()] = 1
            span[i, plen:plen + a.numel()] = True

        feats = None if image_hidden is None else image_hidden.repeat(n, 1, 1)
        with torch.no_grad():
            logits = net.candidate_logits(feats, ids, attn)
        # Only the answer tokens are ever read: the token at position t is
        # predicted by logits[t-1], and the answers live at
        # ids[:, plen:plen+lmax]. Softmax just that slice, not the whole
        # sequence — a full-length float32 log_softmax on an ICL prompt
        # (~6.7k tokens x 49k vocab x chunk) needs 20+ GiB and OOMs.
        logits = logits[:, plen - 1:plen + lmax - 1]
        logprobs = F.log_softmax(logits.float(), dim=-1)
        tgt = ids[:, plen:plen + lmax]
        mask = span[:, plen:plen + lmax]
        tok_lp = logprobs.gather(2, tgt.unsqueeze(-1)).squeeze(-1) * mask
        s = tok_lp.sum(1)
        if net.length_norm:
            s = s / mask.sum(1).clamp(min=1)
        scores[cand] = s
    return scores


def parse_generation(answer: str, class_names: List[str], menu: List[int]) -> int:
    """Map a generated string to a menu class: clean it up, then check which
    class name it starts with, trying the longest names first and requiring
    the name to end at a word boundary ('maple tree' wins over 'maple';
    'appleton' matches nothing). No match -> -1 (counted wrong).

    Args:
        answer: generated text.
        class_names: label-order class names.
        menu: candidate class ids to match against.

    Returns:
        class id or -1 when unparseable.
    """
    a = answer.strip().lower().split('\n')[0].lstrip(' .,:;!?"\'')
    for c in sorted(menu, key=lambda c: -len(class_names[c])):
        name = class_names[c].replace('_', ' ').lower().strip()
        if a.startswith(name) and (len(a) == len(name) or not a[len(name)].isalnum()):
            return c
    return -1
