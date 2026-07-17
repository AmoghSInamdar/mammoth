# Copyright 2026-present, Amogh Inamdar, Vici Milenia, Richard Zemel.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
VLM backbones for k-shot / SAUCE experiments.

Each model = standard registered Mammoth backbone (images in, class scores
out): training loop, --savecheck, eval_checkpoints.py k-shot adaptation,
SAUCE/plot machinery apply unchanged. ``backbone/smolvlm.py`` import this
package at start-up, triggering registrations:

- ``smolvlm-256m`` (default ``HuggingFaceTB/SmolVLM-256M-Base``)
- ``smolvlm-500m`` (default ``HuggingFaceTB/SmolVLM-500M-Base``)
- ``qwen2vl-2b`` (default ``Qwen/Qwen2-VL-2B``)
- ``qwen2vl-7b`` (default ``Qwen/Qwen2-VL-7B``)
- ``openflamingo-9b`` (default ``openflamingo/OpenFlamingo-9B-vitl-mpt7b``)

Shared HF wrapper (classifier + generative modes): ``hf_vlm.HuggingFaceVLM``.
Qwen packs pixels and image features differently and overrides the few methods
that covers: ``qwenvl.QwenVL``. OpenFlamingo also comes from its own library
rather than the HF hub: ``openflamingo.OpenFlamingo``.
"""

from vlm_experiments.backbone.hf_vlm import DEFAULT_PROMPT, HuggingFaceVLM
from vlm_experiments.backbone.openflamingo import OpenFlamingo
from vlm_experiments.backbone.qwenvl import QwenVL
from vlm_experiments.backbone import (smolvlm256, smolvlm500,  # noqa: F401  (register)
                                      qwenvl, openflamingo)
