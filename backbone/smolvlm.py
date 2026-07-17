# Copyright 2026-present, Amogh Inamdar, Vici Milenia, Richard Zemel.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Registration shim: ``backbone/__init__.py`` auto-imports only this folder, so
this file pulls in the VLM backbones implemented under
``vlm_experiments/backbone/`` (registering ``smolvlm-256m``, ``smolvlm-500m``,
``qwen2vl-2b``, ``qwen2vl-7b`` and ``openflamingo-9b`` as ``--backbone``
choices). ``transformers`` and ``open_flamingo`` are only imported when one of
them is actually built.
"""

import logging

try:
    import vlm_experiments.backbone  # noqa: F401  (registers the VLM backbones)
except ImportError as e:
    logging.warning(f"Could not register VLM backbones from vlm_experiments/: {e}")
