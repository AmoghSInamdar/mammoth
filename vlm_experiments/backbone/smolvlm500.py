# Copyright 2026-present, Amogh Inamdar, Vici Milenia, Richard Zemel.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
SmolVLM-500M as a Mammoth backbone (``--backbone smolvlm-500m``).

The mid-size SmolVLM: SigLIP-B/16-512 tower + SmolLM2-360M decoder, hidden
960. Defaults to the base checkpoint; pass ``--vlm_model_id
HuggingFaceTB/SmolVLM-500M-Instruct`` to use the instruct model instead.
"""

from backbone import register_backbone
from vlm_experiments.backbone.hf_vlm import DEFAULT_PROMPT, FreezeMode, HuggingFaceVLM

SMOLVLM_500M_BASE = 'HuggingFaceTB/SmolVLM-500M-Base'


@register_backbone("smolvlm-500m")
def smolvlm_500m(num_classes: int,
                 vlm_model_id: str = SMOLVLM_500M_BASE,
                 vlm_prompt: str = DEFAULT_PROMPT,
                 vlm_data_norm: str = 'imagenet',
                 vlm_dtype: str = 'float32',
                 vlm_freeze: FreezeMode = 'none',
                 vlm_grad_ckpt: int = 0) -> HuggingFaceVLM:
    """Build SmolVLM-500M as a registered Mammoth backbone. The ``vlm_*``
    kwargs are exposed as CLI flags when this backbone is selected
    (utils.args.add_dynamic_parsable_args).

    Args:
        num_classes: head width (filled from dataset.N_CLASSES).
        vlm_model_id: HF hub repo id.
        vlm_prompt: classification prompt (image token prepended if absent).
        vlm_data_norm: dataset normalization to undo ('imagenet', 'none',
            'm1,m2,m3/s1,s2,s3').
        vlm_dtype: 'float32' (train default), 'bfloat16', 'float16'.
        vlm_freeze: 'none', 'vision', 'backbone' (linear probe).
        vlm_grad_ckpt: 1 = gradient checkpointing.

    Returns:
        a ready HuggingFaceVLM.
    """
    return HuggingFaceVLM(model_id=vlm_model_id, num_classes=num_classes,
                          prompt=vlm_prompt, data_norm=vlm_data_norm, dtype=vlm_dtype,
                          freeze=vlm_freeze, grad_ckpt=bool(vlm_grad_ckpt))
