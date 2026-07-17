# Copyright 2026-present, Amogh Inamdar, Vici Milenia, Richard Zemel.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Qwen2-VL as a Mammoth backbone (``--backbone qwen2vl-2b`` / ``qwen2vl-7b``).

Same wrapper as the SmolVLM backbones (HuggingFaceVLM: prompt, pixel bridge,
classifier head, generative scoring, checkpoint format), overriding only what
differs by model family.

Both sizes default to the base (non-instruct) checkpoint, the natural fine-tune
start. Qwen2.5-VL is instruct-only so it is not offered as a base option, but
the class fits it: ``--vlm_model_id Qwen/Qwen2.5-VL-3B-Instruct`` works.

Qwen differs from SmolVLM in three ways: pixels are a flat list of patches plus
an ``image_grid_thw`` rather than a tile; image size comes from a pixel budget
rather than from the model, so ``vlm_image_side`` pins it and every image shares
one prompt template; and there is no image_hidden_states argument, so reusing
the vision tower across candidate names means writing features into the input
embeddings and building the position ids by hand, which ``_vlm_call`` does.
"""

from typing import Optional

import torch

from backbone import register_backbone
from vlm_experiments.backbone.hf_vlm import DEFAULT_PROMPT, FreezeMode, HuggingFaceVLM

QWEN2VL_2B_BASE = 'Qwen/Qwen2-VL-2B'
QWEN2VL_7B_BASE = 'Qwen/Qwen2-VL-7B'

DEFAULT_IMAGE_SIDE = 224


class QwenVL(HuggingFaceVLM):
    """Qwen2-VL / Qwen2.5-VL exposed as Mammoth classification backbone."""

    VISION_MODULES = ('visual',)

    def __init__(self, model_id: str, num_classes: int,
                 image_side: int = DEFAULT_IMAGE_SIDE, **kwargs) -> None:
        """Build the wrapper with the image size pinned before the processor is
        touched. Everything else is left to HuggingFaceVLM.

        Args:
            model_id: HF hub repo id.
            num_classes: classifier head width.
            image_side: square side to pin the processor to; must be a multiple
                of patch_size * merge_size (28).
            kwargs: passed through to HuggingFaceVLM (prompt, data_norm, dtype,
                freeze, grad_ckpt).

        Outputs: none — see HuggingFaceVLM.
        """
        self.image_side = int(image_side)
        super().__init__(model_id=model_id, num_classes=num_classes, **kwargs)

    def _configure_processor(self) -> int:
        """Pin the processor's pixel budget to one square size so every image
        gets the same grid and the prompt template is the same for all of them.
        Qwen has no image splitting to turn off.

        Args:
            none.

        Returns:
            side: dummy image side in pixels.
        """
        img_proc = self.processor.image_processor
        factor = img_proc.patch_size * img_proc.merge_size
        side = self.image_side
        assert side % factor == 0, (
            f"image_side {side} must be a multiple of {factor}")
        n_pix = side * side
        img_proc.min_pixels = img_proc.max_pixels = n_pix
        img_proc.size = {'shortest_edge': n_pix, 'longest_edge': n_pix}
        return side

    def _with_image_token(self, prompt: str) -> str:
        """Put the image token in front of the prompt, wrapped in the vision
        markers Qwen was trained with. The processor expands the image token
        into one placeholder per merged patch.

        Args:
            prompt: classification prompt text.

        Returns:
            prompt: the prompt with a marked image token in it.
        """
        image_token = str(getattr(self.processor, 'image_token', '<|image_pad|>'))
        if image_token in prompt:
            return prompt
        tok, cfg = self.processor.tokenizer, self.vlm.config
        start = tok.decode([cfg.vision_start_token_id])
        end = tok.decode([cfg.vision_end_token_id])
        return f'{start}{image_token}{end}{prompt}'

    def _read_template(self, enc) -> None:
        """Read the patch grid back from the processor output and turn it into
        the pixel size the bridge should resize to. The grid is the same for
        every image because the pixel budget is pinned.

        Args:
            enc: processor output for the dummy image and the prompt.

        Outputs: none — sets ``self._pixel_hw`` and the ``image_grid`` buffer.
        """
        grid = enc['image_grid_thw']
        assert grid.shape[0] == 1, f"Expected one image, got grid {tuple(grid.shape)}"
        patch = self.processor.image_processor.patch_size
        self._pixel_hw = (int(grid[0, 1]) * patch, int(grid[0, 2]) * patch)
        self.register_buffer('image_grid', grid, persistent=False)

    def _patchify(self, pix: torch.Tensor) -> torch.Tensor:
        """Turn normalized images into the flat patch rows Qwen's vision tower
        reads, matching what the processor builds. Differentiable, so it is
        safe at training time.

        Args:
            pix: normalized images [B, 3, S, S].

        Returns:
            patches [B * gh * gw, 3 * temporal_patch_size * patch * patch].
        """
        ip = self.processor.image_processor
        patch, merge, temporal = ip.patch_size, ip.merge_size, ip.temporal_patch_size
        b, c, h, w = pix.shape
        gh, gw = h // patch, w // patch
        x = pix.unsqueeze(1).expand(b, temporal, c, h, w)
        x = x.reshape(b, 1, temporal, c, gh // merge, merge, patch, gw // merge, merge, patch)
        x = x.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
        return x.reshape(b * gh * gw, c * temporal * patch * patch)

    def image_inputs(self, x: torch.Tensor) -> dict:
        """Bridge a dataset-transformed batch into Qwen's image arguments. The
        grid is repeated once per image because they all share one size.

        Args:
            x: dataset-transformed images [B, C, H, W].

        Returns:
            kwargs: pixel_values and image_grid_thw.
        """
        patches = self._patchify(self._normalize_pixels(x)).to(self.vlm_dtype)
        return {'pixel_values': patches,
                'image_grid_thw': self.image_grid.repeat(x.shape[0], 1)}

    def processor_image_inputs(self, enc) -> dict:
        """Pull Qwen's image arguments out of a raw processor output. The grid
        comes from the processor rather than the pinned one, so contexts with
        several images still line up.

        Args:
            enc: processor output.

        Returns:
            kwargs: pixel_values and image_grid_thw.
        """
        return {'pixel_values': enc['pixel_values'].to(self.vlm_dtype),
                'image_grid_thw': enc['image_grid_thw']}

    def image_features(self, image_inputs) -> Optional[torch.Tensor]:
        """Run the vision tower once and stack its per-image outputs. Stacking
        is safe because every image has the same grid.

        Args:
            image_inputs: image arguments, or None.

        Returns:
            features [n_img, n_tok, D], or None when image_inputs is None.
        """
        if image_inputs is None:
            return None
        feats = self.vlm.model.get_image_features(**image_inputs)
        return torch.stack(feats)

    def _vlm_call(self, ids: torch.Tensor, attn: torch.Tensor, image_inputs=None,
                  image_feats=None, labels=None, output_hidden_states: bool = False):
        """Run the whole VLM, taking either raw image arguments or features the
        vision tower already produced. Qwen has no image_hidden_states argument,
        so ready-made features are written into the embeddings at the image
        positions and the grid-dependent position ids are built alongside them.

        Args:
            ids: token ids [N, L].
            attn: attention mask [N, L].
            image_inputs: image arguments from image_inputs(), or None.
            image_feats: row-aligned features from image_features(), or None.
            labels: answer-span labels [N, L] for the training loss, or None.
            output_hidden_states: also return every layer's hidden states.

        Returns:
            the model output: logits, plus loss when labels are given and
            hidden_states when asked for.
        """
        if image_feats is None:
            return self.vlm(input_ids=ids, attention_mask=attn, labels=labels,
                            use_cache=False, output_hidden_states=output_hidden_states,
                            **(image_inputs or {}))

        embeds = self.vlm.get_input_embeddings()(ids)
        mask = (ids == self.vlm.config.image_token_id).unsqueeze(-1).expand_as(embeds)
        flat = image_feats.reshape(-1, image_feats.shape[-1]).to(embeds.dtype)
        embeds = embeds.masked_scatter(mask, flat)
        grid = self.image_grid.repeat(image_feats.shape[0], 1)
        pos, _ = self.vlm.model.get_rope_index(ids, grid, attention_mask=attn)
        return self.vlm(inputs_embeds=embeds, attention_mask=attn, position_ids=pos,
                        labels=labels, use_cache=False,
                        output_hidden_states=output_hidden_states)


@register_backbone("qwen2vl-2b")
def qwen2vl_2b(num_classes: int,
               vlm_model_id: str = QWEN2VL_2B_BASE,
               vlm_prompt: str = DEFAULT_PROMPT,
               vlm_data_norm: str = 'imagenet',
               vlm_dtype: str = 'float32',
               vlm_freeze: FreezeMode = 'none',
               vlm_grad_ckpt: int = 0,
               vlm_image_side: int = DEFAULT_IMAGE_SIDE) -> QwenVL:
    """Build Qwen2-VL-2B as a registered Mammoth backbone. The ``vlm_*`` kwargs
    are exposed as CLI flags when this backbone is selected
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
        vlm_image_side: square side to pin the processor to (multiple of 28).

    Returns:
        a ready QwenVL.
    """
    return QwenVL(model_id=vlm_model_id, num_classes=num_classes,
                  prompt=vlm_prompt, data_norm=vlm_data_norm, dtype=vlm_dtype,
                  freeze=vlm_freeze, grad_ckpt=bool(vlm_grad_ckpt),
                  image_side=vlm_image_side)


@register_backbone("qwen2vl-7b")
def qwen2vl_7b(num_classes: int,
               vlm_model_id: str = QWEN2VL_7B_BASE,
               vlm_prompt: str = DEFAULT_PROMPT,
               vlm_data_norm: str = 'imagenet',
               vlm_dtype: str = 'float32',
               vlm_freeze: FreezeMode = 'none',
               vlm_grad_ckpt: int = 0,
               vlm_image_side: int = DEFAULT_IMAGE_SIDE) -> QwenVL:
    """Build Qwen2-VL-7B as a registered Mammoth backbone. The ``vlm_*`` kwargs
    are exposed as CLI flags when this backbone is selected
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
        vlm_image_side: square side to pin the processor to (multiple of 28).

    Returns:
        a ready QwenVL.
    """
    return QwenVL(model_id=vlm_model_id, num_classes=num_classes,
                  prompt=vlm_prompt, data_norm=vlm_data_norm, dtype=vlm_dtype,
                  freeze=vlm_freeze, grad_ckpt=bool(vlm_grad_ckpt),
                  image_side=vlm_image_side)
