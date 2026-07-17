# Copyright 2026-present, Amogh Inamdar, Vici Milenia, Richard Zemel.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
OpenFlamingo as a Mammoth backbone (``--backbone openflamingo-9b``).

Same wrapper as the SmolVLM and Qwen backbones (HuggingFaceVLM: prompt, pixel
bridge, classifier head, generative scoring, checkpoint format). Only the parts
that differ by model family are overridden here.

Four differences drive everything in this file:

- Nothing is loaded from the HF hub as a model. The released file holds only
  the parts OpenFlamingo trained (perceiver, cross-attention layers, language
  embeddings), so the model is rebuilt from the stock vision and language parts
  it was trained on and the file is dropped on top. RELEASES names those parts
  per checkpoint, and open_flamingo hands back a tokenizer and a torchvision
  transform instead of one processor, so _OpenFlamingoProcessor puts them back
  together in the shape the wrapper and the adaptation code expect.
- Images stay whole. There is one image token per image rather than a block of
  placeholders, and pixels arrive as (batch, image, frame, C, H, W); frames are
  for video, so ours is always 1.
- The language model does not take images at all. The cross-attention layers
  read them off a side channel that has to be filled before the call and
  cleared after, which is what _vlm_call does. That side channel is also why
  the wrapper's text-only path cannot work here.
- The vision tower never trains: OpenFlamingo runs it under no_grad. So
  ``--vlm_freeze none`` trains the perceiver, the cross-attention layers, and
  the language embeddings, which is everything OpenFlamingo itself ever trains.

Answers end with the ``<|endofchunk|>`` token instead of the tokenizer's
end-of-text one: that is what OpenFlamingo was trained to close a caption with,
and what its generate stops on.

open_flamingo is imported lazily in _load_backend so backbone registration at
start-up does not need it.
"""

from typing import Optional

import torch

from backbone import register_backbone
from vlm_experiments.backbone.hf_vlm import DEFAULT_PROMPT, FreezeMode, HuggingFaceVLM

OPENFLAMINGO_9B = 'openflamingo/OpenFlamingo-9B-vitl-mpt7b'

IMAGE_TOKEN = '<image>'

# The stock parts each release was trained on. The released file carries only
# the parts OpenFlamingo trained, so these have to match it exactly.
RELEASES = {
    OPENFLAMINGO_9B: {
        'vision': 'ViT-L-14',
        'vision_pretrained': 'openai',
        'lang': 'anas-awadalla/mpt-7b',
        'cross_attn_every_n_layers': 4,
    },
}


class _OpenFlamingoImages:
    """open_clip's image transform with the stats the pixel bridge needs."""

    def __init__(self, transform) -> None:
        """Keep the transform and read its normalization stats off it. The
        bridge applies the same stats itself, so it reads them from the real
        transform rather than assuming CLIP's.

        Args:
            transform: open_clip torchvision transform for one PIL image.

        Outputs: none — sets ``self.transform``, ``self.image_mean``,
            ``self.image_std``.
        """
        from torchvision.transforms import Normalize

        norms = [t for t in transform.transforms if isinstance(t, Normalize)]
        assert norms, "open_clip transform has no Normalize step to read stats from"
        self.transform = transform
        self.image_mean = tuple(norms[-1].mean)
        self.image_std = tuple(norms[-1].std)

    def __call__(self, image) -> torch.Tensor:
        """Turn one PIL image into normalized pixels. Resizing and cropping to
        the vision tower's size happen inside the transform.

        Args:
            image: PIL image.

        Returns:
            pixels [3, S, S].
        """
        return self.transform(image)


class _OpenFlamingoProcessor:
    """OpenFlamingo's tokenizer and image transform behind one HF-style processor."""

    def __init__(self, tokenizer, transform) -> None:
        """Hold the two pieces open_flamingo hands back under the names the
        wrapper and the adaptation code look for. Only the parts they use are
        provided, not the whole HF processor surface.

        Args:
            tokenizer: HF tokenizer, already carrying OpenFlamingo's tokens.
            transform: open_clip torchvision transform for one PIL image.

        Outputs: none — sets ``self.tokenizer``, ``self.image_processor``,
            ``self.image_token``.
        """
        self.tokenizer = tokenizer
        self.image_processor = _OpenFlamingoImages(transform)
        self.image_token = IMAGE_TOKEN

    def __call__(self, images=None, text=None, return_tensors='pt') -> dict:
        """Pack one context the way an HF processor would: token ids for the
        text, pixels for the images its tokens name. The images come out as one
        context of single-frame images, which is the shape Flamingo reads.

        Args:
            images: PIL images in the order their tokens appear, or None.
            text: prompt text with one image token per image.
            return_tensors: kept for the HF call shape; only 'pt' works.

        Returns:
            dict with input_ids, attention_mask, and pixel_values
            [1, n_img, 1, 3, S, S] when images were given.
        """
        assert return_tensors == 'pt', "only torch tensors are supported"
        enc = dict(self.tokenizer(text, return_tensors='pt'))
        if images:
            pixels = torch.stack([self.image_processor(im) for im in images])
            enc['pixel_values'] = pixels[None, :, None]
        return enc


class OpenFlamingo(HuggingFaceVLM):
    """OpenFlamingo exposed as Mammoth classification backbone."""

    VISION_MODULES = ('vision_encoder', 'perceiver')

    def __init__(self, model_id: str, num_classes: int, **kwargs) -> None:
        """Build the wrapper, then point the candidate answers at
        OpenFlamingo's own end token. Everything else is left to
        HuggingFaceVLM.

        Args:
            model_id: HF hub repo id of a release listed in RELEASES.
            num_classes: classifier head width.
            kwargs: passed through to HuggingFaceVLM (prompt, data_norm, dtype,
                freeze, grad_ckpt).

        Outputs: none — see HuggingFaceVLM.
        """
        super().__init__(model_id=model_id, num_classes=num_classes, **kwargs)
        self.eos_id = self.vlm.eoc_token_id

    def _load_backend(self, model_id: str) -> None:
        """Rebuild the model from the stock parts its release was trained on
        and load the released weights over them. The parts must be the exact
        ones named in RELEASES, since the released file holds only what
        OpenFlamingo trained.

        Args:
            model_id: HF hub repo id of a release listed in RELEASES.

        Outputs: none — sets ``self.vlm`` and ``self.processor``.
        """
        from huggingface_hub import hf_hub_download
        from open_flamingo import create_model_and_transforms

        assert model_id in RELEASES, (
            f"Unknown OpenFlamingo checkpoint {model_id!r}: expected one of "
            f"{tuple(RELEASES)}")
        cfg = RELEASES[model_id]
        self.vlm, transform, tokenizer = create_model_and_transforms(
            clip_vision_encoder_path=cfg['vision'],
            clip_vision_encoder_pretrained=cfg['vision_pretrained'],
            lang_encoder_path=cfg['lang'],
            tokenizer_path=cfg['lang'],
            cross_attn_every_n_layers=cfg['cross_attn_every_n_layers'])

        state = torch.load(hf_hub_download(model_id, 'checkpoint.pt'), map_location='cpu')
        self.vlm.load_state_dict(state, strict=False)  # release is a partial model
        self.vlm.to(self.vlm_dtype)
        self.vlm.lang_encoder.config.use_cache = False  # no generation during training
        self.processor = _OpenFlamingoProcessor(tokenizer, transform)

    def _hidden_size(self) -> int:
        """Width of the language model's hidden state. Flamingo works it out at
        build time because the language model can be any family.

        Args:
            none.

        Returns:
            hidden size in features.
        """
        return self.vlm.lang_dim

    def _enable_grad_ckpt(self) -> None:
        """Turn on gradient checkpointing in the perceiver and in the decoder
        layers. Rebuilding the decoder layers keeps the weights, which live in
        the parts those layers wrap.

        Args:
            none.

        Outputs: none — changes ``self.vlm`` in place.
        """
        self.vlm.perceiver._use_gradient_checkpointing = True
        self.vlm.lang_encoder.init_flamingo_layers(True)

    def context_limit(self) -> Optional[int]:
        """How many tokens the language model can take, so callers can warn
        when a prompt runs past it. None when the model does not say.

        Args:
            none.

        Returns:
            token limit, or None when unknown.
        """
        return getattr(self.vlm.lang_encoder.config, 'max_seq_len', None)

    def _configure_processor(self) -> int:
        """Report the size the vision tower reads. There is nothing to pin: the
        transform already resizes and crops every image to that one size.

        Args:
            none.

        Returns:
            side: dummy image side in pixels.
        """
        side = self.vlm.vision_encoder.image_size
        return int(side[0] if isinstance(side, (tuple, list)) else side)

    def _read_template(self, enc) -> None:
        """Read the image size back from the processed dummy image. Checks it
        came out as one single-frame image, which is what a query looks like.

        Args:
            enc: processor output for the dummy image and the prompt.

        Outputs: none — sets ``self._pixel_hw``.
        """
        pix = enc['pixel_values']
        assert pix.dim() == 6 and tuple(pix.shape[1:3]) == (1, 1), (
            f"Expected one single-frame image, got pixel_values shape "
            f"{tuple(pix.shape)}")
        self._pixel_hw = tuple(pix.shape[-2:])

    def _to_pixel_space(self, x: torch.Tensor) -> torch.Tensor:
        """Bridge a dataset-transformed tensor into Flamingo's pixel space: one
        context per row, each holding one single-frame image.

        Args:
            x: dataset-transformed images [B, C, H, W] (C = 1 or 3).

        Returns:
            vision_x [B, 1, 1, 3, S, S] in the VLM dtype.
        """
        return self._normalize_pixels(x)[:, None, None].to(self.vlm_dtype)

    def image_inputs(self, x: torch.Tensor) -> dict:
        """Bridge a dataset-transformed batch into Flamingo's image argument.
        One image per row, so every row is its own context.

        Args:
            x: dataset-transformed images [B, C, H, W].

        Returns:
            kwargs: vision_x.
        """
        return {'vision_x': self._to_pixel_space(x)}

    def processor_image_inputs(self, enc) -> dict:
        """Pull Flamingo's image argument out of a raw processor output. The
        processor already packs a context's images the way the model wants.

        Args:
            enc: processor output.

        Returns:
            kwargs: vision_x.
        """
        return {'vision_x': enc['pixel_values'].to(self.vlm_dtype)}

    def image_features(self, image_inputs) -> Optional[torch.Tensor]:
        """Run the vision tower and the perceiver once so their output can be
        reused across many candidate names. Images stack along the first
        dimension, matching what the other families return.

        Args:
            image_inputs: image arguments, or None.

        Returns:
            features [n_img, n_latents, vis_dim], or None when image_inputs is
            None.
        """
        if image_inputs is None:
            return None
        vision_x = image_inputs['vision_x']
        b, t, f = vision_x.shape[:3]
        flat = vision_x.reshape(b * t * f, *vision_x.shape[3:])
        # OpenFlamingo trains with the tower frozen and runs it under no_grad,
        # so it never carries gradients here either.
        with torch.no_grad():
            tokens = self.vlm.vision_encoder(flat)[1]
        tokens = tokens.reshape(b, t, f, *tokens.shape[1:])
        feats = self.vlm.perceiver(tokens)
        return feats.reshape(b * t, *feats.shape[2:])

    def _vlm_call(self, ids: torch.Tensor, attn: torch.Tensor, image_inputs=None,
                  image_feats=None, labels=None, output_hidden_states: bool = False):
        """Run the language model on token rows with the images hung on the
        side channel its cross-attention layers read. The channel is filled
        before the call and cleared after, so nothing leaks into the next one.

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
        feats = image_feats if image_feats is not None else self.image_features(image_inputs)
        assert feats is not None, (
            "OpenFlamingo has no text-only path: every context needs an image")

        lang = self.vlm.lang_encoder
        vis = feats.reshape(ids.shape[0], -1, *feats.shape[-2:])
        for layer in lang._get_decoder_layers():
            layer.condition_vis_x(vis)
        # The language model finds the image tokens in ids on its own.
        out = lang(input_ids=ids, attention_mask=attn, labels=labels,
                   use_cache=False, output_hidden_states=output_hidden_states)
        lang.clear_conditioned_layers()
        return out

    def generate_ids(self, ids: torch.Tensor, attn: torch.Tensor, image_inputs,
                     max_new_tokens: int) -> torch.Tensor:
        """Let the model write an answer after the given context, greedily.
        Flamingo takes the token rows as lang_x and runs the vision tower
        itself.

        Args:
            ids: context token ids [N, L].
            attn: attention mask [N, L].
            image_inputs: image arguments from image_inputs() or
                processor_image_inputs().
            max_new_tokens: generation cap.

        Returns:
            token ids [N, L + generated], context included.
        """
        return self.vlm.generate(lang_x=ids, attention_mask=attn,
                                 max_new_tokens=max_new_tokens, do_sample=False,
                                 use_cache=True, pad_token_id=self.pad_id,
                                 **image_inputs)


@register_backbone("openflamingo-9b")
def openflamingo_9b(num_classes: int,
                    vlm_model_id: str = OPENFLAMINGO_9B,
                    vlm_prompt: str = DEFAULT_PROMPT,
                    vlm_data_norm: str = 'imagenet',
                    vlm_dtype: str = 'float32',
                    vlm_freeze: FreezeMode = 'none',
                    vlm_grad_ckpt: int = 0) -> OpenFlamingo:
    """Build OpenFlamingo-9B as a registered Mammoth backbone. The ``vlm_*``
    kwargs are exposed as CLI flags when this backbone is selected
    (utils.args.add_dynamic_parsable_args).

    Args:
        num_classes: head width (filled from dataset.N_CLASSES).
        vlm_model_id: HF hub repo id of a release listed in RELEASES.
        vlm_prompt: classification prompt (image token prepended if absent).
        vlm_data_norm: dataset normalization to undo ('imagenet', 'none',
            'm1,m2,m3/s1,s2,s3').
        vlm_dtype: 'float32' (train default), 'bfloat16', 'float16'.
        vlm_freeze: 'none' (perceiver + cross-attention + language embeddings,
            all OpenFlamingo itself ever trains), 'vision' (also freeze the
            perceiver), 'backbone' (freeze the whole VLM: linear probe).
        vlm_grad_ckpt: 1 = gradient checkpointing.

    Returns:
        a ready OpenFlamingo.
    """
    return OpenFlamingo(model_id=vlm_model_id, num_classes=num_classes,
                        prompt=vlm_prompt, data_norm=vlm_data_norm, dtype=vlm_dtype,
                        freeze=vlm_freeze, grad_ckpt=bool(vlm_grad_ckpt))
