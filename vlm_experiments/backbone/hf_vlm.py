# Copyright 2026-present, Amogh Inamdar, Vici Milenia, Richard Zemel.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
HuggingFace VLM as a Mammoth backbone.

Wraps any HF image-text-to-text model (tested on the SmolVLM / Idefics3
family) as a standard MammothBackbone: image tensor in, class logits out. The
VLM takes the place of the ResNet-18/ViT backbone, so the existing pipeline
works unchanged: training loop, checkpoint save/load, and k-shot eval. The
template and normalization buffers are non-persistent and rebuilt at init, so
checkpoints are weights-only and load cleanly.

Everything that does not depend on the model family lives here. The few places
that do are small overridable methods, so a new family only has to say how it
loads, how it packs pixels, and how it takes precomputed image features.
``qwenvl.QwenVL`` is a worked example that keeps the HF loading;
``openflamingo.OpenFlamingo`` is one that replaces it with another library.

Forward pass: dataset tensor -> pixel bridge -> vision tower + connector + LM
under a fixed prompt -> last-token hidden state -> IncrementalClassifier head.
The last token sees the whole image under causal attention. The default prompt
ends "This is a photo of a" to point the last position toward the class name.

Pixel bridge: the loader tensor is already normalized with dataset stats
(data_norm; 'imagenet' for the *-224 datasets, 'none' for MNIST variants). The
bridge undoes that, clamps to [0, 1], resizes to the processor's tile size,
and applies the model's image stats. This matches the HF processor output for
one square un-split image. The token template is built once at init by running
the real processor on a dummy image, so the placeholder count and pixel grid
always agree.

Generative mode (enable_generative; used by models/vlm_sgd.py): training uses
a masked language-model loss on `` <classname><eos>`` after the prompt, with
the whole prompt masked out; forward returns a log-prob per candidate name.
The vision tower runs once per batch and its features are reused across
candidates. Same weights, bridge, and checkpoint format as classifier mode;
the head is unused.

transformers is imported lazily in _load_backend so backbone registration at
start-up does not need transformers.
"""

from typing import List, Literal, Tuple

import torch
import torch.nn.functional as F

from backbone import MammothBackbone
from backbone.utils.layers import IncrementalClassifier

# The image token is prepended if the prompt does not already contain it.
DEFAULT_PROMPT = 'This is a photo of a'

# Un-normalization presets for the dataset -> pixel bridge.
NORM_PRESETS = {
    'imagenet': ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    'none': ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
}

DTYPES = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}

FreezeMode = Literal['none', 'vision', 'backbone']


def _parse_norm(spec: str) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    """Turn a normalization spec into per-channel (mean, std) for the pixel
    bridge. The spec is either a preset name or an explicit stats string.

    Args:
        spec: 'imagenet', 'none', or 'm1,m2,m3/s1,s2,s3' (1 or 3 values;
            grayscale broadcast to 3 channels).

    Returns:
        (mean, std): two 3-tuples of floats.
    """
    if spec in NORM_PRESETS:
        return NORM_PRESETS[spec]
    try:
        mean_s, std_s = spec.split('/')
        mean = tuple(float(v) for v in mean_s.split(','))
        std = tuple(float(v) for v in std_s.split(','))
        assert len(mean) == len(std) and len(mean) in (1, 3)
        if len(mean) == 1:
            mean, std = mean * 3, std * 3
        return mean, std
    except (ValueError, AssertionError):
        raise ValueError(
            f"Invalid data_norm {spec!r}: expected {tuple(NORM_PRESETS)} or 'm1,m2,m3/s1,s2,s3'")


class HuggingFaceVLM(MammothBackbone):
    """HuggingFace VLM exposed as Mammoth classification backbone."""

    VISION_MODULES = ('vision_model', 'connector')

    def __init__(self, model_id: str, num_classes: int, prompt: str = DEFAULT_PROMPT,
                 data_norm: str = 'imagenet', dtype: str = 'float32',
                 freeze: str = 'none', grad_ckpt: bool = False) -> None:
        """Load the HF model and processor and assemble the classification
        wrapper. Precomputes the constant token template and bridge stats (as
        non-persistent buffers) by running the real processor once on a dummy
        image.

        Args:
            model_id: HF hub repo id (e.g. 'HuggingFaceTB/SmolVLM-256M-Base').
            num_classes: classifier head width (dataset.N_CLASSES).
            prompt: text after the image; '<image>' prepended if missing.
            data_norm: the normalization the dataset applied to incoming
                tensors ('imagenet', 'none', or 'm1,m2,m3/s1,s2,s3'); the
                bridge undoes it.
            dtype: VLM weight/compute dtype. 'float32' for training (the loop
                has no GradScaler); 'bfloat16' is fine when frozen.
            freeze: 'none' (full fine-tune), 'vision' (freeze tower +
                connector), 'backbone' (freeze the whole VLM: linear probe).
            grad_ckpt: enable gradient checkpointing on the VLM (trades compute
                for memory).

        Outputs: none — sets ``self.vlm``, ``self.processor``,
            ``self.classifier``, and the template/normalization buffers.
        """
        super().__init__()
        from PIL import Image

        assert dtype in DTYPES, f"dtype must be one of {tuple(DTYPES)}"
        self.model_id = model_id
        self.num_classes = num_classes
        self.vlm_dtype = DTYPES[dtype]

        self._load_backend(model_id)

        side = self._configure_processor()
        self.prompt = prompt = self._with_image_token(prompt)

        dummy = Image.new('RGB', (side, side))
        enc = self.processor(images=[dummy], text=prompt, return_tensors='pt')
        self._read_template(enc)

        # Non-persistent because they are rebuilt at init: keeps checkpoints
        # weights-only and loading strict.
        self.register_buffer('prompt_ids', enc['input_ids'], persistent=False)
        self.register_buffer('prompt_mask', enc['attention_mask'], persistent=False)

        data_mean, data_std = _parse_norm(data_norm)
        img_proc = self.processor.image_processor
        pix_mean, pix_std = img_proc.image_mean, img_proc.image_std
        for name, vals in (('data_mean', data_mean), ('data_std', data_std),
                           ('pix_mean', pix_mean), ('pix_std', pix_std)):
            self.register_buffer(name, torch.tensor(vals).view(1, 3, 1, 1), persistent=False)

        self.embed_dim = self._hidden_size()
        self.classifier = IncrementalClassifier(self.embed_dim, num_classes)

        # SmolVLM-Base uses the GPT2 tokenizer, which has no pad token; the eos
        # id is a safe fallback since padded positions get attention 0 and
        # labels -100.
        tok = self.processor.tokenizer
        self.pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        self.eos_id = tok.eos_token_id

        # Generative mode is off by default; models/vlm_sgd.py turns it on.
        self.generative = False
        self.length_norm = True
        self.score_chunk = 16
        self.n_active = num_classes
        self._answer_ids = None  # per-class ' <name><eos>' token ids

        self._apply_freeze(freeze)
        if grad_ckpt:
            self._enable_grad_ckpt()

    def _load_backend(self, model_id: str) -> None:
        """Load the model and the processor that packs its inputs, and turn off
        the generation cache. Families that do not come from the HF hub replace
        this whole method.

        Args:
            model_id: HF hub repo id.

        Outputs: none — sets ``self.vlm`` and ``self.processor``.
        """
        from transformers import AutoProcessor
        try:
            from transformers import AutoModelForImageTextToText as _AutoVLM
        except ImportError:
            from transformers import AutoModelForVision2Seq as _AutoVLM

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.vlm = _AutoVLM.from_pretrained(model_id, dtype=self.vlm_dtype)
        self.vlm.config.use_cache = False  # no generation during training

    def _hidden_size(self) -> int:
        """Width of the language model's hidden state, which is what the
        classifier head reads. Families that store it elsewhere override this.

        Args:
            none.

        Returns:
            hidden size in features.
        """
        return self.vlm.config.text_config.hidden_size

    def _enable_grad_ckpt(self) -> None:
        """Turn on gradient checkpointing, trading compute for memory.
        Families with their own switch override this.

        Args:
            none.

        Outputs: none — changes ``self.vlm`` in place.
        """
        self.vlm.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={'use_reentrant': False})

    def context_limit(self):
        """How many tokens the language model can take, so callers can warn
        when a prompt runs past it. None when the model does not say.

        Args:
            none.

        Returns:
            token limit, or None when unknown.
        """
        return getattr(self.vlm.config.text_config, 'max_position_embeddings', None)

    def _configure_processor(self) -> int:
        """Pin the processor to one square un-split tile per image, so the
        bridge can reproduce its output exactly. Turns image splitting off and
        reports the tile size the dummy image should use.

        Args:
            none.

        Returns:
            side: dummy image side in pixels.
        """
        img_proc = self.processor.image_processor
        if hasattr(img_proc, 'do_image_splitting'):
            img_proc.do_image_splitting = False

        side = 512
        max_size = getattr(img_proc, 'max_image_size', None) or getattr(img_proc, 'size', None)
        if isinstance(max_size, dict):
            side = max_size.get('longest_edge', max_size.get('height', side))
        return side

    def _with_image_token(self, prompt: str) -> str:
        """Put the image token in front of the prompt when it is not already
        there. The processor expands that token into the image placeholders.

        Args:
            prompt: classification prompt text.

        Returns:
            prompt: the prompt with an image token in it.
        """
        image_token = str(getattr(self.processor, 'image_token', '<image>'))
        return prompt if image_token in prompt else image_token + prompt

    def _read_template(self, enc) -> None:
        """Read the tile size back from the processor output for the dummy
        image. Checks the image really was left as one un-split tile.

        Args:
            enc: processor output for the dummy image and the prompt.

        Outputs: none — sets ``self._pixel_hw``.
        """
        assert enc['pixel_values'].dim() == 5 and enc['pixel_values'].shape[1] == 1, (
            f"Expected single image tile, got pixel_values shape "
            f"{tuple(enc['pixel_values'].shape)}; image splitting really off?")
        self._pixel_hw = tuple(enc['pixel_values'].shape[-2:])

    def _apply_freeze(self, freeze: str) -> None:
        """Turn off requires_grad on the parts the chosen mode excludes from
        training. The classifier head always stays trainable.

        Args:
            freeze: 'none' (train all), 'vision' (freeze tower + connector),
                'backbone' (freeze the whole VLM).

        Outputs: none — flips requires_grad in place on ``self.vlm``.
        """
        assert freeze in ('none', 'vision', 'backbone'), f"Unknown freeze mode: {freeze}"
        if freeze == 'backbone':
            self.vlm.requires_grad_(False)
        elif freeze == 'vision':
            base = getattr(self.vlm, 'model', self.vlm)
            for attr in self.VISION_MODULES:
                mod = getattr(base, attr, None)
                if mod is not None:
                    mod.requires_grad_(False)

    def _normalize_pixels(self, x: torch.Tensor) -> torch.Tensor:
        """Undo the dataset normalization, resize to the tile size, and apply
        the model's own image stats. All operations are differentiable, so this
        is safe at training time.

        Args:
            x: dataset-transformed images [B, C, H, W] (C = 1 or 3).

        Returns:
            pixels [B, 3, S, S] float32; (S, S) is the tile size captured from
            the processor at init.
        """
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x = x.float() * self.data_std + self.data_mean  # undo dataset norm
        x = x.clamp(0.0, 1.0)
        if x.shape[-2:] != self._pixel_hw:
            x = F.interpolate(x, size=self._pixel_hw, mode='bicubic', align_corners=False)
        return (x - self.pix_mean) / self.pix_std

    def _to_pixel_space(self, x: torch.Tensor) -> torch.Tensor:
        """Bridge a dataset-transformed tensor into VLM pixel space,
        reproducing the HF processor output for one square un-split image. Kept
        as its own method because some families pack pixels differently.

        Args:
            x: dataset-transformed images [B, C, H, W] (C = 1 or 3).

        Returns:
            pixel_values [B, 1, 3, S, S] in the VLM dtype.
        """
        return self._normalize_pixels(x).unsqueeze(1).to(self.vlm_dtype)

    def image_inputs(self, x: torch.Tensor) -> dict:
        """Bridge a dataset-transformed batch into the image arguments the VLM
        forward expects. Families that need more than pixel_values add their
        own keys here.

        Args:
            x: dataset-transformed images [B, C, H, W].

        Returns:
            kwargs: image arguments to pass straight to the VLM.
        """
        return {'pixel_values': self._to_pixel_space(x)}

    def processor_image_inputs(self, enc) -> dict:
        """Pull the image arguments out of a raw processor output. Used by the
        adaptation code, which runs the processor on PIL images instead of
        going through the bridge.

        Args:
            enc: processor output.

        Returns:
            kwargs: image arguments to pass straight to the VLM.
        """
        return {'pixel_values': enc['pixel_values'].to(self.vlm_dtype)}

    def image_features(self, image_inputs) -> torch.Tensor:
        """Run the vision tower once so its output can be reused across many
        candidate names. None passes through for a text-only context.

        Args:
            image_inputs: image arguments, or None.

        Returns:
            features [n_img, S, D], or None when image_inputs is None.
        """
        if image_inputs is None:
            return None
        base = getattr(self.vlm, 'model', None)
        assert base is not None and hasattr(base, 'get_image_features'), (
            "need base model with get_image_features")
        return base.get_image_features(**image_inputs)

    def _vlm_call(self, ids: torch.Tensor, attn: torch.Tensor, image_inputs=None,
                  image_feats=None, labels=None, output_hidden_states: bool = False):
        """Run the whole VLM on token rows plus images, taking either raw image
        arguments or features the vision tower already produced. This is the one
        place the family's forward signature is spelled out.

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
        kwargs = dict(image_inputs or {})
        if image_feats is not None:
            kwargs['image_hidden_states'] = image_feats
        return self.vlm(input_ids=ids, attention_mask=attn, labels=labels,
                        use_cache=False, output_hidden_states=output_hidden_states,
                        **kwargs)

    def candidate_logits(self, image_feats, ids: torch.Tensor,
                         attn: torch.Tensor) -> torch.Tensor:
        """Run the language model over candidate rows using image features that
        were computed once, so the vision tower does not run per candidate. The
        features must already be lined up one entry per image per row.

        Args:
            image_feats: row-aligned features, or None for a text-only context.
            ids: token ids [N, L].
            attn: attention mask [N, L].

        Returns:
            logits [N, L, vocab].
        """
        return self._vlm_call(ids, attn, image_feats=image_feats).logits

    def generate_ids(self, ids: torch.Tensor, attn: torch.Tensor, image_inputs,
                     max_new_tokens: int) -> torch.Tensor:
        """Let the model write an answer after the given context, greedily.
        Used by the adaptation code's generate-and-parse measure.

        Args:
            ids: context token ids [N, L].
            attn: attention mask [N, L].
            image_inputs: image arguments from image_inputs() or
                processor_image_inputs().
            max_new_tokens: generation cap.

        Returns:
            token ids [N, L + generated], context included.
        """
        return self.vlm.generate(input_ids=ids, attention_mask=attn,
                                 max_new_tokens=max_new_tokens, do_sample=False,
                                 use_cache=True, pad_token_id=self.pad_id,
                                 **image_inputs)

    def _hidden_states(self, ids: torch.Tensor, attn: torch.Tensor,
                       image_inputs) -> torch.Tensor:
        """Last-layer hidden state for every token of every row. Reads it off
        the base model when there is one, which skips the language head.

        Args:
            ids: token ids [N, L].
            attn: attention mask [N, L].
            image_inputs: image arguments from image_inputs().

        Returns:
            hidden states [N, L, embed_dim].
        """
        base = getattr(self.vlm, 'model', None)
        if base is not None:
            return base(input_ids=ids, attention_mask=attn, use_cache=False,
                        **image_inputs).last_hidden_state
        return self._vlm_call(ids, attn, image_inputs=image_inputs,
                              output_hidden_states=True).hidden_states[-1]

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """Run images through the full VLM (tower + connector + LM) under the
        fixed prompt and pool one feature vector per image. The last-token
        hidden state is used as the pooled summary.

        Args:
            x: dataset-transformed images [B, C, H, W].

        Returns:
            features [B, embed_dim], float32 for head.
        """
        bsz = x.shape[0]
        hidden = self._hidden_states(self.prompt_ids.expand(bsz, -1),
                                     self.prompt_mask.expand(bsz, -1),
                                     self.image_inputs(x))
        return hidden[:, -1].float()

    def enable_generative(self, class_names: List[str], length_norm: bool = True,
                          score_chunk: int = 16) -> None:
        """Switch to generative mode: tokenize the per-class answer
        `` <name><eos>`` and make forward() return candidate-scoring log-probs
        instead of head logits. Underscores in names become spaces.

        Args:
            class_names: label-order class names, len == num_classes.
            length_norm: if True, score = mean per-token log-prob (removes the
                bias toward short names); if False, raw sum.
            score_chunk: candidates scored per forward (memory knob).

        Outputs: none — sets ``self.generative``, ``self._answer_ids``,
            ``self.length_norm``, ``self.score_chunk``.
        """
        assert len(class_names) == self.num_classes, (
            f"got {len(class_names)} names for {self.num_classes} classes")
        tok = self.processor.tokenizer
        self._answer_ids = []
        for name in class_names:
            ids = tok.encode(' ' + str(name).replace('_', ' '), add_special_tokens=False)
            self._answer_ids.append(torch.tensor(ids + [self.eos_id], dtype=torch.long))
        self.generative = True
        self.length_norm = length_norm
        self.score_chunk = score_chunk

    def set_active_classes(self, n: int) -> None:
        """Restrict candidate scoring to the first ``n`` classes (the classes
        seen so far in class-il). The rest score -1e9; evaluate() masking is
        unaffected.

        Args:
            n: candidate set width, clamped to [1, num_classes].

        Outputs: none — set ``self.n_active``.
        """
        self.n_active = max(1, min(int(n), self.num_classes))

    def _pack_answers(self, answers: List[torch.Tensor]):
        """Build padded (input_ids, attention_mask, labels) rows: the fixed
        prompt plus one answer each. Labels are -100 everywhere except the
        answer span, so the prompt, image placeholders, and padding are masked.

        Args:
            answers: list of 1-D answer token-id tensors.

        Returns:
            (input_ids, attention_mask, labels): each [len(answers), P + Lmax].
        """
        device = self.prompt_ids.device
        plen = self.prompt_ids.shape[1]
        lmax = max(a.numel() for a in answers)
        n = len(answers)
        ids = torch.full((n, plen + lmax), self.pad_id, dtype=torch.long, device=device)
        attn = torch.zeros((n, plen + lmax), dtype=torch.long, device=device)
        labels = torch.full((n, plen + lmax), -100, dtype=torch.long, device=device)
        ids[:, :plen] = self.prompt_ids
        attn[:, :plen] = 1
        for i, a in enumerate(answers):
            a = a.to(device)
            ids[i, plen:plen + a.numel()] = a
            attn[i, plen:plen + a.numel()] = 1
            labels[i, plen:plen + a.numel()] = a
        return ids, attn, labels

    def lm_loss(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Masked-sequence language-model loss: the model learns to generate
        the true class name after the prompt. This is the training twin of
        score_classes.

        Args:
            x: dataset-transformed images [B, C, H, W].
            labels: class ids [B].

        Returns:
            scalar cross-entropy over the answer-span tokens (HF shifts inside;
            prompt and pad positions are -100).
        """
        assert self._answer_ids is not None, "call enable_generative first"
        ids, attn, lab = self._pack_answers([self._answer_ids[int(y)] for y in labels])
        out = self._vlm_call(ids, attn, image_inputs=self.image_inputs(x), labels=lab)
        return out.loss

    def score_classes(self, x: torch.Tensor) -> torch.Tensor:
        """Log-prob of each active candidate name given the image and prompt,
        scored under teacher forcing. The vision tower runs once per batch and
        its features are reused across candidate chunks.

        Args:
            x: dataset-transformed images [B, C, H, W].

        Returns:
            scores [B, num_classes] float32; mean per-token log-prob if
            length_norm else sum; inactive classes are -1e9.
        """
        assert self._answer_ids is not None, "call enable_generative first"
        bsz = x.shape[0]
        img = self.image_features(self.image_inputs(x))  # [B, S, D]
        scores = torch.full((bsz, self.num_classes), -1e9,
                            device=x.device, dtype=torch.float32)
        for lo in range(0, self.n_active, self.score_chunk):
            cand = range(lo, min(lo + self.score_chunk, self.n_active))
            ids, attn, lab = self._pack_answers([self._answer_ids[c] for c in cand])
            n_c = len(cand)
            # Rows are batch-major: [b0(c0..cN), b1(c0..cN), ...]; the image
            # rows line up the same way.
            ids, attn, lab = ids.repeat(bsz, 1), attn.repeat(bsz, 1), lab.repeat(bsz, 1)
            logits = self.candidate_logits(img.repeat_interleave(n_c, dim=0), ids, attn)
            # Softmax only the answer-span slice, not the whole sequence:
            # log_softmax over the full length here would allocate ~7 GiB and
            # can run out of memory once the agem/ewc buffers fill the GPU.
            plen = self.prompt_ids.shape[1]
            logprobs = F.log_softmax(logits[:, plen - 1:-1].float(), dim=-1)
            mask = (lab[:, plen:] != -100)
            tok_lp = logprobs.gather(2, ids[:, plen:].unsqueeze(-1)).squeeze(-1) * mask
            s = tok_lp.sum(1)
            if self.length_norm:
                s = s / mask.sum(1).clamp(min=1)
            scores[:, list(cand)] = s.view(bsz, n_c)
        return scores

    def forward(self, x: torch.Tensor, returnt: str = 'out') -> torch.Tensor:
        """Forward pass: encode the batch, then read the head (classifier mode)
        or score the candidate names (generative mode). This is the entry point
        for the training loop, evaluator, and k-shot adaptation.

        Args:
            x: dataset-transformed input batch [B, C, H, W]
            returnt: 'out' (logits/scores), 'features' (pooled), or 'both'

        Returns:
            [B, num_classes] if 'out', [B, embed_dim] if 'features', tuple
            (out, features) if 'both'.
        """
        assert returnt in ('out', 'features', 'both'), f"Unsupported returnt: {returnt}"

        if returnt == 'features':
            return self._encode(x)

        if self.generative:
            out = self.score_classes(x)
            if returnt == 'both':
                return out, self._encode(x)
            return out

        feats = self._encode(x)
        out = self.classifier(feats)
        if returnt == 'both':
            return out, feats
        return out
