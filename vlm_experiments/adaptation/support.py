# Copyright 2026-present, Amogh Inamdar, Vici Milenia, Richard Zemel.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Shared support-set plumbing for the adaptation modes.

One EvalState is built once per run: all test loaders, offsets, class names, and
the full raw train pool (uint8 images plus targets). k-shot sampling matches
utils/few_shot.create_k_shot_loader exactly, including its per-class
np.random.seed reset, so gradient numbers stay comparable with the ResNet
pipeline. The raw uint8 images are kept around so ICL can build clean PIL demo
images and protonet can push support through the TEST transform (same input
domain as queries).
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset


@dataclass
class EvalState:
    """Everything the adapters need, built once: loaders, offsets, names, raw
    train pool, transforms."""
    test_loaders: List                 # per-task test loader (TEST transform applied)
    offsets: List[Tuple[int, int]]     # per-task (class_lo, class_hi)
    class_names: List[str]             # names in label order
    all_data: np.ndarray               # [N, H, W, C] uint8, whole train stream
    all_targets: np.ndarray            # [N] int
    train_transform: object            # dataset train transform (gradient adapt)
    test_transform: object             # dataset test transform (protonet support)
    n_tasks: int
    n_classes: int


def build_eval_state(args) -> EvalState:
    """Build the dataset twice: once advanced through all tasks to get test
    loaders, offsets, and names, once to collect the whole train pool as raw
    arrays. Like few_shot.create_k_shot_eval_dataset, but one state for all tasks.

    Args:
        args: mammoth args namespace (dataset already resolved).

    Returns:
        EvalState: loaders, offsets, names, raw train pool, and transforms.
    """
    from datasets import get_dataset_class
    dataset_class = get_dataset_class(args)

    ds = dataset_class(args)
    for _ in range(ds.N_TASKS):
        ds.get_data_loaders()
    names = ds.get_class_names() if hasattr(ds, 'get_class_names') else None
    names = list(names) if names is not None else [str(c) for c in range(ds.N_CLASSES)]
    offsets = [ds.get_offsets(t) for t in range(ds.N_TASKS)]

    # Second pass to collect raw train data (same approach as few_shot).
    collect = dataset_class(args)
    all_data, all_targets = [], []
    for _ in range(collect.N_TASKS):
        train_loader, _ = collect.get_data_loaders()
        wrapped = train_loader.dataset
        assert hasattr(wrapped, 'data'), "train dataset needs .data for k-shot sampling"
        data = wrapped.data if isinstance(wrapped.data, np.ndarray) else np.array(wrapped.data)
        all_data.append(data.copy())
        all_targets.extend(list(wrapped.targets))

    test_transform = getattr(ds, 'TEST_TRANSFORM', None)
    if test_transform is None:
        test_transform = transforms.Compose([transforms.ToTensor(),
                                             ds.get_normalization_transform()])

    return EvalState(
        test_loaders=ds.test_loaders,
        offsets=offsets,
        class_names=names,
        all_data=np.vstack(all_data),
        all_targets=np.array(all_targets),
        train_transform=ds.get_transform() if hasattr(ds, 'get_transform') else None,
        test_transform=test_transform,
        n_tasks=ds.N_TASKS,
        n_classes=ds.N_CLASSES,
    )


def sample_k_shot(state: EvalState, task_id: int, k: int,
                  seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Sample k raw train examples per class of ``task_id``. Matches
    few_shot.create_k_shot_loader exactly: call np.random.seed(seed) then choice
    for each class (kept identical on purpose so draws match the repo pipeline).

    Args:
        state: EvalState from build_eval_state.
        task_id: task whose classes get sampled.
        k: examples per class (<=0 gives empty support).
        seed: sampling seed (repo convention: reset before every class).

    Returns:
        (imgs, labels): uint8 [n, H, W, C] and int [n]; n <= k * classes.
    """
    if k <= 0:
        return (np.empty((0,) + state.all_data.shape[1:], dtype=state.all_data.dtype),
                np.empty(0, dtype=np.int64))
    lo, hi = state.offsets[task_id]
    picked = []
    for class_id in range(lo, hi):
        idx = np.where(state.all_targets == class_id)[0]
        n = min(k, len(idx))
        if n > 0:
            np.random.seed(seed)  # reset before every class, to match few_shot
            picked.extend(np.random.choice(idx, size=n, replace=False))
    return state.all_data[picked], state.all_targets[picked].astype(np.int64)


def sample_k_shot_range(state: EvalState, lo: int, hi: int, k: int,
                        seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Sample k raw train examples per class in the class range [lo, hi). Resets
    np.random.seed(seed) before every class, just like ``sample_k_shot``, so a
    class drawn here is identical to the draw ``sample_k_shot`` would make for
    that class on its own.

    Args:
        state: EvalState from build_eval_state.
        lo: first class id (inclusive).
        hi: last class id (exclusive).
        k: examples per class (<=0 gives empty support).
        seed: sampling seed (repo convention: reset before every class).

    Returns:
        (imgs, labels): uint8 [n, H, W, C] and int [n]; n <= k * (hi - lo).
    """
    if k <= 0:
        return (np.empty((0,) + state.all_data.shape[1:], dtype=state.all_data.dtype),
                np.empty(0, dtype=np.int64))
    picked = []
    for class_id in range(lo, hi):
        idx = np.where(state.all_targets == class_id)[0]
        n = min(k, len(idx))
        if n > 0:
            np.random.seed(seed)  # reset before every class, to match few_shot
            picked.extend(np.random.choice(idx, size=n, replace=False))
    return state.all_data[picked], state.all_targets[picked].astype(np.int64)


def sample_k_shot_cil(state: EvalState, ckpt_task: int, k: int,
                      seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Sample k raw train examples per class of every task the checkpoint has
    seen (tasks 0..ckpt_task inclusive), covering classes [0, class_hi(ckpt_task)).
    Thin wrapper over ``sample_k_shot_range``.

    Args:
        state: EvalState from build_eval_state.
        ckpt_task: last task the checkpoint trained on; classes [0, class_hi(ckpt_task)) are seen.
        k: examples per class (<=0 gives empty support).
        seed: sampling seed (repo convention: reset before every class).

    Returns:
        (imgs, labels): uint8 [n, H, W, C] and int [n]; n <= k * n_seen_classes.
    """
    return sample_k_shot_range(state, 0, state.offsets[ckpt_task][1], k, seed)


class _SupportDataset(Dataset):
    """Small in-memory dataset for adaptation: (augmented, target, unaugmented)
    triples. Twin of few_shot.SimpleDataset."""

    def __init__(self, data: np.ndarray, targets: np.ndarray, transform=None):
        """Store raw arrays and transform.

        Args:
            data: uint8 images [n, H, W, C].
            targets: int labels [n].
            transform: dataset train transform (None uses plain ToTensor).

        Outputs: none, plain container.
        """
        self.data, self.targets, self.transform = data, targets, transform

    def __len__(self):
        """Number of stored examples.

        Args: none.

        Returns:
            int: number of support examples.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """One (augmented, target, unaugmented) triple, the format observe()
        expects. The transform gets the raw array because mammoth's
        get_transform() starts with ToPILImage, so converting to PIL first would
        break it. Exact mirror of few_shot.SimpleDataset.

        Args:
            idx: example index.

        Returns:
            (img_aug, target, img_noaug): transformed tensor, int, raw tensor.
        """
        img = self.data[idx]
        pil = Image.fromarray(img.astype(np.uint8))
        img_aug = self.transform(img) if self.transform else transforms.ToTensor()(pil)
        img_noaug = transforms.ToTensor()(pil)
        return img_aug, self.targets[idx], img_noaug


def make_adapt_loader(imgs: np.ndarray, labels: np.ndarray, transform,
                      batch_size: int = 32) -> Optional[DataLoader]:
    """Wrap sampled support in a shuffled DataLoader for gradient adaptation.
    Matches the few_shot loader: batch min(n, batch_size), drop_last off, and
    num_workers 0 to avoid the cluster /tmp socket crash.

    Args:
        imgs: uint8 support images [n, H, W, C].
        labels: int labels [n].
        transform: dataset train transform.
        batch_size: cap on batch size.

    Returns:
        DataLoader, or None when support is empty.
    """
    if len(imgs) == 0:
        return None
    return DataLoader(_SupportDataset(imgs, labels, transform),
                      batch_size=min(len(imgs), batch_size), shuffle=True,
                      num_workers=0, drop_last=False)


def to_test_tensors(imgs: np.ndarray, test_transform) -> torch.Tensor:
    """Push raw uint8 support through the dataset TEST transform, so support and
    query end up in the same input domain with no augmentation.

    Args:
        imgs: uint8 images [n, H, W, C].
        test_transform: dataset TEST_TRANSFORM.

    Returns:
        tensor [n, C, H', W'] in the query domain.
    """
    return torch.stack([test_transform(Image.fromarray(im.astype(np.uint8))) for im in imgs])


def to_pil(imgs: np.ndarray) -> List[Image.Image]:
    """Convert raw uint8 images to a list of PIL RGB images for ICL prompt demos.
    The processor does its own resize and normalize downstream.

    Args:
        imgs: uint8 images [n, H, W, C].

    Returns:
        list of PIL RGB images.
    """
    return [Image.fromarray(im.astype(np.uint8)).convert('RGB') for im in imgs]


def batch_to_pil(x: torch.Tensor, data_mean: torch.Tensor,
                 data_std: torch.Tensor) -> List[Image.Image]:
    """Undo the dataset normalization on a transformed batch (same math as the
    backbone pixel bridge) and return PIL images. Used for ICL query images,
    which only exist as loader tensors.

    Args:
        x: transformed batch [B, C, H, W].
        data_mean: backbone data_mean buffer [1, 3, 1, 1].
        data_std: backbone data_std buffer [1, 3, 1, 1].

    Returns:
        list of PIL RGB images, length B.
    """
    if x.shape[1] == 1:
        x = x.repeat(1, 3, 1, 1)
    x = (x.detach().float().cpu() * data_std.cpu() + data_mean.cpu()).clamp(0.0, 1.0)
    x = (x * 255).round().to(torch.uint8).permute(0, 2, 3, 1).numpy()
    return [Image.fromarray(im) for im in x]
