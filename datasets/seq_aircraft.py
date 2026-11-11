# Copyright 2026-present, Amogh Inamdar, Vici Milenia, Richard Zemel.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Split FGVC-Aircraft: 100 aircraft variants -> 10 tasks x 10 classes.

Fine-grained variant discrimination (737-700 vs 737-800) is a capability
ImageNet pretraining never needed, so the pretrained ViT starts far from
ceiling without the imagery being ImageNet-derived — a non-ImageNet OOD
companion to seq-imagenet-a. Labels follow a seeded shuffle of variants.txt
so each task mixes manufacturers (see datasets/aircraft_utils/generate_splits.py).

Same held-out support contract as seq-imagenet-a: 10 trainval images/class are
excluded from training and exposed as ``heldout_data``/``heldout_targets`` for
utils/few_shot.py; evaluation uses the official test split. The 20-pixel
copyright banner at the bottom of every FGVC image is cropped on load.
"""

import os
import pickle
from typing import Tuple

import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms  # type: ignore[import-untyped]
import yaml
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import InterpolationMode  # type: ignore[import-untyped]

from datasets.transforms.denormalization import DeNormalize
from datasets.utils import set_default_from_args
from datasets.utils.continual_dataset import ContinualDataset, fix_class_names_order, store_masked_loaders

BANNER_PX = 20


def _load_cropped(path):
    img = Image.open(path).convert('RGB')
    return img.crop((0, 0, img.width, img.height - BANNER_PX))


class MyAircraft(Dataset):
    N_CLASSES = 100

    def __init__(self, root, train=True, transform=None,
                 target_transform=None, download=False) -> None:
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        self.not_aug_transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor()])

        pwd = os.path.dirname(os.path.abspath(__file__))
        split = 'train' if self.train else 'test'
        cfg = yaml.load(open(f'{pwd}/aircraft_utils/aircraft_{split}.yaml'),
                        Loader=yaml.Loader)
        self.data = np.array(cfg['data'])
        self.targets = np.array(cfg['targets'])

        if self.train:
            held = yaml.load(open(f'{pwd}/aircraft_utils/aircraft_heldout.yaml'),
                             Loader=yaml.Loader)
            self.heldout_data = np.array(held['data'])
            self.heldout_targets = np.array(held['targets'])

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index: int) -> Tuple[Image.Image, int, Image.Image]:
        img, target = self.data[index], self.targets[index]
        img = _load_cropped(img)
        original_img = img.copy()
        not_aug_img = self.not_aug_transform(original_img)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        if not self.train:
            return img, target
        if hasattr(self, 'logits'):
            return img, target, not_aug_img, self.logits[index]
        return img, target, not_aug_img


class SequentialAircraft(ContinualDataset):

    NAME = 'seq-aircraft'
    SETTING = 'class-il'
    N_TASKS = 10
    N_CLASSES = 100
    N_CLASSES_PER_TASK = N_CLASSES // N_TASKS
    MEAN, STD = (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)
    SIZE = (224, 224)

    TRANSFORM = transforms.Compose([
        transforms.RandomResizedCrop(SIZE[0], scale=(0.5, 1.0),
                                     interpolation=InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])
    TEST_TRANSFORM = transforms.Compose([
        transforms.Resize(size=(256, 256), interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(SIZE[0]),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)])

    def get_data_loaders(self):
        train_dataset = MyAircraft('data/FGVC_AIRCRAFT', train=True,
                                   transform=self.TRANSFORM)
        test_dataset = MyAircraft('data/FGVC_AIRCRAFT', train=False,
                                  transform=self.TEST_TRANSFORM)
        train, test = store_masked_loaders(train_dataset, test_dataset, self)
        return train, test

    def get_class_names(self):
        if self.class_names is not None:
            return self.class_names
        pwd = os.path.dirname(os.path.abspath(__file__))
        with open(pwd + '/aircraft_utils/label_to_class_name.pkl', 'rb') as f:
            label_to_class_name = pickle.load(f)
        class_names = [label_to_class_name[i] for i in range(self.N_CLASSES)]
        class_names = fix_class_names_order(class_names, self.args)
        self.class_names = class_names
        return self.class_names

    @staticmethod
    def get_transform():
        return transforms.Compose([transforms.ToPILImage(), SequentialAircraft.TRANSFORM])

    @set_default_from_args("backbone")
    def get_backbone():
        return "vit"

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_normalization_transform():
        return transforms.Normalize(mean=SequentialAircraft.MEAN, std=SequentialAircraft.STD)

    @staticmethod
    def get_denormalization_transform():
        return DeNormalize(SequentialAircraft.MEAN, SequentialAircraft.STD)

    @set_default_from_args('n_epochs')
    def get_epochs(self):
        return 20

    @set_default_from_args('batch_size')
    def get_batch_size(self):
        return 32
