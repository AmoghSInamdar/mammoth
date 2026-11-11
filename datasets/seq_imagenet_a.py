# Copyright 2026-present, Amogh Inamdar, Vici Milenia, Richard Zemel.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Split ImageNet-A: 200 natural adversarial classes -> 10 tasks x 20 classes.

ImageNet-A (Hendrycks et al., CVPR'21) was adversarially curated against
ImageNet-trained models, so pretrained backbones start far from ceiling
(ViT-B/16 in21k-ft-in1k linear probe: ~40%) — a harder companion to
seq-imagenet-r for continual-learning-with-pretrained-models experiments.
Standard 10x20 split as in ADAM (IJCV'24) / RanPAC (NeurIPS'23) / EASE.

The dataset is small (7,450 images, 3-100 per class), so splits are generated
per class (datasets/imagenet_a_utils/): ~40% test, up to 10 held-out images
reserved as the k-shot support source (exposed as ``heldout_data`` /
``heldout_targets``, the same contract seq-imagenet-r uses — utils/few_shot.py
draws support from them and evaluates on the full test split), remainder train.
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


class MyImagenetA(Dataset):
    N_CLASSES = 200
    UTILS_DIR = 'imagenet_a_utils'

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
        cfg = yaml.load(open(f'{pwd}/{self.UTILS_DIR}/imagenet-a_{split}.yaml'),
                        Loader=yaml.Loader)
        self.data = np.array(cfg['data'])
        self.targets = np.array(cfg['targets'])

        if self.train:
            held = yaml.load(open(f'{pwd}/{self.UTILS_DIR}/imagenet-a_heldout.yaml'),
                             Loader=yaml.Loader)
            self.heldout_data = np.array(held['data'])
            self.heldout_targets = np.array(held['targets'])

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index: int) -> Tuple[Image.Image, int, Image.Image]:
        img, target = self.data[index], self.targets[index]
        img = Image.open(img).convert('RGB')
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


class SequentialImagenetA(ContinualDataset):

    NAME = 'seq-imagenet-a'
    SETTING = 'class-il'
    N_TASKS = 10
    N_CLASSES = 200
    N_CLASSES_PER_TASK = N_CLASSES // N_TASKS
    MEAN, STD = (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)
    SIZE = (224, 224)

    TRANSFORM = transforms.Compose([
        transforms.RandomResizedCrop(SIZE[0], interpolation=InterpolationMode.BICUBIC),
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
        train_dataset = MyImagenetA('data/IMAGENET_A/imagenet-a', train=True,
                                    transform=self.TRANSFORM)
        test_dataset = MyImagenetA('data/IMAGENET_A/imagenet-a', train=False,
                                   transform=self.TEST_TRANSFORM)
        train, test = store_masked_loaders(train_dataset, test_dataset, self)
        return train, test

    def get_class_names(self):
        if self.class_names is not None:
            return self.class_names
        pwd = os.path.dirname(os.path.abspath(__file__))
        with open(pwd + '/imagenet_a_utils/label_to_class_name.pkl', 'rb') as f:
            label_to_class_name = pickle.load(f)
        class_names = [x.split(',')[0].replace('_', ' ') for x in label_to_class_name.values()]
        class_names = fix_class_names_order(class_names, self.args)
        self.class_names = class_names
        return self.class_names

    @staticmethod
    def get_transform():
        return transforms.Compose([transforms.ToPILImage(), SequentialImagenetA.TRANSFORM])

    @set_default_from_args("backbone")
    def get_backbone():
        return "vit"

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_normalization_transform():
        return transforms.Normalize(mean=SequentialImagenetA.MEAN, std=SequentialImagenetA.STD)

    @staticmethod
    def get_denormalization_transform():
        return DeNormalize(SequentialImagenetA.MEAN, SequentialImagenetA.STD)

    @set_default_from_args('n_epochs')
    def get_epochs(self):
        return 20

    @set_default_from_args('batch_size')
    def get_batch_size(self):
        return 32


class MyImagenetA100(MyImagenetA):
    """ImageNet-A restricted to the 100 best-supported classes (labels remapped 0-99)."""
    N_CLASSES = 100
    UTILS_DIR = 'imagenet_a100_utils'


class SequentialImagenetA100(SequentialImagenetA):
    """
    Winnowed Split ImageNet-A: the 100 classes with the most training images
    (84% of the train data; >=9 train, >=10 test, >=6 held-out images per class),
    composed into a shorter 5-task sequence to reduce small-sample noise.
    """
    NAME = 'seq-imagenet-a-100'
    N_TASKS = 5
    N_CLASSES = 100
    N_CLASSES_PER_TASK = N_CLASSES // N_TASKS

    def get_data_loaders(self):
        train_dataset = MyImagenetA100('data/IMAGENET_A/imagenet-a', train=True,
                                       transform=self.TRANSFORM)
        test_dataset = MyImagenetA100('data/IMAGENET_A/imagenet-a', train=False,
                                      transform=self.TEST_TRANSFORM)
        train, test = store_masked_loaders(train_dataset, test_dataset, self)
        return train, test

    def get_class_names(self):
        if self.class_names is not None:
            return self.class_names
        pwd = os.path.dirname(os.path.abspath(__file__))
        with open(pwd + '/imagenet_a100_utils/label_to_class_name.pkl', 'rb') as f:
            label_to_class_name = pickle.load(f)
        class_names = [x.split(',')[0].replace('_', ' ') for x in label_to_class_name.values()]
        class_names = fix_class_names_order(class_names, self.args)
        self.class_names = class_names
        return self.class_names
