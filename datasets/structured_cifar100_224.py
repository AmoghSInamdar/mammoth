# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import pickle
from typing import Tuple

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR100

from datasets.transforms.denormalization import DeNormalize
from datasets.utils.continual_dataset import (fix_class_names_order,
                                              store_masked_loaders)
from utils.conf import base_path
from datasets.utils import set_default_from_args
from datasets.seq_cifar100_224 import SequentialCIFAR100224
from datasets.structured_cifar100 import MyCIFAR100Superclass, TCIFAR100Superclass
from utils.prompt_templates import templates


class StructuredCIFAR100224(SequentialCIFAR100224):
    """Sequential CIFAR100 Dataset organized by superclasses, at 224x224 resolution.

    Identical to struct-cifar100 (20 superclass tasks, 5 fine classes each), but
    derived from the seq-cifar100-224 variant: 224x224 images, ViT-B/16 backbone,
    bicubic resize transforms and ImageNet normalization statistics.

    Args:
        NAME (str): name of the dataset.
        SETTING (str): setting of the dataset.
        N_CLASSES_PER_TASK (int): number of classes per task (5 fine classes per superclass).
        N_TASKS (int): number of tasks (20 superclasses).
        N_CLASSES (int): number of classes (100 fine classes).
        SIZE (tuple): size of the images.
        MEAN (tuple): mean of the dataset.
        STD (tuple): standard deviation of the dataset.
        TRANSFORM (torchvision.transforms): transformation to apply to the data.
        TEST_TRANSFORM (torchvision.transforms): transformation to apply to the test data."""

    NAME = 'struct-cifar100-224'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 5  # 5 fine classes per superclass
    N_TASKS = 20  # 20 superclasses
    N_CLASSES = N_CLASSES_PER_TASK * N_TASKS  # 100 total fine classes

    def get_data_loaders(self) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        transform = self.TRANSFORM

        test_transform = self.TEST_TRANSFORM

        train_dataset = MyCIFAR100Superclass(base_path() + 'CIFAR100', train=True,
                                             download=True, transform=transform)
        test_dataset = TCIFAR100Superclass(base_path() + 'CIFAR100', train=False,
                                           download=True, transform=test_transform,
                                           fine_to_coarse=train_dataset.fine_to_coarse)

        # Remap targets to be ordered by superclass
        # Original fine labels 0-99 -> new labels 0-99 ordered by superclass
        coarse_to_fine = {}
        for fine_idx, coarse_idx in train_dataset.fine_to_coarse.items():
            if coarse_idx not in coarse_to_fine:
                coarse_to_fine[coarse_idx] = []
            coarse_to_fine[coarse_idx].append(fine_idx)

        # Create mapping from old fine label to new fine label
        old_to_new_label = {}
        new_label = 0
        for coarse_idx in range(20):
            if coarse_idx in coarse_to_fine:
                for old_fine_idx in sorted(coarse_to_fine[coarse_idx]):
                    old_to_new_label[old_fine_idx] = new_label
                    new_label += 1

        # Apply remapping to targets
        train_dataset.targets = [old_to_new_label.get(t, t) for t in train_dataset.targets]
        test_dataset.targets = [old_to_new_label.get(t, t) for t in test_dataset.targets]

        train, test = store_masked_loaders(train_dataset, test_dataset, self)

        return train, test

    @staticmethod
    def get_transform():
        transform = transforms.Compose(
            [transforms.ToPILImage(), StructuredCIFAR100224.TRANSFORM])
        return transform

    def get_class_names(self):
        if self.class_names is not None:
            return self.class_names

        # Load the CIFAR100 dataset to get fine class names
        cifar100 = CIFAR100(base_path() + 'CIFAR100', train=True, download=True)
        fine_classes = cifar100.classes  # 100 fine class names

        # Load training data to get fine-to-coarse mapping
        train_path = os.path.join(base_path() + 'CIFAR100', 'cifar-100-python', 'train')
        with open(train_path, 'rb') as f:
            train_data = pickle.load(f, encoding='latin1')

        # Build fine_to_coarse mapping
        fine_to_coarse = {}
        for fine_label, coarse_label in zip(train_data['fine_labels'], train_data['coarse_labels']):
            fine_to_coarse[fine_label] = coarse_label

        # Create class names ordered by superclass
        coarse_to_fine = {}
        for fine_idx, coarse_idx in fine_to_coarse.items():
            if coarse_idx not in coarse_to_fine:
                coarse_to_fine[coarse_idx] = []
            coarse_to_fine[coarse_idx].append(fine_idx)

        # Order class names by superclass
        class_names = []
        for coarse_idx in range(20):
            if coarse_idx in coarse_to_fine:
                for fine_idx in sorted(coarse_to_fine[coarse_idx]):
                    class_names.append(fine_classes[fine_idx])

        classes = fix_class_names_order(class_names, self.args)
        self.class_names = classes
        return self.class_names

    @set_default_from_args("backbone")
    def get_backbone():
        return "vit"

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_normalization_transform():
        transform = transforms.Normalize(SequentialCIFAR100224.MEAN, SequentialCIFAR100224.STD)
        return transform

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize(SequentialCIFAR100224.MEAN, SequentialCIFAR100224.STD)
        return transform

    @set_default_from_args('n_epochs')
    def get_epochs(self):
        return 20

    @set_default_from_args('batch_size')
    def get_batch_size(self):
        return 128

    @staticmethod
    def get_prompt_templates():
        return templates['cifar100']
