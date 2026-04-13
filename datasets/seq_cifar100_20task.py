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
from PIL import Image
from torchvision.datasets import CIFAR100

from datasets.seq_cifar100 import SequentialCIFAR100
from datasets.transforms.denormalization import DeNormalize
from datasets.utils.continual_dataset import (ContinualDataset, fix_class_names_order,
                                              store_masked_loaders)
from utils.conf import base_path
from datasets.utils import set_default_from_args


class SequentialCIFAR100_20Task(SequentialCIFAR100):
    """Sequential CIFAR100 Dataset, but with 20 tasks."""

    NAME = 'seq-cifar100-20task'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 5
    N_TASKS = 20
    N_CLASSES = N_CLASSES_PER_TASK * N_TASKS

    @set_default_from_args("backbone")
    def get_backbone():
        return "resnet18"

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_normalization_transform():
        transform = transforms.Normalize(SequentialCIFAR100.MEAN, SequentialCIFAR100.STD)
        return transform

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize(SequentialCIFAR100.MEAN, SequentialCIFAR100.STD)
        return transform

    @set_default_from_args('n_epochs')
    def get_epochs(self):
        return 50

    @set_default_from_args('batch_size')
    def get_batch_size(self):
        return 32

    @set_default_from_args('lr_scheduler')
    def get_scheduler_name(self):
        return 'multisteplr'

    @set_default_from_args('lr_milestones')
    def get_scheduler_name(self):
        return [35, 45]

    def get_class_names(self):
        if self.class_names is not None:
            return self.class_names
        classes = CIFAR100(base_path() + 'CIFAR100', train=True, download=True).classes
        classes = fix_class_names_order(classes, self.args)
        self.class_names = classes
        return self.class_names