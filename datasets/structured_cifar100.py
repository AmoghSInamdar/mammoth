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

from datasets.transforms.denormalization import DeNormalize
from datasets.utils.continual_dataset import (ContinualDataset, fix_class_names_order,
                                              store_masked_loaders)
from utils.conf import base_path
from datasets.utils import set_default_from_args
from datasets.seq_cifar100 import SequentialCIFAR100


class TCIFAR100(CIFAR100):
    """Workaround to avoid printing the already downloaded messages."""

    def __init__(self, root, train=True, transform=None,
                 target_transform=None, download=False) -> None:
        self.root = root
        super(TCIFAR100, self).__init__(root, train, transform, target_transform, download=not self._check_integrity())


class TCIFAR100Superclass(CIFAR100):
    """Test CIFAR100 with fine labels (remapped by superclass ordering)."""

    def __init__(self, root, train=True, transform=None,
                 target_transform=None, download=False, fine_to_coarse=None) -> None:
        self.root = root
        self.fine_to_coarse = fine_to_coarse or {}
        super(TCIFAR100Superclass, self).__init__(root, train, transform, target_transform, download=not self._check_integrity())


class MyCIFAR100Superclass(CIFAR100):
    """
    Overrides the CIFAR100 dataset to change the getitem function.
    """

    def __init__(self, root, train=True, transform=None,
                 target_transform=None, download=False) -> None:
        self.not_aug_transform = transforms.Compose([transforms.ToTensor()])
        self.root = root
        super(MyCIFAR100Superclass, self).__init__(root, train, transform, target_transform, not self._check_integrity())
        
        # Load metadata to get superclass (coarse) labels
        meta_path = os.path.join(self.root, 'cifar-100-python', 'meta')
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f, encoding='latin1')
        self.superclass_names = meta['coarse_label_names']
        
        # Load training data to build fine-to-coarse mapping
        if train:
            data_path = os.path.join(self.root, 'cifar-100-python', 'train')
        else:
            data_path = os.path.join(self.root, 'cifar-100-python', 'test')
        
        with open(data_path, 'rb') as f:
            train_data = pickle.load(f, encoding='latin1')
        
        # Create mapping from fine class to coarse class
        self.fine_to_coarse = {}
        for fine_label, coarse_label in zip(train_data['fine_labels'], train_data['coarse_labels']):
            self.fine_to_coarse[fine_label] = coarse_label
        # and the coarse label is derivable from the metadata
        # For now, we'll compute it on the fly from the training split
        self.coarse_labels = train_data['coarse_labels']
        
        # Rebuild: create a mapping from fine class to coarse class
        self.fine_to_coarse = {}
        for i, (fine_label, coarse_label) in enumerate(zip(train_data['fine_labels'], train_data['coarse_labels'])):
            if fine_label not in self.fine_to_coarse:
                self.fine_to_coarse[fine_label] = coarse_label

    def __getitem__(self, index: int) -> Tuple[Image.Image, int, Image.Image]:
        """
        Gets the requested element from the dataset.

        Args:
            index: index of the element to be returned

        Returns:
            tuple: (image, target, not_aug_img) where target is the fine class index (0-99).
        """
        img, target = self.data[index], self.targets[index]

        # to return a PIL Image
        img = Image.fromarray(img, mode='RGB')
        original_img = img.copy()

        not_aug_img = self.not_aug_transform(original_img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if hasattr(self, 'logits'):
            return img, target, not_aug_img, self.logits[index]

        return img, target, not_aug_img


class StructuredCIFAR100(SequentialCIFAR100):
    """Sequential CIFAR100 Dataset organized by superclasses.
    
    Each of the 20 coarse classes (superclasses) becomes a separate task,
    with 5 fine classes per task.

    Args:
        NAME (str): name of the dataset.
        SETTING (str): setting of the dataset.
        N_CLASSES_PER_TASK (int): number of classes per task (5 fine classes per superclass).
        N_TASKS (int): number of tasks (20 superclasses).
        N_CLASSES (int): number of classes (100 fine classes).
        SIZE (tuple): size of the images.
        MEAN (tuple): mean of the dataset.
        STD (tuple): standard deviation of the dataset.
        TRANSFORM (torchvision.transforms): transformation to apply to the data."""

    NAME = 'struct-cifar100'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 5  # 5 fine classes per superclass
    N_TASKS = 20  # 20 superclasses
    N_CLASSES = N_CLASSES_PER_TASK * N_TASKS  # 100 total fine classes

    def get_data_loaders(self) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        transform = self.TRANSFORM

        test_transform = transforms.Compose(
            [transforms.ToTensor(), self.get_normalization_transform()])

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
            [transforms.ToPILImage(), StructuredCIFAR100.TRANSFORM])
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
