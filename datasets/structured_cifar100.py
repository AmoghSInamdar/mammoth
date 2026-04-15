from argparse import Namespace
import os
import pickle

from PIL import Image
import torch
from torchvision.datasets import CIFAR100

from datasets.utils.continual_dataset import ContinualDataset
from typing import Tuple

import torch.nn.functional as F
import torchvision.transforms as transforms

from datasets.transforms.denormalization import DeNormalize
from datasets.utils.continual_dataset import (ContinualDataset, fix_class_names_order,
                                              store_masked_loaders)
from utils.conf import base_path
from datasets.utils import set_default_from_args

class SuperclassSplitCIFAR100(CIFAR100):
    """
    Overrides the CIFAR100 dataset to split by superclass.
    """
    def __init__(self, root, train=True, transform=None,
                 target_transform=None, download=False) -> None:
        self.not_aug_transform = transforms.Compose([transforms.ToTensor()])
        self.root = root
        super(SuperclassSplitCIFAR100, self).__init__(root, train, transform, target_transform, not self._check_integrity())
        
        meta_path = os.path.join(self.root, 'cifar-100-python', 'meta')
        data_path = os.path.join(self.root, 'cifar-100-python', 'train' if train else 'test')
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f, encoding='latin1')
        with open(data_path, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
        self.superclass = meta['coarse_label_names']
        self.targets = data['coarse_labels']

    def __getitem__(self, index: int) -> Tuple[Image.Image, int, Image.Image]:
        """
        Gets the requested element from the dataset.

        Args:
            index: index of the element to be returned

        Returns:
            tuple: (image, target) where target is index of the target class.
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

class StructuredCIFAR100(ContinualDataset):
    """Structured Split CIFAR100 Dataset. 
    (Based on Integrating Present and Past in Unsupervised Continual Learning https://arxiv.org/pdf/2404.19132)
    20 tasks total, one task = one superclass label

    Args:
        NAME (str): name of the dataset.
        SETTING (str): setting of the dataset.
        N_CLASSES_PER_TASK (int): number of classes per task.
        N_TASKS (int): number of tasks.
        N_CLASSES (int): number of classes.
        SIZE (tuple): size of the images.
        MEAN (tuple): mean of the dataset.
        STD (tuple): standard deviation of the dataset.
        TRANSFORM (torchvision.transforms): transformation to apply to the data."""
    
    NAME = 'structured-cifar100-bug'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 5
    N_TASKS = 20
    N_CLASSES = 20
    SIZE = (32, 32)
    MEAN, STD = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
    TRANSFORM = transforms.Compose(
        [transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)])
    
    def __init__(self, args: Namespace) -> None:
        super().__init__(args)
        self.NAME = 'structured-cifar100'
        self.SETTING = 'class-il'
        self.SIZE = (32, 32)
        self.MEAN, self.STD = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
        self.TRANSFORM = transforms.Compose(
            [transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(self.MEAN, self.STD)])
        
        self.train_dataset = SuperclassSplitCIFAR100(base_path() + 'CIFAR100', train=True,
                                   download=True, transform=self.TRANSFORM)
        self.test_dataset = SuperclassSplitCIFAR100(base_path() + 'CIFAR100', train=False,
                                 download=True, transform=transforms.Compose(
            [transforms.ToTensor(), self.get_normalization_transform()]))
    
    def get_data_loaders(self) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        train, test = store_masked_loaders(self.train_dataset, self.test_dataset, self)

        return train, test

    @staticmethod
    def get_transform():
        transform = transforms.Compose(
            [transforms.ToPILImage(), StructuredCIFAR100.TRANSFORM])
        return transform

    @set_default_from_args("backbone")
    def get_backbone():
        return "resnet18"

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_normalization_transform():
        transform = transforms.Normalize(StructuredCIFAR100.MEAN, StructuredCIFAR100.STD)
        return transform

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize(StructuredCIFAR100.MEAN, StructuredCIFAR100.STD)
        return transform

    @set_default_from_args('n_epochs')
    def get_epochs(self):
        return 50

    @set_default_from_args('batch_size')
    def get_batch_size(self):
        return 32

    @set_default_from_args('lr_milestones')
    def get_scheduler_name(self):
        return [35, 45]
    
    @set_default_from_args('permute_classes')
    def get_permute_classes(self):
        return False

    def get_class_names(self):
        if self.class_names is not None:
            return self.class_names
        classes = self.train_dataset.superclass
        classes = fix_class_names_order(classes, self.args)
        self.class_names = classes
        return self.class_names
