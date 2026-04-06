from argparse import Namespace

import torch

from datasets.seq_cifar100 import TCIFAR100, MyCIFAR100
from datasets.utils.continual_dataset import ContinualDataset
from typing import Tuple

import torch.nn.functional as F
import torchvision.transforms as transforms

from datasets.transforms.denormalization import DeNormalize
from datasets.utils.continual_dataset import (ContinualDataset, fix_class_names_order,
                                              store_masked_loaders)
from utils.conf import base_path
from datasets.utils import set_default_from_args

class StandardSplitCIFAR100(ContinualDataset):
    """Standard Split CIFAR100 Dataset. 
    (Based on Integrating Present and Past in Unsupervised Continual Learning https://arxiv.org/pdf/2404.19132)
    Contains 50,000 32×32 images from 100 classes that are randomly grouped into a disjoint set of tasks.

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
    
    NAME = 'std-split-cifar100'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 0
    N_TASKS = 0
    N_CLASSES = 100
    SIZE = (32, 32)
    MEAN, STD = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
    TRANSFORM = transforms.Compose(
        [transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)])
    
    def __init__(self, args: Namespace) -> None:
        super().__init__(args)
        self.NAME = 'std-split-cifar100'
        self.SETTING = 'class-il'
        self.SIZE = (32, 32)
        self.MEAN, self.STD = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
        self.TRANSFORM = transforms.Compose(
            [transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(self.MEAN, self.STD)])
        
        self.N_CLASSES_PER_TASK = args.n_classes_per_task
        self.N_TASKS = args.n_tasks
        self.N_CLASSES = self.N_CLASSES_PER_TASK * self.N_TASKS

        self.train_dataset = MyCIFAR100(base_path() + 'CIFAR100', train=True,
                                   download=True, transform=self.TRANSFORM)
        self.test_dataset = TCIFAR100(base_path() + 'CIFAR100', train=False,
                                 download=True, transform=transforms.Compose(
            [transforms.ToTensor(), self.get_normalization_transform()]))
    
    def get_data_loaders(self) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        train, test = store_masked_loaders(self.train_dataset, self.test_dataset, self)

        return train, test

    @staticmethod
    def get_transform():
        transform = transforms.Compose(
            [transforms.ToPILImage(), StandardSplitCIFAR100.TRANSFORM])
        return transform

    @set_default_from_args("backbone")
    def get_backbone():
        return "resnet18"

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_normalization_transform():
        transform = transforms.Normalize(StandardSplitCIFAR100.MEAN, StandardSplitCIFAR100.STD)
        return transform

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize(StandardSplitCIFAR100.MEAN, StandardSplitCIFAR100.STD)
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
        classes = self.train_dataset.classes
        classes = fix_class_names_order(classes, self.args)
        self.class_names = classes
        return self.class_names
