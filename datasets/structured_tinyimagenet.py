from argparse import Namespace
import logging
import os
from typing import Tuple
import pickle
import urllib.request
import zipfile
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from datasets.seq_tinyimagenet import CLASS_NAMES, MyTinyImagenet, TinyImagenet
from datasets.transforms.denormalization import DeNormalize
from datasets.utils.continual_dataset import (ContinualDataset, fix_class_names_order,
                                              store_masked_loaders)
from utils.conf import base_path
from datasets.utils import set_default_from_args

class StructuredTinyImagenet(ContinualDataset):
    """Structured Tiny Imagenet dataset.
    (Based on Integrating Present and Past in Unsupervised Continual Learning https://arxiv.org/pdf/2404.19132)

    Args:
        NAME (str): name of the dataset.
        SETTING (str): setting of the dataset.
        N_CLASSES_PER_TASK (int): number of classes per task.
        N_TASKS (int): number of tasks.
        N_CLASSES (int): number of classes.
        SIZE (tuple): size of the images.
        MEAN (tuple): mean of the dataset.
        STD (tuple): standard deviation of the dataset.
        TRANSFORM (torchvision.transforms): transformations to apply to the dataset.
    """

    NAME = 'structured-tinyimgnet'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 20
    N_TASKS = 10
    N_CLASSES = N_CLASSES_PER_TASK * N_TASKS
    MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    SIZE = (64, 64)
    TRANSFORM = transforms.Compose(
        [transforms.RandomCrop(64, padding=4),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize(MEAN, STD)])
    ASSETS_DIR = os.path.join('./', 'assets')

    
    def __init__(self, args: Namespace) -> None:
        args.class_order = StructuredTinyImagenet._build_class_order()
        super().__init__(args)

    @staticmethod
    def _get_words_path():
        return os.path.join(StructuredTinyImagenet.ASSETS_DIR, 'words.txt')

    @staticmethod
    def _get_order_path():
        return os.path.join(StructuredTinyImagenet.ASSETS_DIR, 't200_structured_order.pkl')

    @staticmethod
    def _ensure_mapping_files():
        os.makedirs(os.path.join(base_path(), 'TINYIMG'), exist_ok=True)

        order_path = StructuredTinyImagenet._get_order_path()
        if not os.path.exists(order_path):
            logging.info('Downloading Tiny-ImageNet structured order...')
            urllib.request.urlretrieve(
                'https://raw.githubusercontent.com/SkrighYZ/Osiris/main/data_order/t200_structured_order.pkl',
                order_path
            )

        words_path = StructuredTinyImagenet._get_words_path()
        if not os.path.exists(words_path):
            logging.info('Downloading Tiny-ImageNet words.txt...')
            import io
            with urllib.request.urlopen('http://cs231n.stanford.edu/tiny-imagenet-200.zip') as r:
                data = io.BytesIO(r.read())
            with zipfile.ZipFile(data) as z:
                with z.open('tiny-imagenet-200/words.txt') as src, open(words_path, 'w') as dst:
                    dst.write(src.read().decode('utf-8'))

    @staticmethod
    def _build_class_order():
        StructuredTinyImagenet._ensure_mapping_files()

        with open(StructuredTinyImagenet._get_order_path(), 'rb') as f:
            structured_order = pickle.load(f, encoding='latin1')

        wordnet_to_name = {}
        with open(StructuredTinyImagenet._get_words_path(), 'r') as f:
            for line in f:
                nid, name = line.strip().split('\t')
                wordnet_to_name[nid] = name.split(',')[0].strip().lower().replace(' ', '_')

        name_to_idx = {name: i for i, name in enumerate(CLASS_NAMES)}
        missing = [nid for nid in structured_order if wordnet_to_name.get(nid) not in name_to_idx]
        if missing:
            raise ValueError(f'Could not map WordNet IDs to CLASS_NAMES: {missing}')

        structured_indices = [name_to_idx[wordnet_to_name[nid]] for nid in structured_order]
        class_order = np.empty(len(CLASS_NAMES), dtype=int)
        class_order[structured_indices] = np.arange(len(CLASS_NAMES))
        return class_order

    def get_data_loaders(self) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        transform = self.TRANSFORM
        test_transform = transforms.Compose(
            [transforms.ToTensor(), self.get_normalization_transform()])

        train_dataset = MyTinyImagenet(base_path() + 'TINYIMG',
                                       train=True, download=True, transform=transform)
        test_dataset = TinyImagenet(base_path() + 'TINYIMG',
                                    train=False, download=True, transform=test_transform)
        train, test = store_masked_loaders(train_dataset, test_dataset, self)
        return train, test

    @set_default_from_args("backbone")
    def get_backbone():
        return "resnet18"

    @staticmethod
    def get_loss():
        return F.cross_entropy

    def get_transform(self):
        transform = transforms.Compose(
            [transforms.ToPILImage(), self.TRANSFORM])
        return transform

    @staticmethod
    def get_normalization_transform():
        transform = transforms.Normalize(StructuredTinyImagenet.MEAN, StructuredTinyImagenet.STD)
        return transform

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize(StructuredTinyImagenet.MEAN, StructuredTinyImagenet.STD)
        return transform

    @set_default_from_args('n_epochs')
    def get_epochs(self):
        return 50

    @set_default_from_args('batch_size')
    def get_batch_size(self):
        return 32

    def get_class_names(self):
        if self.class_names is not None:
            return self.class_names
        classes = fix_class_names_order(CLASS_NAMES, self.args)
        self.class_names = classes
        return self.class_names