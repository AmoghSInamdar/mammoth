from argparse import Namespace
import copy
import logging
from typing import Tuple

from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from datasets.utils import set_default_from_args

from datasets.perm_mnist import MyMNIST
from datasets.seq_mnist import SequentialMNIST
from datasets.transforms.rotation import SmoothRotation
from datasets.utils.continual_dataset import MammothDatasetWrapper, store_masked_loaders
from utils.conf import base_path


class MNISTSmoothRotation(SequentialMNIST):
    NAME = 'smooth-mnist'
    N_CLASSES = 10
    N_TASKS = 10
    IN_TASK_ANGLE_RANGE = 15
    BETWEEN_TASK_ANGLE_RANGE = 5
    INIT_ANGLE = 0
    MAX_ANGLE = 360
    N_CLASSES_PER_TASK = 10
    IS_SLICED = True
    train_slices = {}  # keyed by task_id
    test_slices = {}
    SETTING = 'domain-il'
    c_task = 0
    TRANSFORM =  transforms.Compose([transforms.Grayscale(num_output_channels=3), transforms.ToTensor()])

    def __init__(self, args: Namespace) -> None:
        self.rotation = SmoothRotation(
            self.INIT_ANGLE,
            self.IN_TASK_ANGLE_RANGE,
            self.BETWEEN_TASK_ANGLE_RANGE,
            self.MAX_ANGLE
        )
        super().__init__(args)

    def get_data_loaders(self):
        logging.info(f"[{self.NAME}] get_data_loaders called, c_task={self.c_task}")

        if not self.train_slices:
            train_dataset = MammothDatasetWrapper(
                MyMNIST(base_path() + 'MNIST', train=True, download=True, transform=self.TRANSFORM),
                train=True
            )
            self.reorder_samples_by_task(train_dataset, is_train=True)

        if not self.test_slices:
            test_dataset = MammothDatasetWrapper(
                MNIST(base_path() + 'MNIST', train=False, download=True, transform=self.TRANSFORM),
                train=False
            )
            self.reorder_samples_by_task(test_dataset, is_train=False)

        self.c_task += 1

        rotated_train_data = self.rotation.transform_batch_for_task(self.train_slices[self.c_task].dataset.data, self.c_task, self.NAME)
        self.train_slices[self.c_task].dataset.data = rotated_train_data
        train_slice = self.train_slices[self.c_task]

        rotated_test_data  = self.rotation.transform_batch_for_task(self.test_slices[self.c_task].dataset.data, self.c_task, self.NAME)
        self.test_slices[self.c_task].dataset.data = rotated_test_data
        test_slice = self.test_slices[self.c_task]

        unique, counts = np.unique(train_slice.dataset.targets.numpy(), return_counts=True)
        te_unique, te_counts = np.unique(test_slice.dataset.targets.numpy(), return_counts=True)

        self.class_names = unique

        logging.info(f"[{self.NAME} - Train] task={self.c_task} class distribution: {dict(zip(unique.tolist(), counts.tolist()))} train slice={len(train_slice)}")
        logging.info(f"[{self.NAME} - Test] task={self.c_task} class distribution: {dict(zip(te_unique.tolist(), te_counts.tolist()))} test slice={len(test_slice)}")

        train, test = store_masked_loaders(train_slice, test_slice, self)
        return train, test

    def reorder_samples_by_task(self, wrapped_dataset: MammothDatasetWrapper, is_train=True):
        targets = np.array(wrapped_dataset.dataset.targets)
        dataset_name = "Train" if is_train else "Test"

        class_indices = {c: np.where(targets == c)[0].tolist() for c in range(self.N_CLASSES_PER_TASK)}
        min_class_count = min(len(v) for v in class_indices.values())
        samples_per_class = min_class_count // self.N_TASKS
        task_size = samples_per_class * self.N_CLASSES_PER_TASK
        new_total = task_size * self.N_TASKS

        logging.info(f"[{self.NAME}] {dataset_name} class distribution: { {c: len(v) for c, v in class_indices.items()} }")
        logging.info(f"[{self.NAME}] {dataset_name} min_class_count={min_class_count} samples_per_class={samples_per_class} task_size={task_size} new_total={new_total}")

        task_indices = []
        task_ids = []
        for task_id in range(self.N_TASKS):
            idx = []
            for c in range(self.N_CLASSES_PER_TASK):
                idx.extend(class_indices[c][task_id * samples_per_class:(task_id + 1) * samples_per_class])
            task_indices.append(idx)
            task_ids.append(np.full(len(idx), task_id, dtype=np.int64))

        all_indices = np.concatenate([np.array(idx) for idx in task_indices])
        wrapped_dataset.dataset.data    = wrapped_dataset.dataset.data[all_indices]
        wrapped_dataset.dataset.targets = torch.tensor(targets[all_indices])
        wrapped_dataset.indexes         = np.arange(len(all_indices))
        wrapped_dataset.task_ids        = np.concatenate(task_ids)

        logging.info(f"[{self.NAME}] {dataset_name} reordered: new_total={new_total} task_size={task_size} samples_per_class={samples_per_class}")

        for task_id in range(self.N_TASKS):
            start = task_id * task_size
            end   = start + task_size

            chunk = wrapped_dataset.dataset.data[start:end].numpy()
            logging.info(f"[{self.NAME}] {dataset_name} Task {task_id} - new_total={new_total} task_size={task_size} start={start} end={end}")

            cache = self.train_slices if is_train else self.test_slices
            sliced = copy.copy(wrapped_dataset)
            sliced.dataset = copy.copy(wrapped_dataset.dataset)
            sliced.dataset.data    = wrapped_dataset.dataset.data[start:end]
            sliced.dataset.targets = wrapped_dataset.dataset.targets[start:end]
            sliced.task_ids        = wrapped_dataset.task_ids[start:end].copy()
            sliced.indexes         = np.arange(task_size)
            cache[task_id] = sliced         

    def get_class_names(self):
        if self.class_names is None:
            all_classes = MNIST(base_path() + 'MNIST', train=True, download=True).classes
            self.class_names = [c.split('-')[1].strip() for c in all_classes]
        return self.class_names

    @staticmethod
    def get_transform():
        return transforms.Compose([transforms.ToPILImage(), MNISTSmoothRotation.TRANSFORM])
    
    @set_default_from_args("backbone")
    def get_backbone():
        return "resnet18"