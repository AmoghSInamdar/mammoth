# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging

from PIL import Image
import numpy as np
import torch
import torchvision.transforms.functional as F


class Rotation(object):
    """
    Defines a fixed rotation for a numpy array.
    """

    def __init__(self, deg_min: int = 0, deg_max: int = 180) -> None:
        """
        Initializes the rotation with a random angle.

        Args:
            deg_min: lower extreme of the possible random angle
            deg_max: upper extreme of the possible random angle
        """
        self.deg_min = deg_min
        self.deg_max = deg_max
        self.degrees = np.random.uniform(self.deg_min, self.deg_max)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Applies the rotation.

        Args:
            x: image to be rotated

        Returns:
            rotated image
        """
        return F.rotate(x, self.degrees)


class FixedRotation(object):
    """
    Defines a fixed rotation for a numpy array.
    """

    def __init__(self, seed: int, deg_min: int = 0, deg_max: int = 180) -> None:
        """
        Initializes the rotation with a random angle.

        Args:
            seed: seed of the rotation
            deg_min: lower extreme of the possible random angle
            deg_max: upper extreme of the possible random angle
        """
        self.seed = seed
        self.deg_min = deg_min
        self.deg_max = deg_max

        np.random.seed(seed)
        self.degrees = np.random.uniform(self.deg_min, self.deg_max)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Applies the rotation.

        Args:
            x: image to be rotated

        Returns:
            rotated image
        """
        return F.rotate(x, self.degrees)


class IncrementalRotation(object):
    """
    Defines an incremental rotation for a numpy array.
    """

    def __init__(self, init_deg: int = 0, increase_per_iteration: float = 0.006) -> None:
        """
        Defines the initial angle as well as the increase for each rotation

        Args:
            init_deg:
            increase_per_iteration:
        """
        self.increase_per_iteration = increase_per_iteration
        self.iteration = 0
        self.degrees = init_deg

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Applies the rotation.

        Args:
            x: image to be rotated

        Returns:
            rotated image
        """
        degs = (self.iteration * self.increase_per_iteration + self.degrees) % 360
        self.iteration += 1
        return F.rotate(x, degs)

    def set_iteration(self, x: int) -> None:
        """
        Set the iteration to a given integer

        Args:
            x: iteration index
        """
        self.iteration = x

class SmoothRotation(object):
    def __init__(self, init_deg: int = 0, increase_in_task: int = 5, increase_between_task=15, max_deg=360) -> None:
        self.increase_in_task = increase_in_task
        self.increase_between_task = increase_between_task
        self.init_deg = init_deg
        self.max_deg = max_deg

    def get_task_range(self, task_id: int):
        start = self.init_deg + (task_id * (self.increase_in_task + self.increase_between_task))
        end = min(start + self.increase_in_task, self.max_deg)
        return start, end

    # Rotating with F.rotate makes run time twice as long due to grid_sample in F.rotate
    def transform_batch_for_task(self, chunk: torch.Tensor, task_id: int, dataset_name: str) -> torch.Tensor:
        start_range, end_range = self.get_task_range(task_id)
        rng = np.random.default_rng(seed=task_id)
        angles = rng.integers(int(start_range), int(end_range) + 1, size=len(chunk))
        logging.info(f"[SmoothRotation][{dataset_name}] Task {task_id} start={start_range} end={end_range} angles=[{angles.min():.2f}, {angles.max():.2f}]")
        return torch.from_numpy(np.stack([
            np.array(Image.fromarray(img.numpy(), mode='L').rotate(float(angle)))
            for img, angle in zip(chunk, angles)
        ]))
    
class NonRandomSmoothRotation(SmoothRotation):
    def __init__(self, init_deg: int = 0, increase_in_task: int = 5, increase_between_task=15, max_deg=360) -> None:
       super().__init__(init_deg, increase_in_task, increase_between_task, max_deg)

    def transform_batch_for_task(self, chunk: torch.Tensor, targets: torch.Tensor, task_id: int, dataset_name: str) -> torch.Tensor:
        start_range, end_range = self.get_task_range(task_id)
        angle_range = list(range(int(start_range), int(end_range) + 1))
        n_angles = len(angle_range)
        
        angles = np.zeros(len(chunk), dtype=int)
        
        for cls in torch.unique(targets):
            cls_mask = (targets == cls).numpy()
            cls_indices = np.where(cls_mask)[0]
            n_samples = len(cls_indices)
            # tile angles evenly across samples for this class
            cls_angles = np.tile(angle_range, n_samples // n_angles + 1)[:n_samples]
            angles[cls_indices] = cls_angles
        
        logging.info(f"[SmoothRotation][{dataset_name}] Task {task_id} start={start_range} end={end_range} angles=[{angles.min()}, {angles.max()}]")
        return torch.from_numpy(np.stack([
            np.array(Image.fromarray(img.numpy(), mode='L').rotate(float(angle)))
            for img, angle in zip(chunk, angles)
        ]))
