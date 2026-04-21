# Copyright 2026-present, Amogh Inamdar, Vici Milenia, Richard Zemel.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Few-shot adaptation utilities for post-hoc checkpoint evaluation.

This module provides functionality to temporarily adapt a model on k-shot examples
via gradient descent, then evaluate it, while preserving the original model state.
"""

import copy
import logging
from typing import List, Optional, Tuple, Union
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, Dataset
from tqdm import tqdm
from datasets.utils.continual_dataset import ContinualDataset
from models.utils.continual_model import ContinualModel


def create_k_shot_eval_dataset(dataset: ContinualDataset, task_id: int) -> ContinualDataset:
    """
    Create a fresh dataset instance for k-shot evaluation for the specified task.

    This function also caches per-task fresh datasets so that repeated calls
    for different k values do not reload the dataset repeatedly.
    """
    if not hasattr(dataset, '_k_shot_eval_cache'):
        dataset._k_shot_eval_cache = {}

    if task_id in dataset._k_shot_eval_cache:
        return dataset._k_shot_eval_cache[task_id]

    from datasets import get_dataset_class
    dataset_args = dataset.args
    dataset_class = get_dataset_class(dataset_args)

    # Create a fresh dataset for evaluation and advance it to the requested task
    task_dataset = dataset_class(dataset_args)
    for _ in range(task_id + 1):
        task_dataset.get_data_loaders()

    # Create a second fresh dataset to collect all training examples across tasks
    collect_dataset = dataset_class(dataset_args)
    all_data = []
    all_targets = []
    for _ in range(collect_dataset.N_TASKS):
        train_loader, _ = collect_dataset.get_data_loaders()
        wrapped_dataset = train_loader.dataset
        if hasattr(wrapped_dataset, 'data'):
            all_data.append(wrapped_dataset.data.copy() if isinstance(wrapped_dataset.data, np.ndarray) else np.array(wrapped_dataset.data))
            all_targets.extend(list(wrapped_dataset.targets))

    if not all_data:
        raise RuntimeError(f"Could not collect any training data for task {task_id}")

    task_dataset._k_shot_all_data = np.vstack(all_data)
    task_dataset._k_shot_all_targets = np.array(all_targets)
    dataset._k_shot_eval_cache[task_id] = task_dataset
    return task_dataset


def create_k_shot_loader(dataset: ContinualDataset,
                        task_id: int,
                        k: int,
                        num_samples_per_class: Optional[int] = None,
                        batch_size: int = 32,
                        sampling_seed: int = 42) -> DataLoader:
    """
    Create a k-shot data loader for a specific task by sampling k examples per class.

    Args:
        dataset: A dataset prepared for the target task by create_k_shot_eval_dataset
        task_id: The task index to sample from
        k: Number of examples per class (if num_samples_per_class is None)
        num_samples_per_class: Override for number of samples per class (defaults to k)
        batch_size: Batch size for the returned DataLoader
        sampling_seed: Random seed for reproducibility of sampling

    Returns:
        DataLoader with k-shot examples from the specified task
    """
    if k <= 0:
        logging.info(f"k={k} specified, skipping adaptation for task {task_id}")
        return None

    if num_samples_per_class is None:
        num_samples_per_class = k

    if not hasattr(dataset, '_k_shot_all_data') or not hasattr(dataset, '_k_shot_all_targets'):
        dataset = create_k_shot_eval_dataset(dataset, task_id)

    all_data_array = dataset._k_shot_all_data
    all_targets = dataset._k_shot_all_targets

    start_class, end_class = dataset.get_offsets(task_id)
    task_classes = list(range(start_class, end_class))
    logging.info(f"Sampling for task {task_id}, classes {start_class}-{end_class - 1}: {task_classes}")

    sampled_indices = []
    for class_id in task_classes:
        class_mask = all_targets == class_id
        class_indices = np.where(class_mask)[0]
        logging.debug(f"Found {len(class_indices)} examples for class {class_id} in task {task_id}")

        num_to_sample = min(num_samples_per_class, len(class_indices))
        if num_to_sample > 0:
            np.random.seed(sampling_seed)
            sampled_class_indices = np.random.choice(class_indices, size=num_to_sample, replace=False)
            sampled_indices.extend(sampled_class_indices)

    if len(sampled_indices) == 0:
        logging.warning(f"No examples found for task {task_id} with k={k}")
        return None

    sampled_data = all_data_array[sampled_indices]
    sampled_targets = all_targets[sampled_indices]

    class SimpleDataset(Dataset):
        def __init__(self, data, targets, transform=None):
            self.data = data
            self.targets = targets
            self.transform = transform

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            img = self.data[idx]
            target = self.targets[idx]

            # Keep as ndarray for transforms; convert to PIL only for ToTensor fallback
            if isinstance(img, Image.Image):
                img_np = np.array(img, dtype=np.uint8)
            else:
                img_np = img.astype(np.uint8)

            if self.transform:
                img_aug = self.transform(img_np)
            else:
                img_aug = transforms.ToTensor()(Image.fromarray(img_np))

            img_noaug = transforms.ToTensor()(Image.fromarray(img_np))

            return img_aug, target, img_noaug

    transform = dataset.get_transform() if hasattr(dataset, 'get_transform') else None
    sampled_dataset = SimpleDataset(sampled_data, sampled_targets, transform=transform)

    loader = DataLoader(sampled_dataset,
                       batch_size=min(len(sampled_indices), batch_size),
                       shuffle=True,
                       num_workers=0,
                       drop_last=False)

    logging.info(f"Created {k}-shot loader for task {task_id} with {len(sampled_indices)} examples "
                f"from {len(task_classes)} classes")
    return loader


def simple_observe(model: ContinualModel, inputs, labels, not_aug_inputs, epoch=None) -> float:
    """
    Simple observe function for few-shot adaptation that just computes the loss and updates the model.

    This is used when the model does not have a meta_observe method or when we want to bypass any special handling.
    """
    model.opt.zero_grad()
    outputs = model.net(inputs)
    loss_fn = model.loss if hasattr(model, 'loss') else torch.nn.functional.cross_entropy
    loss = loss_fn(outputs, labels)
    loss.backward()
    model.opt.step()
    return loss.item()


def adapt_model(model: ContinualModel,
               k_shot_loader: DataLoader,
               num_steps: int = 5,
               learning_rate: float = 0.1,
               task_id: Optional[int] = None,
               use_model_opt: bool = True,
               use_model_observe: bool = True,
               use_model_loss: bool = False) -> ContinualModel:
    """
    Temporarily adapt a model on k-shot examples via gradient descent.

    This function creates a copy of the model, adapts it on the provided examples,
    and returns the adapted model. The original model remains unchanged.

    Args:
        model: The model to adapt
        k_shot_loader: DataLoader with k-shot examples
        num_steps: Number of gradient steps for adaptation
        learning_rate: Learning rate for adaptation
        task_id: Task ID for adaptation (passed to model.observe if needed)
        use_model_opt: Whether to use the model's existing optimizer (if available) or create a new one
        use_model_observe: Whether to call model.observe() for adaptation steps instead of manual optimization
        use_model_loss: Whether to use the model's loss function (if available) instead of default CrossEntropyLoss

    Returns:
        Adapted model (original model is unchanged)
    """
    if k_shot_loader is None or len(k_shot_loader) == 0:
        # No adaptation needed, return copy of original model
        return copy.deepcopy(model)

    # Create a deep copy of the model to avoid modifying the original
    adapted_model = copy.deepcopy(model)

    # Set up optimizer for adaptation (only optimize certain parameters)
    # For few-shot adaptation, we typically adapt the classifier head and/or task-specific parameters
    if hasattr(adapted_model, 'opt') and use_model_opt:
        # Use the same optimizer type but with adaptation-specific settings
        optimizer_class = type(adapted_model.opt)
        adapted_model.opt = optimizer_class(
            adapted_model.net.parameters(),
            lr=learning_rate,
            weight_decay=0.0  # No regularization for few-shot adaptation
        )
    else:
        # Fallback: create SGD optimizer
        adapted_model.opt = torch.optim.SGD(
            adapted_model.net.parameters(),
            lr=learning_rate,
            weight_decay=0.0
        )

    # Use simple CE loss for models such as DER++
    model_loss = copy.deepcopy(adapted_model.loss) if hasattr(adapted_model, 'loss') else None
    if not use_model_loss:
        adapted_model.loss = torch.nn.functional.cross_entropy 

    # Set model to training mode for adaptation
    adapted_model.net.train()

    logging.info(f"Adapting model for {num_steps} steps with lr={learning_rate}")

    # Perform adaptation steps
    pbar = tqdm(range(num_steps), desc="Adapting model")
    for step in pbar:
        total_loss = 0.0
        num_batches = 0

        for batch_data in k_shot_loader:
            inputs, labels, not_aug_inputs = batch_data[0], batch_data[1], batch_data[2]
            inputs, labels = inputs.to(adapted_model.device), labels.to(adapted_model.device, dtype=torch.long)
            not_aug_inputs = not_aug_inputs.to(adapted_model.device)

            if use_model_observe and hasattr(adapted_model, 'observe'):
                loss = adapted_model.observe(inputs, labels, not_aug_inputs, epoch=0)
            else:
                loss = simple_observe(adapted_model, inputs, labels, not_aug_inputs, epoch=0)

            total_loss += loss
            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        pbar.set_postfix(loss=f"{avg_loss:.4f}")
        if step % max(1, num_steps // 5) == 0:  # Log every 20% of steps
            logging.debug(f"Adaptation step {step+1}/{num_steps}, avg loss: {avg_loss:.4f}")

    # Set model back to eval mode
    adapted_model.net.eval()

    # Restore model's original loss function
    adapted_model.loss = model_loss

    logging.info(f"Model adaptation completed. Final avg loss: {avg_loss:.4f}")
    return adapted_model


def evaluate_adapted_model(model: ContinualModel,
                          dataset: ContinualDataset,
                          task_id: int,
                          return_loss: bool = True) -> Union[float, Tuple[float, float]]:
    """
    Evaluate an adapted model on a specific task.

    Args:
        model: The adapted model to evaluate
        dataset: The continual dataset
        task_id: The task to evaluate on
        return_loss: Whether to return loss in addition to accuracy

    Returns:
        Accuracy (and loss if requested) for the specified task
    """
    from utils.evaluate import evaluate

    task_dataset = create_k_shot_eval_dataset(dataset, task_id)

    original_test_loaders = task_dataset.test_loaders
    task_dataset.test_loaders = [task_dataset.test_loaders[task_id]]

    try:
        accs, accs_mask_classes, losses = evaluate(model, task_dataset, last=False, return_per_task_loss=return_loss)
        accuracy, loss = accs[0], losses[0] if return_loss else None
        logging.info(f"Evaluation on task {task_id}: accuracy={accuracy:.4f}" + (f", loss={loss:.4f}" if return_loss else ""))
        return (accuracy, loss) if return_loss else accuracy
    finally:
        task_dataset.test_loaders = original_test_loaders