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
import os
from pathlib import Path
from typing import List, Optional, Tuple, Union
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, Dataset
from tqdm import tqdm
from datasets.utils.continual_dataset import ContinualDataset
from models.utils.continual_model import ContinualModel


def _snapshot_dataset_state(dataset: ContinualDataset) -> ContinualDataset:
    """
    Shallow-copy a dataset together with its own containers (lists/dicts/sets), so that
    advancing the original with further get_data_loaders() calls leaves the snapshot in
    the state it was copied in. The (large) data arrays and the loaders are shared.
    """
    snapshot = copy.copy(dataset)
    for name, value in vars(dataset).items():
        if isinstance(value, (list, dict, set)):
            setattr(snapshot, name, copy.copy(value))
    return snapshot


def _build_k_shot_eval_datasets(dataset: ContinualDataset) -> dict:
    """
    Build the k-shot evaluation dataset of every task in a single pass over the task sequence.

    A fresh dataset is advanced one task at a time; after the i-th get_data_loaders() call
    (which yields task i, verified for both class-IL and domain-IL) its state is snapshotted
    as the evaluation dataset of task i, and that task's training examples are collected.
    This costs N_TASKS get_data_loaders() calls in total, instead of rebuilding the dataset
    from scratch for every task (which is quadratic in the number of tasks and dominated
    evaluation time).

    Each returned dataset also carries all training examples across tasks, together with the
    task each example belongs to. The latter is essential for domain-IL datasets (e.g.
    smooth-rot-mnist), where every task shares the same classes and only the input
    distribution (e.g. rotation) changes: class-based sampling alone would draw examples
    from arbitrary tasks/domains, so sampling must be restricted to the requested task.
    """
    from datasets import get_dataset_class
    dataset_args = dataset.args
    builder = get_dataset_class(dataset_args)(dataset_args)

    task_datasets = {}
    all_data = []
    all_targets = []
    all_task_ids = []
    for task_id in range(builder.N_TASKS):
        train_loader, _ = builder.get_data_loaders()
        task_datasets[task_id] = _snapshot_dataset_state(builder)
        wrapped_dataset = train_loader.dataset
        if hasattr(wrapped_dataset, 'data'):
            chunk = wrapped_dataset.data.copy() if isinstance(wrapped_dataset.data, np.ndarray) else np.array(wrapped_dataset.data)
            all_data.append(chunk)
            all_targets.extend(list(wrapped_dataset.targets))
            all_task_ids.extend([task_id] * len(chunk))

    if not all_data:
        raise RuntimeError("Could not collect any training data for k-shot evaluation")

    # Image datasets store pixel arrays (N,H,W[,C]) -> vstack; path-based datasets
    # (e.g. seq-imagenet-r) store 1-D arrays of file paths -> concatenate.
    all_data = np.concatenate(all_data) if all_data[0].ndim == 1 else np.vstack(all_data)
    all_targets = np.array(all_targets)
    all_task_ids = np.array(all_task_ids)

    for task_dataset in task_datasets.values():
        task_dataset._k_shot_all_data = all_data
        task_dataset._k_shot_all_targets = all_targets
        task_dataset._k_shot_all_task_ids = all_task_ids

        # Some datasets expose train examples that were deliberately EXCLUDED from
        # training (e.g. seq-imagenet-r with IMAGENET_R_TRAIN_CAP): unseen by every
        # checkpoint, they are the preferred k-shot support source, freeing the whole
        # test set for evaluation. The arrays cover all classes; per-task filtering
        # happens at sampling time via get_offsets.
        if hasattr(wrapped_dataset, 'heldout_data'):
            task_dataset._k_shot_heldout_data = np.asarray(wrapped_dataset.heldout_data)
            task_dataset._k_shot_heldout_targets = np.asarray(wrapped_dataset.heldout_targets)

    return task_datasets


def create_k_shot_eval_dataset(dataset: ContinualDataset, task_id: int) -> ContinualDataset:
    """
    Return a dataset instance for k-shot evaluation, advanced to the specified task.

    The per-task datasets of the whole sequence are built once (see
    :func:`_build_k_shot_eval_datasets`) and cached on ``dataset``, so repeated calls for
    different tasks and k values do not reload the dataset.
    """
    if not hasattr(dataset, '_k_shot_eval_cache'):
        dataset._k_shot_eval_cache = _build_k_shot_eval_datasets(dataset)

    return dataset._k_shot_eval_cache[task_id]


# Number of each class's held-out TEST examples reserved as the support pool, from
# which k-shot adaptation examples are drawn. Everything else becomes the (fixed)
# query set used for evaluation. Reserving a small fixed count (rather than a large
# fraction) keeps the query set as large as possible for a better test signal, while
# still covering the k values we use (k <= this value). Because this partition does
# not depend on k, the evaluation (query) set is identical across all k values,
# keeping k-shot curves comparable.
KSHOT_SUPPORT_POOL_SIZE = 10


def get_test_support_query_split(task_dataset: ContinualDataset,
                                 task_id: int,
                                 sampling_seed: int = 42):
    """
    Deterministically partition a task's held-out TEST set, per class, into a
    support pool (for k-shot adaptation) and a disjoint query pool (for evaluation).

    Sampling the k-shot support from the test split -- rather than from the training
    data the checkpoint was trained on -- ensures adaptation happens on examples the
    model has never seen, so the measured few-shot gain reflects the representation
    rather than re-fitting already-memorized training examples. The query pool is
    disjoint from every possible support set and does not depend on k, so evaluation
    is leakage-free and comparable across k values.

    The result is cached on ``task_dataset`` per ``(task_id, sampling_seed)``.

    Args:
        task_dataset: A dataset advanced to ``task_id`` (as returned by
            :func:`create_k_shot_eval_dataset`), so ``test_loaders[task_id]`` exists.
        task_id: The task whose test set is partitioned.
        sampling_seed: Seed controlling the (deterministic) partition.

    Returns:
        (support_pool, query_pool) where ``support_pool`` maps class id -> a
        (shuffled) array of up to ``KSHOT_SUPPORT_POOL_SIZE`` reserved support
        indices, and ``query_pool`` is a sorted array of all remaining (held-out
        evaluation) indices. Indices are into the dataset underlying
        ``task_dataset.test_loaders[task_id]``.
    """
    cache = task_dataset.__dict__.setdefault('_k_shot_tt_split', {})
    key = (task_id, sampling_seed)
    if key in cache:
        return cache[key]

    test_ds = task_dataset.test_loaders[task_id].dataset
    targets = np.asarray(test_ds.targets)
    start_class, end_class = task_dataset.get_offsets(task_id)

    rng = np.random.RandomState(sampling_seed)
    support_pool = {}
    query_pool = []
    for class_id in range(start_class, end_class):
        class_indices = np.where(targets == class_id)[0]
        perm = rng.permutation(class_indices)
        # Cap support at half the class's test examples so that classes with few test
        # images (e.g. 8 ImageNet-R classes have <=10) still contribute to the query
        # set instead of being silently excluded from evaluation.
        n_support = min(KSHOT_SUPPORT_POOL_SIZE, len(perm) // 2)
        support_pool[class_id] = perm[:n_support]  # reserved for support, already shuffled
        query_pool.extend(perm[n_support:].tolist())  # everything else -> evaluation

    query_pool = np.array(sorted(query_pool), dtype=int)
    cache[key] = (support_pool, query_pool)
    return support_pool, query_pool


def create_k_shot_loader(dataset: ContinualDataset,
                        task_id: int,
                        k: int,
                        num_samples_per_class: Optional[int] = None,
                        batch_size: int = 32,
                        sampling_seed: int = 42,
                        from_test_split: bool = False) -> DataLoader:
    """
    Create a k-shot data loader for a specific task by sampling k examples per class.

    Args:
        dataset: A dataset prepared for the target task by create_k_shot_eval_dataset
        task_id: The task index to sample from
        k: Number of examples per class (if num_samples_per_class is None)
        num_samples_per_class: Override for number of samples per class (defaults to k)
        batch_size: Batch size for the returned DataLoader
        sampling_seed: Random seed for reproducibility of sampling
        from_test_split: If True, draw the k-shot support from the held-out TEST
            split's support pool (see :func:`get_test_support_query_split`) instead of
            from the training data. Use this for post-hoc evaluation so adaptation
            happens on examples the checkpoint never trained on; the evaluator
            (:func:`evaluate_adapted_model`) must then be called with the matching
            ``from_test_split`` / ``sampling_seed`` so it evaluates on the disjoint
            query pool. Leave False (default) for meta-training, where sampling from
            the training data is intended.

    Returns:
        DataLoader with k-shot examples from the specified task
    """
    if k <= 0:
        logging.info(f"k={k} specified, skipping adaptation for task {task_id}")
        return None

    if num_samples_per_class is None:
        num_samples_per_class = k

    if (not hasattr(dataset, '_k_shot_all_data') or not hasattr(dataset, '_k_shot_all_targets')
            or not hasattr(dataset, '_k_shot_all_task_ids')):
        dataset = create_k_shot_eval_dataset(dataset, task_id)

    start_class, end_class = dataset.get_offsets(task_id)
    task_classes = list(range(start_class, end_class))
    logging.info(f"Sampling for task {task_id}, classes {start_class}-{end_class - 1} "
                 f"(source={'test-support' if from_test_split else 'train'}): {task_classes}")

    if from_test_split and hasattr(dataset, '_k_shot_heldout_data'):
        # Preferred source: train examples deliberately EXCLUDED from training (e.g.
        # via IMAGENET_R_TRAIN_CAP). Unseen by the checkpoint, and their use leaves
        # the entire test set for evaluation. Deterministic shuffle + prefix keeps
        # draws nested across k, mirroring the other support sources.
        source_data = dataset._k_shot_heldout_data
        source_targets = dataset._k_shot_heldout_targets
        logging.info(f"Task {task_id}: sampling k-shot support from held-out (excluded-from-train) pool "
                     f"({len(source_targets)} examples total)")

        rng = np.random.RandomState(sampling_seed)
        sampled_indices = []
        for class_id in task_classes:
            class_indices = rng.permutation(np.where(source_targets == class_id)[0])
            pool = class_indices[:KSHOT_SUPPORT_POOL_SIZE]
            num_to_sample = min(num_samples_per_class, len(pool))
            if num_to_sample < num_samples_per_class:
                logging.warning(f"Task {task_id} class {class_id}: held-out pool has only {len(pool)} "
                                f"examples, requested {num_samples_per_class}")
            sampled_indices.extend(pool[:num_to_sample].tolist())
    elif from_test_split:
        # Draw support from the held-out test split's support pool (disjoint from the
        # query pool used for evaluation), restricted to this task's own test data.
        support_pool, _ = get_test_support_query_split(dataset, task_id, sampling_seed)
        test_ds = dataset.test_loaders[task_id].dataset
        source_data = np.asarray(test_ds.data)
        source_targets = np.asarray(test_ds.targets)

        sampled_indices = []
        for class_id in task_classes:
            pool = support_pool.get(class_id, np.array([], dtype=int))
            num_to_sample = min(num_samples_per_class, len(pool))
            if num_to_sample < num_samples_per_class:
                logging.warning(f"Task {task_id} class {class_id}: support pool has only {len(pool)} "
                                f"test examples, requested {num_samples_per_class}")
            # pool is pre-shuffled by get_test_support_query_split -> take a deterministic prefix
            sampled_indices.extend(pool[:num_to_sample].tolist())
    else:
        # Draw support from the pooled training data across tasks. Restrict candidates
        # to the requested task's own examples: for class-IL this is implied by the
        # class filter (each class belongs to one task), but for domain-IL datasets it
        # is what ties the sampled examples to the correct input domain (e.g. rotation)
        # instead of pooling examples across every task.
        source_data = dataset._k_shot_all_data
        source_targets = dataset._k_shot_all_targets
        task_mask = dataset._k_shot_all_task_ids == task_id

        sampled_indices = []
        for class_id in task_classes:
            class_indices = np.where((source_targets == class_id) & task_mask)[0]
            logging.debug(f"Found {len(class_indices)} examples for class {class_id} in task {task_id}")

            num_to_sample = min(num_samples_per_class, len(class_indices))
            if num_to_sample > 0:
                np.random.seed(sampling_seed)
                sampled_class_indices = np.random.choice(class_indices, size=num_to_sample, replace=False)
                sampled_indices.extend(sampled_class_indices)

    if len(sampled_indices) == 0:
        logging.warning(f"No examples found for task {task_id} with k={k}")
        return None

    sampled_data = source_data[sampled_indices]
    sampled_targets = source_targets[sampled_indices]

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

            # Path-based datasets (e.g. seq-imagenet-r) store file paths; load to an
            # array so the transform pipeline (which starts with ToPILImage) works.
            from_path = isinstance(img, (str, np.str_))
            if from_path:
                img = np.array(Image.open(img).convert('RGB'))

            if self.transform:
                img_aug = self.transform(img)
            else:
                pil_img = Image.fromarray(img.astype(np.uint8))
                img_aug = transforms.ToTensor()(pil_img)

            pil_img = Image.fromarray(img.astype(np.uint8))
            if from_path:
                # Variable-size source images: resize the not-aug copy to the augmented
                # size so the DataLoader can collate the batch.
                size = img_aug.shape[-2:] if hasattr(img_aug, 'shape') else (224, 224)
                img_noaug = transforms.Compose([
                    transforms.Resize(tuple(size)),
                    transforms.ToTensor(),
                ])(pil_img)
            else:
                img_noaug = transforms.Compose([
                    transforms.ToTensor(),
                ])(pil_img)

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


def _freeze_below_layer(net: torch.nn.Module, layer_min: int) -> Tuple[List[torch.nn.Parameter], List[torch.nn.Parameter]]:
    """
    Freeze all parameters that live before ``layer_min`` and return the split.

    "Layers" are counted as the weight-bearing modules (``Conv2d`` and ``Linear``)
    in registration order, 1-indexed. For a standard ResNet-18 this yields:
    index 1 = the initial ``conv1``, indices 2-20 = the residual-block/shortcut
    convolutions, and index 21 = the ``classifier``. So, e.g., ``layer_min=16``
    freezes everything up to (and including) ``layer3`` and adapts ``layer4``
    onward plus the classifier.

    Each ``BatchNorm`` (or other parameterized) module is associated with the most
    recently seen weight layer, so it is frozen/trained together with that layer.
    Frozen modules are additionally put into ``eval()`` mode so that BatchNorm
    running statistics do not drift during adaptation -- this keeps the earlier
    representation truly fixed.

    Args:
        net: The backbone network to (partially) freeze
        layer_min: Only weight layers with index >= ``layer_min`` are trainable

    Returns:
        (trainable_params, frozen_params)
    """
    running = 0
    trainable: List[torch.nn.Parameter] = []
    frozen: List[torch.nn.Parameter] = []
    frozen_module_names: List[str] = []
    boundary_name: Optional[str] = None

    for name, module in net.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            running += 1
            if running >= layer_min and boundary_name is None:
                boundary_name = name

        direct_params = [p for _, p in module.named_parameters(recurse=False)]
        if not direct_params:
            continue

        if running < layer_min:
            for p in direct_params:
                p.requires_grad_(False)
                frozen.append(p)
            module.eval()  # keep BatchNorm running stats fixed
            frozen_module_names.append(name)
        else:
            for p in direct_params:
                p.requires_grad_(True)
                trainable.append(p)

    if not trainable:
        raise ValueError(
            f"layer_min={layer_min} leaves no trainable parameters: this backbone "
            f"({type(net).__name__}) only has {running} weight layers, so layer_min must be "
            f"<= {running}. Note that layer indices are backbone-specific -- e.g. a ResNet-18 "
            f"has 21 weight layers, but an MNIST MLP has only 3."
        )

    logging.info(f"layer_min={layer_min}/{running}: froze {len(frozen)} params in "
                 f"{len(frozen_module_names)} modules, adapting {len(trainable)} params "
                 f"from '{boundary_name}' onward")
    return trainable, frozen


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
               use_model_loss: bool = False,
               layer_min: int = 0) -> ContinualModel:
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
        layer_min: Only adapt weight layers with (1-indexed, registration-order) index >=
            ``layer_min``, freezing everything earlier. ``0`` (default) adapts the whole
            network. Setting this later (e.g. ``16`` for a ResNet-18, which freezes up to
            ``layer3`` and adapts ``layer4`` + classifier) tests whether few-shot gains
            come from a good frozen representation rather than from re-fitting the backbone.
            See :func:`_freeze_below_layer` for the exact layer-indexing scheme.

    Returns:
        Adapted model (original model is unchanged)
    """
    if k_shot_loader is None or len(k_shot_loader) == 0:
        # No adaptation needed, return copy of original model
        return copy.deepcopy(model)

    # Create a deep copy of the model to avoid modifying the original
    adapted_model = copy.deepcopy(model)

    # Set model to training mode for adaptation (done before freezing so that the
    # subsequent eval() on frozen modules is not overridden).
    adapted_model.net.train()

    # Optionally freeze the early layers so only the later part of the network adapts.
    if layer_min and layer_min > 0:
        trainable_params, _ = _freeze_below_layer(adapted_model.net, layer_min)
    else:
        trainable_params = list(adapted_model.net.parameters())

    # Set up optimizer for adaptation (only optimize the trainable parameters)
    # For few-shot adaptation, we typically adapt the classifier head and/or task-specific parameters
    if hasattr(adapted_model, 'opt') and use_model_opt:
        # Use the same optimizer type but with adaptation-specific settings
        optimizer_class = type(adapted_model.opt)
        adapted_model.opt = optimizer_class(
            trainable_params,
            lr=learning_rate,
            weight_decay=0.0  # No regularization for few-shot adaptation
        )
    else:
        # Fallback: create SGD optimizer
        adapted_model.opt = torch.optim.SGD(
            trainable_params,
            lr=learning_rate,
            weight_decay=0.0
        )

    # Use simple CE loss for models such as DER++
    model_loss = copy.deepcopy(adapted_model.loss) if hasattr(adapted_model, 'loss') else None
    if not use_model_loss:
        adapted_model.loss = torch.nn.functional.cross_entropy

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


def _cached_test_loader(task_dataset: ContinualDataset, task_id: int, batch_size: int = 32) -> DataLoader:
    """
    A DataLoader over the task's test set with all transforms pre-applied, cached
    in-process (reused across k values) and on disk (reused across the parallel
    per-checkpoint eval processes). Valid because test transforms are deterministic.
    Falls back to the live loader if materialization fails.
    """
    cache = task_dataset.__dict__.setdefault('_k_shot_test_tensor_cache', {})
    if task_id in cache:
        return cache[task_id]

    live_loader = task_dataset.test_loaders[task_id]
    cache_dir = Path('data') / 'eval_tensor_cache'
    cache_file = cache_dir / f'{task_dataset.NAME}_task{task_id}_test.pt'

    try:
        if cache_file.exists():
            blob = torch.load(cache_file, weights_only=True)
            xs, ys = blob['x'], blob['y']
        else:
            xs, ys = [], []
            for batch in live_loader:
                xs.append(batch[0].cpu())
                ys.append(batch[1].cpu())
            xs, ys = torch.cat(xs), torch.cat(ys)
            cache_dir.mkdir(parents=True, exist_ok=True)
            tmp = cache_file.with_suffix(f'.tmp{os.getpid()}')
            torch.save({'x': xs, 'y': ys}, tmp)
            tmp.replace(cache_file)  # atomic: concurrent writers just overwrite equally
        loader = DataLoader(torch.utils.data.TensorDataset(xs, ys),
                            batch_size=batch_size, shuffle=False, num_workers=0)
    except Exception as e:
        logging.warning(f"test tensor cache failed for task {task_id} ({e}); using live loader")
        loader = live_loader

    cache[task_id] = loader
    return loader


def evaluate_adapted_model(model: ContinualModel,
                          dataset: ContinualDataset,
                          task_id: int,
                          return_loss: bool = True,
                          from_test_split: bool = False,
                          sampling_seed: int = 42,
                          batch_size: int = 32) -> Union[float, Tuple[float, float]]:
    """
    Evaluate an adapted model on a specific task.

    Args:
        model: The adapted model to evaluate
        dataset: The continual dataset
        task_id: The task to evaluate on
        return_loss: Whether to return loss in addition to accuracy
        from_test_split: If True, evaluate on the held-out query pool that is disjoint
            from the k-shot support pool (see :func:`get_test_support_query_split`).
            Must match the ``from_test_split`` / ``sampling_seed`` used when the k-shot
            support loader was built, so the support examples are excluded from
            evaluation. If False, evaluate on the task's full test set (legacy behavior).
        sampling_seed: Seed identifying the support/query partition (must match the
            value passed to :func:`create_k_shot_loader`).
        batch_size: Batch size for the query loader (only used when ``from_test_split``).

    Returns:
        Accuracy (and loss if requested) for the specified task
    """
    from utils.evaluate import evaluate

    task_dataset = create_k_shot_eval_dataset(dataset, task_id)

    original_test_loaders = task_dataset.test_loaders
    if from_test_split and hasattr(task_dataset, '_k_shot_heldout_data'):
        # Support came from the held-out (excluded-from-train) pool, so the test set
        # contains no support examples and can be used for evaluation in full.
        # Materialize the transformed test tensors once per task and reuse them across
        # every k value (and across checkpoint processes, via a disk cache): the test
        # transform is deterministic, and repeated per-round PIL preprocessing of the
        # full test set dominates eval wall-time on a CPU-loaded machine.
        logging.info(f"Task {task_id}: support is held-out train data; evaluating on the full test set")
        task_dataset.test_loaders = [_cached_test_loader(task_dataset, task_id, batch_size)]
    elif from_test_split:
        # Evaluate only on the query pool (disjoint from any k-shot support example).
        _, query_pool = get_test_support_query_split(task_dataset, task_id, sampling_seed)
        test_ds = task_dataset.test_loaders[task_id].dataset
        query_loader = DataLoader(Subset(test_ds, query_pool.tolist()),
                                  batch_size=batch_size, shuffle=False, num_workers=0)
        logging.info(f"Task {task_id}: evaluating on held-out query pool of {len(query_pool)} examples")
        task_dataset.test_loaders = [query_loader]
    else:
        task_dataset.test_loaders = [task_dataset.test_loaders[task_id]]

    try:
        accs, accs_mask_classes, losses = evaluate(model, task_dataset, last=False, return_per_task_loss=return_loss)
        accuracy, loss = accs[0], losses[0] if return_loss else None
        logging.info(f"Evaluation on task {task_id}: accuracy={accuracy:.4f}" + (f", loss={loss:.4f}" if return_loss else ""))
        return (accuracy, loss) if return_loss else accuracy
    finally:
        task_dataset.test_loaders = original_test_loaders