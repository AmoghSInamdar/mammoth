#!/usr/bin/env python3
"""
Post-hoc evaluation script for continual learning checkpoints.

This script loads saved model checkpoints and evaluates them on specified tasks
with optional few-shot adaptation (k=0,1,2,5,10,... examples per class).

Usage:
    python eval_checkpoints.py --checkpoint_paths path/to/checkpoint1.pt path/to/checkpoint2.pt \
                               --eval_dataset seq-cifar10 \
                               --eval_tasks 0,2,4 \
                               --k_values 0,1,2,5,10 \
                               --output_dir results/
"""

# Copyright 2026-present, Amogh Inamdar, Vici Milenia, Richard Zemel.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import copy
import glob
import importlib
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional
import torch

from utils.multirun_eval_results import MultirunEvaluationResult, MultirunEvaluationResults

# Add mammoth path
if os.path.dirname(__file__) == 'utils':
    mammoth_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
else:
    mammoth_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, mammoth_path)

from datasets.utils.continual_dataset import ContinualDataset
from utils import setup_logging
from utils.checkpoints import mammoth_load_checkpoint
from utils.few_shot import create_k_shot_loader, adapt_model, evaluate_adapted_model, evaluate_per_digit
from utils.eval_results import EvaluationResult, EvaluationResults
from utils.args import add_experiment_args, add_management_args, add_initial_args, add_dynamic_parsable_args
from utils.conf import get_device, get_checkpoint_path
from utils.per_shot_plasticity import add_plasticity_scores_to_csv

# Import after path setup
from datasets import get_dataset_class
from models import get_model_class, get_model
from backbone import get_backbone


def parse_eval_args() -> argparse.Namespace:
    """
    Parse command-line arguments for evaluation.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(description='Evaluate continual learning checkpoints with few-shot adaptation')

    # Stage 1: Initial args (dataset, model, backbone)
    # In evaluation mode, model and dataset may be optional (auto-discovery via checkpoint names)
    add_initial_args(parser, strict=False)

    # Stage 2: Evaluation-specific args
    eval_group = parser.add_argument_group('evaluation')
    eval_group.add_argument('--checkpoint_paths', nargs='+', required=False,
                           help='Paths to checkpoint files (supports glob patterns). If not provided, will auto-discover based on --model and --dataset')
    eval_group.add_argument('--task_indices', type=str, default='all',
                           help='Task indices to evaluate (comma-separated, e.g., "0,2,4" or "all"). Only used when auto-discovering checkpoints.')
    eval_group.add_argument('--eval_dataset', type=str,
                           help='Dataset for evaluation (if different from training dataset)')
    eval_group.add_argument('--eval_tasks', type=str, default='all',
                           help='Task indices for evaluation (comma-separated, e.g., "0,2,4" or "all")')
    eval_group.add_argument('--k_values', type=str, default='0',
                           help='k values for few-shot evaluation (comma-separated, e.g., "0,1,2,5,10")')
    eval_group.add_argument('--num_adapt_steps', type=int, default=5,
                           help='Number of gradient steps for k-shot adaptation')
    eval_group.add_argument('--adapt_lr', type=float, default=0.1,
                           help='Learning rate for k-shot adaptation')
    eval_group.add_argument('--output_dir', type=str, default='eval_results',
                           help='Directory to save evaluation results')
    eval_group.add_argument('--custom_metric_module', type=str,
                           help='Path to custom metric module for aggregation')
    eval_group.add_argument('--n_seeds', type=int, default=10,
                           help='num of random seeds for evaluation')
    eval_group.add_argument('--multirun_temp_csv_dir', type=str, default='eval_results_multirun',
                            help='Directory to save temp multirun evaluation results csv')
    eval_group.add_argument('--icl_backend', type=str, default=None,
                            choices=['clip', 'dinov2', 'vit', 'vlm', 'qwen2vl', 'llava'],
                            help='If set, measure SAUCE via in-context few-shot classification '
                                 'instead of gradient-descent adaptation of a checkpoint. The k '
                                 'shots become in-context examples. clip/dinov2/vit = frozen-feature '
                                 'few-shot; qwen2vl/llava = local open-source VLMs (true ICL); '
                                 'vlm = Claude API (true ICL). Requires --dataset (or '
                                 '--eval_dataset); no checkpoint is loaded.')
    eval_group.add_argument('--vlm_max_queries', type=int, default=50,
                            help='Max query images scored per (task,k,seed) for the generative VLM '
                                 'backends (vlm/qwen2vl/llava) — generation/API cost control.')
    eval_group.add_argument('--icl_anonymize', action='store_true',
                            help='Generative-VLM backends only: replace class names with abstract '
                                 'indices in the prompt, so the model must learn the label mapping '
                                 'purely from the in-context examples (ICL ablation vs zero-shot). '
                                 'Output files are suffixed "-anon" to avoid overwriting named runs.')
    eval_group.add_argument('--icl_vlm_batch_size', type=int, default=1,
                            help='Queries scored per forward for the generative VLM backends. '
                                 '>1 amortizes the (dominant) demonstration-token compute across '
                                 'the batch on CUDA/A100 (numerically identical results); keep at 1 '
                                 'on MPS/CPU where batching gives no speedup. Try 8-16 on an A100.')

    # Stage 3: Experiment args (reuse from training)
    add_experiment_args(parser)

    # Stage 4: Management args
    add_management_args(parser)

    args = parser.parse_args()

    # In-context (transformer) mode: no checkpoint is loaded. The k shots are fed
    # as in-context examples to a foundation model. We only need a dataset.
    if args.icl_backend is not None:
        if not (args.dataset or args.eval_dataset):
            parser.error("--icl_backend requires --dataset (or --eval_dataset)")
        args.checkpoint_paths = []  # sentinel: handled by the in-context branch in main()
        # Parse eval_tasks / k_values / seeds and return early (skip checkpoint glob).
        if args.eval_tasks == 'all':
            args.eval_tasks = None
        else:
            try:
                args.eval_tasks = [int(x.strip()) for x in args.eval_tasks.split(',')]
            except ValueError:
                parser.error("--eval_tasks must be 'all' or comma-separated integers")
        try:
            args.k_values = [int(x.strip()) for x in args.k_values.split(',')]
        except ValueError:
            parser.error("--k_values must be comma-separated integers")
        args.seeds = list(range(args.n_seeds))
        return args

    # Auto-discover checkpoints if not provided
    if not args.checkpoint_paths:
        if not args.model or not args.dataset:
            parser.error("--checkpoint_paths is required unless both --model and --dataset are provided for auto-discovery")
        
        # Find checkpoint directory
        checkpoint_dir = get_checkpoint_path()
        if not os.path.exists(checkpoint_dir):
            parser.error(f"Checkpoint directory {checkpoint_dir} does not exist")
        
        # Look for checkpoints with pattern: model_dataset_task.pt
        pattern = f"{args.model}_{args.dataset}_*.pt"
        checkpoint_paths = []
        for file in os.listdir(checkpoint_dir):
            if file.startswith(f"{args.model}_{args.dataset}_") and file.endswith(".pt"):
                checkpoint_paths.append(os.path.join(checkpoint_dir, file))
        
        if not checkpoint_paths:
            parser.error(f"No checkpoints found matching pattern {pattern} in {checkpoint_dir}")
        
        # Sort by task number
        def extract_task_num(path):
            filename = os.path.basename(path)
            # Extract task number from filename like "er_seq-cifar10_5.pt"
            parts = filename.replace('.pt', '').split('_')
            try:
                return int(parts[-1])  # Last part should be task number
            except ValueError:
                return 999  # Put invalid names at the end
        
        checkpoint_paths.sort(key=extract_task_num)
        args.checkpoint_paths = checkpoint_paths
        
        # Filter by task indices if specified
        if args.task_indices != 'all':
            try:
                requested_tasks = [int(x.strip()) for x in args.task_indices.split(',')]
            except ValueError:
                parser.error("--task_indices must be 'all' or comma-separated integers")
            
            filtered_paths = []
            for path in checkpoint_paths:
                task_num = extract_task_num(path)
                if task_num in requested_tasks:
                    filtered_paths.append(path)
            
            if not filtered_paths:
                parser.error(f"No checkpoints found for requested tasks {requested_tasks}")
            
            args.checkpoint_paths = filtered_paths
        
        logging.info(f"Auto-discovered {len(args.checkpoint_paths)} checkpoints: {[os.path.basename(p) for p in args.checkpoint_paths]}")
    
    else:
        # Expand glob patterns in manually provided checkpoint paths
        expanded_paths = []
        for path_pattern in args.checkpoint_paths:
            matches = glob.glob(path_pattern)
            if not matches:
                logging.warning(f"No files found matching pattern: {path_pattern}")
            expanded_paths.extend(matches)
        args.checkpoint_paths = expanded_paths

    if not args.checkpoint_paths:
        parser.error("No checkpoint files found")

    # Parse eval_tasks
    if args.eval_tasks == 'all':
        args.eval_tasks = None  # Will be set after loading dataset
    else:
        try:
            args.eval_tasks = [int(x.strip()) for x in args.eval_tasks.split(',')]
        except ValueError:
            parser.error("--eval_tasks must be 'all' or comma-separated integers")

    # Parse k_values
    try:
        args.k_values = [int(x.strip()) for x in args.k_values.split(',')]
    except ValueError:
        parser.error("--k_values must be comma-separated integers")

    # Generate seeds
    args.seeds = list(range(args.n_seeds))
    
    return args


def load_checkpoint_for_eval(checkpoint_path: str, device: str, eval_args: argparse.Namespace) -> tuple:
    """
    Load a checkpoint and return the model and args.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
        eval_args: Evaluation arguments (may include eval_dataset)

    Returns:
        Tuple of (model, args, checkpoint_info)
    """
    logging.info(f"Loading checkpoint: {checkpoint_path}")

    # First, load only the args from the checkpoint
    ckpt_args = mammoth_load_checkpoint(checkpoint_path, return_only_args=True)
    
    # Use eval_dataset if specified, otherwise use the dataset from checkpoint
    dataset_name = eval_args.eval_dataset if eval_args.eval_dataset else ckpt_args.dataset
    ckpt_args.dataset = dataset_name  # Update for model creation
    
    # Create dataset for model initialization
    dataset_class = get_dataset_class(ckpt_args)
    dataset = dataset_class(ckpt_args)
    
    # Create backbone
    backbone = get_backbone(ckpt_args)
    
    # Get loss and transform
    loss = dataset.get_loss()
    transform = dataset.get_transform()
    
    # Create model
    model = get_model(ckpt_args, backbone, loss, transform, dataset=dataset)
    
    # Now load the full checkpoint with the model
    loaded_checkpoint = mammoth_load_checkpoint(
        checkpoint_path, model=model, return_only_args=False
    )
    if isinstance(loaded_checkpoint, tuple):
        model = loaded_checkpoint[0]
        results = loaded_checkpoint[1] if len(loaded_checkpoint) > 1 else None
    else:
        model = loaded_checkpoint
        results = None

    # Move model to device
    model.to(device)
    model.device = device

    # Extract checkpoint identifier
    checkpoint_id = Path(checkpoint_path).stem

    logging.info(f"Loaded model: {ckpt_args.model}, dataset: {ckpt_args.dataset}, checkpoint: {checkpoint_id}")

    return model, ckpt_args, checkpoint_id


def setup_evaluation_dataset(eval_args: argparse.Namespace, train_args: argparse.Namespace) -> ContinualDataset:
    """
    Set up the dataset for evaluation.

    Args:
        eval_args: Evaluation arguments
        train_args: Training arguments from checkpoint

    Returns:
        Configured dataset for evaluation
    """
    # Determine which dataset to use
    dataset_name = eval_args.eval_dataset or train_args.dataset
    logging.info(f"Using dataset: {dataset_name} for evaluation")

    # Create dataset args by merging eval and train args
    dataset_args = argparse.Namespace(**vars(train_args))
    # Get dataset class and configure
    dataset_class = get_dataset_class(dataset_args)

    # Override with eval-specific settings
    if eval_args.eval_dataset:
        dataset_args.dataset = eval_args.eval_dataset
    dataset_args.device = eval_args.device

    # Initialize dataset
    dataset = dataset_class(dataset_args)

    # Set eval_tasks if not specified
    if eval_args.eval_tasks is None:
        eval_args.eval_tasks = list(range(dataset.N_TASKS))
        logging.info(f"Evaluating on all {dataset.N_TASKS} tasks")
    else:
        logging.info(f"Evaluating on tasks: {eval_args.eval_tasks}")

    return dataset


def evaluate_checkpoint(checkpoint_path: str,
                       eval_args: argparse.Namespace,
                       train_args: argparse.Namespace,
                       dataset: 'ContinualDataset') -> List[EvaluationResult]:
    """
    Evaluate a single checkpoint on specified tasks with different k values.

    Args:
        checkpoint_path: Path to checkpoint
        eval_args: Evaluation arguments
        train_args: Training arguments
        dataset: Evaluation dataset

    Returns:
        List of evaluation results
    """
    results = []

    # Load checkpoint
    model, loaded_args, checkpoint_id = load_checkpoint_for_eval(checkpoint_path, eval_args.device, eval_args)

    # For each evaluation task
    for eval_task_id in eval_args.eval_tasks:
        logging.info(f"Evaluating checkpoint {checkpoint_id} on task {eval_task_id}")

        # For each k value
        for k in eval_args.k_values:
            logging.info(f"\nk={k} adaptation")

            # Create k-shot loader (if k > 0)
            k_shot_loader, adapted_model = None, copy.deepcopy(model)
            if k > 0:
                batch_size = 32 if model.NAME != 'mer' else 1  # MER only supports a batch size of 1
                k_shot_loader = create_k_shot_loader(dataset, eval_task_id, k, batch_size=batch_size)
                if k_shot_loader is None:
                    logging.warning(f"Could not create {k}-shot loader for task {eval_task_id}, skipping")
                    continue

                # Adapt model
                adapted_model = adapt_model(
                    model=model,
                    k_shot_loader=k_shot_loader,
                    num_steps=eval_args.num_adapt_steps if k > 0 else 0,
                    learning_rate=eval_args.adapt_lr,
                    task_id=eval_task_id
                )
                logging.info(f"Adapted model for task {eval_task_id} with k={k}")

            # Evaluate adapted model on the task
            accuracy, loss = evaluate_adapted_model(
                adapted_model, dataset, eval_task_id, return_loss=True
            )

            # Create result
            result = EvaluationResult(
                checkpoint_id=checkpoint_id,
                eval_task_id=eval_task_id,
                k_value=k,
                accuracy=accuracy,
                loss=loss,
                num_adapt_steps=eval_args.num_adapt_steps if k > 0 else 0,
                adapt_lr=eval_args.adapt_lr if k > 0 else None,
                num_examples_used=len(k_shot_loader) * k_shot_loader.batch_size if k_shot_loader else 0
            )

            results.append(result)
            logging.info(f"    Task {eval_task_id}, k={k}: {accuracy:.2f}% accuracy")

    return results

def evaluate_checkpoint_multirun(checkpoint_path: str,
                                  eval_args: argparse.Namespace,
                                  train_args: argparse.Namespace,
                                  dataset: 'ContinualDataset') -> List[MultirunEvaluationResult]:
    results = []

    model, loaded_args, checkpoint_id = load_checkpoint_for_eval(checkpoint_path, eval_args.device, eval_args)

    for eval_task_id in eval_args.eval_tasks:
        logging.info(f"Evaluating checkpoint {checkpoint_id} on task {eval_task_id}")

        for k in eval_args.k_values:
            logging.info(f"\nk={k} adaptation")

            for seed in eval_args.seeds:
                k_shot_loader, adapted_model = None, copy.deepcopy(model)
                if k > 0:
                    batch_size = 32 if model.NAME != 'mer' else 1
                    k_shot_loader = create_k_shot_loader(dataset, eval_task_id, k, batch_size=batch_size, sampling_seed=seed)
                    if k_shot_loader is None:
                        logging.warning(f"Could not create {k}-shot loader for task {eval_task_id}, skipping")
                        continue

                    adapted_model = adapt_model(
                        model=model,
                        k_shot_loader=k_shot_loader,
                        num_steps=eval_args.num_adapt_steps if k > 0 else 0,
                        learning_rate=eval_args.adapt_lr,
                        task_id=eval_task_id
                    )
                    logging.info(f"Adapted model for task {eval_task_id} with k={k}, seed={seed}")

                accuracy, loss = evaluate_adapted_model(adapted_model, dataset, eval_task_id, return_loss=True)
                per_digit = evaluate_per_digit(adapted_model, dataset, eval_task_id) 

                result = MultirunEvaluationResult(
                    checkpoint_id=checkpoint_id,
                    eval_task_id=eval_task_id,
                    k_value=k,
                    accuracy=accuracy,
                    loss=loss,
                    num_adapt_steps=eval_args.num_adapt_steps if k > 0 else 0,
                    adapt_lr=eval_args.adapt_lr if k > 0 else None,
                    num_examples_used=len(k_shot_loader) * k_shot_loader.batch_size if k_shot_loader else 0,
                    seed=seed,
                    metadata={f"digit_{d}_acc": acc for d, acc in per_digit.items()}, 
                )

                results.append(result)
                logging.info(f"    Task {eval_task_id}, k={k}, seed={seed}: {accuracy:.2f}% accuracy")

    return results


def icl_group_tag(backend: str, anonymize: bool) -> str:
    """Backend tag used in checkpoint_id and output filename. The '-anon' suffix
    keeps anonymized-label runs from overwriting the named-label run's CSV."""
    return f"{backend}-anon" if anonymize else backend


def evaluate_in_context_multirun(eval_args: argparse.Namespace,
                                 dataset: 'ContinualDataset') -> List[MultirunEvaluationResult]:
    """
    In-context (transformer) analogue of ``evaluate_checkpoint_multirun``.

    No checkpoint is loaded and there is NO gradient descent: for each
    (eval_task, k, seed) the k support examples per class are fed to the chosen
    foundation model as in-context examples, and accuracy/loss are read off in
    the exact scale ``utils.evaluate.evaluate`` uses (percent, argmax over the
    cumulative ``get_offsets()[1]`` label set). Results carry a single constant
    ``checkpoint_id`` per (backend, dataset) ending in ``_0`` so the SAUCE
    groupby and the plotting's checkpoint-index parsing stay well-formed.
    """
    from utils.in_context_eval import evaluate_in_context

    backend = eval_args.icl_backend
    anonymize = getattr(eval_args, 'icl_anonymize', False)
    dataset_name = eval_args.eval_dataset or eval_args.dataset
    tag = icl_group_tag(backend, anonymize)          # e.g. 'qwen2vl-anon'
    checkpoint_id = f"icl-{tag}_{dataset_name}_0"

    results: List[MultirunEvaluationResult] = []
    for eval_task_id in eval_args.eval_tasks:
        logging.info(f"[icl:{tag}] evaluating task {eval_task_id}")
        for k in eval_args.k_values:
            for seed in eval_args.seeds:
                accuracy, loss, per_class = evaluate_in_context(
                    dataset=dataset,
                    task_id=eval_task_id,
                    k=k,
                    backend=backend,
                    seed=seed,
                    device=eval_args.device,
                    vlm_max_queries=eval_args.vlm_max_queries,
                    anonymize=anonymize,
                    vlm_batch_size=getattr(eval_args, 'icl_vlm_batch_size', 1),
                )
                results.append(MultirunEvaluationResult(
                    checkpoint_id=checkpoint_id,
                    eval_task_id=eval_task_id,
                    k_value=k,
                    accuracy=accuracy,
                    loss=loss,
                    num_adapt_steps=0,          # no gradient steps in the ICL path
                    adapt_lr=None,
                    num_examples_used=k,        # k in-context examples per class
                    seed=seed,
                    metadata={f"digit_{d}_acc": acc for d, acc in per_class.items()},
                ))
    return results


def get_results_group_name(checkpoint_paths: List[str]) -> str:
    """Derive a representative group name from checkpoint stems."""
    checkpoint_ids = [Path(path).stem for path in checkpoint_paths]
    if len(checkpoint_ids) == 1:
        return checkpoint_ids[0]

    common_prefix = os.path.commonprefix(checkpoint_ids)
    common_prefix = common_prefix.rstrip('_-')
    if not common_prefix:
        return checkpoint_ids[0]
    return common_prefix


def main():
    """Main evaluation function."""
    setup_logging()
    logging.info("Starting post-hoc checkpoint evaluation")

    # Parse arguments
    args = parse_eval_args()

    # Set device
    args.device = get_device(args.device)
    is_multirun = len(args.seeds) > 1

    # ----- In-context (transformer) branch: no checkpoint, no gradient descent ----- #
    if args.icl_backend is not None:
        # In-context results are always written as multirun so the aggregated CSV
        # (with accuracy_std) is produced and plasticity runs on it, exactly like
        # the ResNet multirun path. A single seed still yields a valid 1-seed file.
        is_multirun = True
        output_dir = Path(args.multirun_temp_csv_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        dataset = setup_evaluation_dataset(args, args)  # builds dataset from args alone
        dataset.get_data_loaders()                      # sets up test_loaders
        if args.eval_tasks is None:
            args.eval_tasks = list(range(dataset.N_TASKS))

        all_results = MultirunEvaluationResults()
        all_results.add_results(evaluate_in_context_multirun(args, dataset))

        dataset_name = args.eval_dataset or args.dataset
        checkpoint_group_name = f"icl-{icl_group_tag(args.icl_backend, getattr(args, 'icl_anonymize', False))}_{dataset_name}"
        csv_path = output_dir / f"evaluation_results_{checkpoint_group_name}.csv"
        agg_path = output_dir / 'aggregated' / f"evaluation_results_{checkpoint_group_name}.csv"
        agg_path.parent.mkdir(parents=True, exist_ok=True)

        all_results.save_to_csv(csv_path)
        try:
            add_plasticity_scores_to_csv(agg_path, metric_for_sauce='accuracy')
            logging.info(f"Plasticity scores added to {agg_path}")
        except Exception as e:
            logging.warning(f"Failed to compute plasticity scores: {e}")
        logging.info(f"In-context evaluation complete. Results saved to {output_dir}")
        return

    # Create output directory
    output_dir = Path(args.multirun_temp_csv_dir) if is_multirun else Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load first checkpoint to get training args (assume all checkpoints have same structure)
    first_checkpoint = args.checkpoint_paths[0]
    _, train_args, _ = load_checkpoint_for_eval(first_checkpoint, args.device, args)

    # Set up evaluation dataset
    dataset = setup_evaluation_dataset(args, train_args)

    # Initialize dataset (load data)
    dataset.get_data_loaders()  # This sets up test_loaders

    # Evaluate all checkpoints
    all_results = MultirunEvaluationResults() if is_multirun else EvaluationResults()
    eval_fn = evaluate_checkpoint_multirun if is_multirun else evaluate_checkpoint

    for checkpoint_path in args.checkpoint_paths:
        try:
            checkpoint_results = eval_fn(checkpoint_path, args, train_args, dataset)
            all_results.add_results(checkpoint_results)
        except Exception as e:
            logging.error(f"Failed to evaluate checkpoint {checkpoint_path}: {e}")
            raise e
            continue

    # Save results
    checkpoint_group_name = get_results_group_name(args.checkpoint_paths)
    csv_path = output_dir / f"evaluation_results_{checkpoint_group_name}.csv"
    agg_path = output_dir / 'aggregated' / f"evaluation_results_{checkpoint_group_name}.csv"
    agg_path.parent.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / f"evaluation_results_{checkpoint_group_name}.json"

    all_results.save_to_csv(csv_path)
    # all_results.save_to_json(json_path)

    try:
        target_path = agg_path if is_multirun else csv_path
        add_plasticity_scores_to_csv(target_path, metric_for_sauce='accuracy')
        logging.info(f"Plasticity scores added to {target_path}")
    except Exception as e:
        logging.warning(f"Failed to compute plasticity scores: {e}")

    logging.info(f"Evaluation complete. Results saved to {output_dir}")

    # Apply custom metric if provided
    if args.custom_metric_module:
        try:
            # Import custom metric module
            spec = importlib.util.spec_from_file_location("custom_metric", args.custom_metric_module)
            custom_metric_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(custom_metric_module)

            if hasattr(custom_metric_module, 'compute_metric'):
                aggregated = all_results.aggregate_by_metric(custom_metric_module.compute_metric)
                logging.info(f"Custom metric results: {aggregated}")

                # Save aggregated results
                import json
                custom_agg_path = output_dir / "aggregated_metrics.json"
                with open(custom_agg_path, 'w') as f:
                    json.dump(aggregated, f, indent=2)
                logging.info(f"Aggregated metrics saved to {custom_agg_path}")
            else:
                logging.warning("Custom metric module does not have 'compute_metric' function")

        except Exception as e:
            logging.error(f"Failed to apply custom metric: {e}")

    logging.info("Evaluation completed successfully")


if __name__ == '__main__':
    main()