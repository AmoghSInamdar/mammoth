#!/usr/bin/env python3
"""
Helper script to batch-run few-shot checkpoint evaluation for continual models.

This script defines:
- a list of continual models to evaluate
- a list of continual datasets to evaluate on
- a list of k-values for adaptation
- a function that discovers checkpoint names from the checkpoints folder
  given a model and dataset, and evaluates them for each k value
- a runner function that iterates every model/dataset combination
"""

import argparse
import json
import sys
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

CONTINUAL_MODELS: List[str] = [
    'sgd',
    'er',
    'der',
    'derpp',
    # 'agem',
    # 'mer',
    'ewc-on',
    # 'moe_adapters',
    # 'lwf',
    # 'gdumb'
]

CONTINUAL_DATASETS: List[str] = [
    # 'seq-mnist',
    # 'seq-cifar10',
    'seq-cifar100',
    # 'seq-tinyimg',
    # 'std-split-cifar100',
    # 'structured-cifar100',
    # 'structured-tinyimgnet'
]

K_VALUES: List[int] = [0, 1, 2, 5, 10]

CHECKPOINT_DIR = Path('checkpoints')
OUTPUT_DIR = Path('results/k_shot_evaluation')
ADAPT_SETTINGS_FILE = Path(__file__).resolve().parent / 'k_shot_adapt_settings.json'


def load_adapt_settings(settings_path: Path = ADAPT_SETTINGS_FILE) -> Dict[str, Any]:
    """Load adaptation parameter settings for each model."""
    if not settings_path.exists():
        raise FileNotFoundError(f"Adapt settings file not found: {settings_path}")
    with settings_path.open('r', encoding='utf-8') as file:
        return json.load(file)


def get_adapt_settings(
    model: str,
    settings: Dict[str, Any],
    adapt_lr: Optional[float] = None,
    num_adapt_steps: Optional[int] = None,
) -> Dict[str, Any]:
    """Return adaptation settings for a specific model, with CLI overrides."""
    default_settings = settings.get('default', {'adapt_lr': 0.001, 'num_adapt_steps': 5})
    model_settings = settings.get(model, default_settings)

    if adapt_lr is not None:
        model_settings['adapt_lr'] = adapt_lr
    if num_adapt_steps is not None:
        model_settings['num_adapt_steps'] = num_adapt_steps

    return {
        'adapt_lr': model_settings.get('adapt_lr', default_settings['adapt_lr']),
        'num_adapt_steps': model_settings.get('num_adapt_steps', default_settings['num_adapt_steps']),
    }


def get_checkpoint_names(model: str, dataset: str, checkpoint_dir: Path = CHECKPOINT_DIR) -> List[str]:
    """Return sorted checkpoint paths for a model/dataset pair."""
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint_dir}")

    pattern = f"{model}_{dataset}_*.pt"
    checkpoint_paths = sorted(checkpoint_dir.glob(pattern))
    return [str(path) for path in checkpoint_paths]


def evaluate_checkpoints_for_k(
    checkpoint_paths: Sequence[str],
    model: str,
    dataset: str,
    k_values: Sequence[int],
    adapt_settings: Dict[str, Any],
    output_dir: Path = OUTPUT_DIR,
    max_subprocesses: int = 10,
) -> None:
    """Evaluate the matching checkpoints for a model/dataset over all requested k values.
    
    Runs evaluation in chunks to limit concurrent subprocesses.
    """
    if not checkpoint_paths:
        print(f"No checkpoints found for model={model}, dataset={dataset}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    k_values_csv = ','.join(str(k) for k in k_values)
    adapt_lr = adapt_settings.get('adapt_lr')
    num_adapt_steps = adapt_settings.get('num_adapt_steps')

    # Process checkpoints in chunks
    for i in range(0, len(checkpoint_paths), max_subprocesses):
        chunk = checkpoint_paths[i:i + max_subprocesses]
        chunk_str = f"{i//max_subprocesses + 1}/{(len(checkpoint_paths) + max_subprocesses - 1) // max_subprocesses}"

        command = [
            sys.executable,
            'eval_checkpoints.py',
            '--checkpoint_paths',
            *chunk,
            '--model',
            model,
            '--eval_dataset',
            dataset,
            '--lr',
            '0.0',  # No learning during evaluation, adaptation is separate
            '--k_values',
            k_values_csv,
            '--adapt_lr',
            str(adapt_lr),
            '--num_adapt_steps',
            str(num_adapt_steps),
            '--output_dir',
            str(output_dir),
        ]

        print(
            f"Evaluating model={model}, dataset={dataset}, chunk {chunk_str}, "
            f"k_values={k_values_csv}, adapt_lr={adapt_lr}, num_adapt_steps={num_adapt_steps}"
        )
        subprocess.run(command, check=True)



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run k-shot checkpoint evaluation with model-specific adaptation settings')
    parser.add_argument('--models', type=str, default=None,
                        help='Comma-separated list of models to evaluate (defaults to internal CONTINUAL_MODELS)')
    parser.add_argument('--datasets', type=str, default=None,
                        help='Comma-separated list of datasets to evaluate (defaults to internal CONTINUAL_DATASETS)')
    parser.add_argument('--k_values', type=str, default=','.join(str(k) for k in K_VALUES),
                        help='Comma-separated list of k values for few-shot adaptation')
    parser.add_argument('--adapt_lr', type=float, default=None,
                        help='Override adaptation learning rate from the command line')
    parser.add_argument('--num_adapt_steps', type=int, default=None,
                        help='Override number of adaptation steps from the command line')
    parser.add_argument('--max_subprocesses', type=int, default=10,
                        help='Maximum number of checkpoints to evaluate per subprocess chunk')
    parser.add_argument('--checkpoint_dir', type=str, default=str(CHECKPOINT_DIR),
                        help='Directory containing checkpoint files')
    parser.add_argument('--output_dir', type=str, default=str(OUTPUT_DIR),
                        help='Directory where evaluation result files should be saved')
    parser.add_argument('--adapt_settings_file', type=str, default=str(ADAPT_SETTINGS_FILE),
                        help='Path to the JSON file containing model-specific adaptation settings')
    args = parser.parse_args()

    if args.models:
        args.models = [model.strip() for model in args.models.split(',') if model.strip()]
    else:
        args.models = CONTINUAL_MODELS

    if args.datasets:
        args.datasets = [dataset.strip() for dataset in args.datasets.split(',') if dataset.strip()]
    else:
        args.datasets = CONTINUAL_DATASETS

    args.k_values = [int(k.strip()) for k in args.k_values.split(',') if k.strip()]
    args.checkpoint_dir = Path(args.checkpoint_dir)
    args.output_dir = Path(args.output_dir)
    args.adapt_settings_file = Path(args.adapt_settings_file)
    return args


def run_all(args: argparse.Namespace) -> None:
    """Run few-shot evaluation for each model/dataset combination."""
    adapt_settings = load_adapt_settings(args.adapt_settings_file)
    for model in args.models:
        model_settings = get_adapt_settings(model, adapt_settings, args.adapt_lr, args.num_adapt_steps)
        for dataset in args.datasets:
            checkpoint_paths = get_checkpoint_names(model, dataset, args.checkpoint_dir)
            evaluate_checkpoints_for_k(
                checkpoint_paths,
                model,
                dataset,
                args.k_values,
                model_settings,
                output_dir=args.output_dir,
                max_subprocesses=args.max_subprocesses,
            )


if __name__ == '__main__':
    run_all(parse_args())
