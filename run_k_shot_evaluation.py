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
import csv
import json
import multiprocessing as mp
import os
import shutil
import sys
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

CONTINUAL_MODELS: List[str] = [
    # 'meta-sgd',
    'meta-er',
    # 'meta-ewc',
    # 'meta-derpp',
    # 'sgd',
    # 'er',
    # 'der',
    # 'derpp',
    # 'agem',
    # 'mer',
    # 'ewc-on',
    # 'moe_adapters',
    # 'lwf',
    # 'gdumb'
]

META_CL_METHODS = ['no_meta', 'reptile'] #, 'maml']
META_CL_STRATEGIES = ['parallel'] #, 'sequential']

CONTINUAL_DATASETS: List[str] = [
    # 'seq-mnist',
    'rot-mnist'
    # 'seq-cifar10',
    # 'seq-cifar100',
    # 'seq-cifar100-20task',
    # 'seq-tinyimg',
    # 'std-split-cifar100',
    # 'structured-cifar100',
    # 'structured-tinyimgnet'
]

K_VALUES: List[int] = [0, 1, 2, 5, 10]

CHECKPOINT_DIR = Path('checkpoints')
OUTPUT_DIR = Path('results/k_shot_evaluation')
TEMP_OUTPUT_DIR = Path('results/temp_csvs')
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
    default_settings = settings.get('default', {'adapt_lr': 0.01, 'num_adapt_steps': 5})
    model_settings = settings.get(model, default_settings)

    if adapt_lr is not None:
        model_settings['adapt_lr'] = adapt_lr
    if num_adapt_steps is not None:
        model_settings['num_adapt_steps'] = num_adapt_steps

    return {
        'adapt_lr': model_settings.get('adapt_lr', default_settings['adapt_lr']),
        'num_adapt_steps': model_settings.get('num_adapt_steps', default_settings['num_adapt_steps']),
    }


def get_checkpoint_names(model: str, dataset: str, method=None, strategy=None, checkpoint_dir: Path = CHECKPOINT_DIR) -> List[str]:
    """Return sorted checkpoint paths for a model/dataset pair."""
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint_dir}")

    checkpoint_paths = []
    if model.startswith('meta'):
        pattern = f"{model}_{dataset}_{method}_{strategy}_*.pt"
        checkpoint_paths.extend(sorted(checkpoint_dir.glob(pattern)))
        print(f"Searching for meta-CL checkpoints with pattern: {pattern}")
    else:
        pattern = f"{model}_{dataset}_*.pt"
        checkpoint_paths = sorted(checkpoint_dir.glob(pattern))
    return [str(path) for path in checkpoint_paths]


def get_available_gpus() -> List[str]:
    """Return the list of GPUs from CUDA_VISIBLE_DEVICES, or default to ['0']."""
    raw_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '').strip()
    if raw_visible:
        gpus = [gpu.strip() for gpu in raw_visible.split(',') if gpu.strip()]
        if gpus:
            return gpus
    return ['0']


def run_checkpoint_evaluation(
    checkpoint_path: str,
    model: str,
    dataset: str,
    k_values_csv: str,
    adapt_lr: Optional[float],
    num_adapt_steps: Optional[int],
    output_dir: Path,
    gpu_id: str,
) -> Path:
    """Run eval_checkpoints.py for a single checkpoint on a specific GPU."""
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = gpu_id
    output_dir.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        'eval_checkpoints.py',
        '--checkpoint_paths',
        checkpoint_path,
        '--model',
        model,
        '--eval_dataset',
        dataset,
        '--lr',
        '0.0',
        '--k_values',
        k_values_csv,
        '--adapt_lr',
        str(adapt_lr),
        '--num_adapt_steps',
        str(num_adapt_steps),
        '--output_dir',
        str(output_dir),
    ]

    checkpoint_name = Path(checkpoint_path).stem
    print(
        f"Evaluating checkpoint={checkpoint_name}, model={model}, dataset={dataset}, "
        f"gpu={gpu_id}, k_values={k_values_csv}, adapt_lr={adapt_lr}, num_adapt_steps={num_adapt_steps}"
    )
    subprocess.run(command, check=True, env=env)
    return output_dir / f"evaluation_results_{checkpoint_name}.csv"


def _merge_temp_csvs(final_csv_path: Path, temp_csv_paths: List[Path]) -> None:
    """Merge temporary CSV files into a single aggregate CSV."""
    if not temp_csv_paths:
        raise ValueError("No temporary CSV files found to merge.")

    final_csv_path.parent.mkdir(parents=True, exist_ok=True)
    header_written = False
    with final_csv_path.open('w', newline='', encoding='utf-8') as out_file:
        writer = csv.writer(out_file)
        for temp_csv_path in temp_csv_paths:
            with temp_csv_path.open('r', newline='', encoding='utf-8') as in_file:
                reader = csv.reader(in_file)
                header = next(reader, None)
                if header is None:
                    continue
                if not header_written:
                    writer.writerow(header)
                    header_written = True
                for row in reader:
                    writer.writerow(row)


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

    Runs evaluation in parallel across GPUs using multiprocessing pools.
    """
    if not checkpoint_paths:
        print(f"No checkpoints found for model={model}, dataset={dataset}")
        return

    gpus = get_available_gpus()
    output_dir.mkdir(parents=True, exist_ok=True)
    k_values_csv = ','.join(str(k) for k in k_values)
    adapt_lr = adapt_settings.get('adapt_lr')
    num_adapt_steps = adapt_settings.get('num_adapt_steps')

    tasks = []
    for idx, checkpoint_path in enumerate(checkpoint_paths):
        gpu_id = gpus[idx % len(gpus)]
        temp_dir = TEMP_OUTPUT_DIR / model / dataset / Path(checkpoint_path).stem
        tasks.append((checkpoint_path, model, dataset, k_values_csv, adapt_lr, num_adapt_steps, temp_dir, gpu_id))

    pool_map = []
    pools = {gpu: mp.Pool(processes=max_subprocesses) for gpu in gpus}
    try:
        for task in tasks:
            checkpoint_path, model, dataset, k_values_csv, adapt_lr, num_adapt_steps, temp_dir, gpu_id = task
            pool_map.append(pools[gpu_id].apply_async(
                run_checkpoint_evaluation,
                args=(checkpoint_path, model, dataset, k_values_csv, adapt_lr, num_adapt_steps, temp_dir, gpu_id)
            ))

        for pool in pools.values():
            pool.close()
        for pool in pools.values():
            pool.join()

        temp_csv_paths = [result.get() for result in pool_map]
    finally:
        for pool in pools.values():
            try:
                pool.terminate()
            except Exception:
                pass

    out_filename = "_".join(Path(checkpoint_paths[0]).stem.split('_')[:-1])
    final_csv_path = output_dir / f"evaluation_results_{out_filename}.csv"
    _merge_temp_csvs(final_csv_path, temp_csv_paths)

    # Clean up temporary directories after merging results
    for temp_csv_path in temp_csv_paths:
        temp_dir = temp_csv_path.parent
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)

    print(f"Merged checkpoint evaluations into {final_csv_path}")



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run k-shot checkpoint evaluation with model-specific adaptation settings')
    parser.add_argument('--models', type=str, default=None,
                        help='Comma-separated list of models to evaluate (defaults to internal CONTINUAL_MODELS)')
    parser.add_argument('--datasets', type=str, default=None,
                        help='Comma-separated list of datasets to evaluate (defaults to internal CONTINUAL_DATASETS)')
    parser.add_argument('--k_values', type=str, default=','.join(str(k) for k in K_VALUES),
                        help='Comma-separated list of k values for few-shot adaptation')
    parser.add_argument('--adapt_lr', type=float, default=0.01,
                        help='Override adaptation learning rate from the command line')
    parser.add_argument('--num_adapt_steps', type=int, default=5,
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
            if model.startswith('meta'):
                eval_methods = [args.meta_method] if hasattr(args, 'meta_method') else META_CL_METHODS
                eval_strategies = [args.meta_strategy] if hasattr(args, 'meta_strategy') else META_CL_STRATEGIES
                for method in eval_methods:
                    for strategy in eval_strategies:
                        checkpoint_paths = get_checkpoint_names(
                            model, dataset, method=method, strategy=strategy, checkpoint_dir=args.checkpoint_dir)
                        evaluate_checkpoints_for_k(
                            checkpoint_paths,
                            model,
                            dataset,
                            args.k_values,
                            model_settings,
                            output_dir=args.output_dir,
                            max_subprocesses=args.max_subprocesses,
                        )
            else:
                checkpoint_paths = get_checkpoint_names(model, dataset, checkpoint_dir=args.checkpoint_dir)
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
