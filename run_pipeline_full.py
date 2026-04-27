#!/usr/bin/env python3
"""
Full pipeline for continual learning experiments: train -> evaluate -> plot.

This script orchestrates the complete workflow:
1. Training: Trains a continual learning model on a dataset
2. Evaluation: Performs k-shot evaluation on saved checkpoints
3. Plasticity: Computes plasticity metrics on evaluation results
4. Plotting: Generates visualizations of results and plasticity scores
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

# Add mammoth path for imports
import os
mammoth_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, mammoth_path)

from run_k_shot_evaluation import run_all as run_eval_all, parse_args as parse_eval_args
from plot_k_shot_results import load_evaluation_results, plot_all, plot_k_shot_comparisons, plot_k_shot_improvement, plot_plasticity_comparisons
from utils.per_shot_plasticity import add_plasticity_scores_to_all_csvs


def setup_logging():
    """Set up basic logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def run_training(
    dataset: str,
    model: str,
    lr: float = 0.1,
    n_epochs: int = 50,
    batch_size: int = 32,
    buffer_size: Optional[int] = None,
    savecheck: str = 'task',
    meta_adapt_lr: Optional[float] = None,
    meta_adapt_steps: Optional[float] = None,
    **kwargs
) -> None:
    """Run training using main.py.
    
    Args:
        dataset: Dataset name (e.g., 'seq-cifar100')
        model: Model name (e.g., 'der', 'meta_sgd')
        lr: Learning rate
        n_epochs: Number of training epochs
        batch_size: Batch size for training
        buffer_size: Replay buffer size (optional)
        savecheck: Checkpoint save frequency ('task' or integer)
        device: Device to use ('cuda' or 'cpu')
        **kwargs: Additional model-specific arguments
    """
    try:
        import main
        
        # Build argument list for main.py
        args_list = [
            '--dataset', dataset,
            '--model', model,
            '--lr', str(lr),
            '--savecheck', savecheck,
        ]

        if n_epochs is not None:
            args_list.extend(['--n_epochs', str(n_epochs)])
        if batch_size is not None:
            args_list.extend(['--batch_size', str(batch_size)])
        
        if buffer_size is not None:
            args_list.extend(['--buffer_size', str(buffer_size)])

        if meta_adapt_lr is not None:
            args_list.extend(['--adapt_lr', str(meta_adapt_lr)])
        if meta_adapt_steps is not None:
            args_list.extend(['--num_adapt_steps', str(meta_adapt_steps)])
           
        # Add any additional kwargs
        for key, value in kwargs.items():
            if value is not None:
                args_list.append(f'--{key}')
                args_list.append(str(value))
        
        logging.info(f"Starting training: {' '.join(args_list)}")
        sys.argv = ['main.py'] + args_list
        main.main()
    except Exception as e:
        logging.error(f"Training failed: {e}")
        raise


def run_evaluation(
    models: Optional[str] = None,
    datasets: Optional[str] = None,
    k_values: str = '0,1,2,5,10',
    adapt_lr: float = 0.01,
    seeds: str = '42',
    multirun_csv_dir: str = '',
    num_adapt_steps: int = 5,
    max_subprocesses: int = 10,
    meta_method: Optional[str] = None,
    meta_strategy: Optional[str] = None
) -> None:
    """Run k-shot evaluation on trained checkpoints.
    
    Args:
        models: Comma-separated list of model names (e.g., 'der,meta_sgd')
        datasets: Comma-separated list of dataset names (e.g., 'seq-cifar100')
        k_values: Comma-separated list of k-values for adaptation
        adapt_lr: Adaptation learning rate
        num_adapt_steps: Number of adaptation steps
        max_subprocesses: Max concurrent evaluations per GPU
    """
    try:
        logging.info("Starting k-shot evaluation...")
        
        # Create eval args namespace
        class EvalArgs:
            pass
        
        args = EvalArgs()
        args.models = models
        args.datasets = datasets
        args.k_values = k_values
        args.adapt_lr = adapt_lr
        args.num_adapt_steps = num_adapt_steps
        args.max_subprocesses = max_subprocesses
        args.checkpoint_dir = Path('checkpoints')
        args.output_dir = Path('results/k_shot_evaluation')
        args.seeds = seeds
        args.multirun_output_dir = Path(multirun_csv_dir)
        args.adapt_settings_file = Path(__file__).resolve().parent / 'k_shot_adapt_settings.json'
        if meta_method is not None:
            args.meta_method = meta_method
        if meta_strategy is not None:
            args.meta_strategy = meta_strategy

        # Parse to proper format
        if args.models:
            args.models = [m.strip() for m in args.models.split(',') if m.strip()]
        if args.datasets:
            args.datasets = [d.strip() for d in args.datasets.split(',') if d.strip()]
        args.k_values = [int(k.strip()) for k in args.k_values.split(',') if k.strip()]
        
        run_eval_all(args)
        logging.info("Evaluation completed successfully")
    except Exception as e:
        logging.error(f"Evaluation failed: {e}")
        raise


def compute_plasticity(metric: str = 'loss') -> None:
    """Compute plasticity metrics on evaluation results.
    
    Args:
        metric: Metric to use for SAUCE computation ('accuracy' or 'loss')
    """
    try:
        logging.info("Computing plasticity metrics...")
        add_plasticity_scores_to_all_csvs(metric_for_sauce=metric)
        logging.info("Plasticity metrics computed successfully")
    except Exception as e:
        logging.error(f"Plasticity computation failed: {e}")
        raise


def run_plotting(
    dataset: Optional[str] = None,
) -> None:
    """Generate plots for evaluation results.
    
    Args:
        metric: Metric to plot ('accuracy' or 'loss')
        dataset: Optional dataset name to filter plots
    """
    try:
        logging.info("Generating plots...")
        
        # Plot k-shot results
        plot_all(metric='accuracy', dataset=dataset)
        plot_all(metric='loss', dataset=dataset)
        plot_k_shot_comparisons(dataset=dataset, metric='accuracy')
        plot_k_shot_comparisons(dataset=dataset, metric='loss')
        plot_k_shot_improvement(dataset=dataset, metric='accuracy')
        plot_k_shot_improvement(dataset=dataset, metric='loss')

        # Plot plasticity comparisons
        if dataset:
            logging.info(f"Plotting plasticity comparisons for {dataset}...")
            plot_plasticity_comparisons(dataset=dataset)
        else:
            logging.info("Plotting plasticity comparisons for all datasets...")
            plot_plasticity_comparisons()
        
        logging.info("Plotting completed successfully")
    except Exception as e:
        logging.error(f"Plotting failed: {e}")
        raise

def run_plotting_multirun(
    dir: str,
    dataset: Optional[str] = None,
) -> None:
    """Generate plots for multirun evaluation results, aggregating across seeds."""
    try:
        logging.info("Generating multirun plots...")

        csv_dir = Path(dir)

        # Now run all existing plot functions against the aggregated directory
        plot_all(metric='accuracy', dataset=dataset, results_dir=csv_dir)
        plot_all(metric='loss', dataset=dataset, results_dir=csv_dir)

        if dataset:
            plot_k_shot_comparisons(dataset=dataset, metric='accuracy', results_dir=csv_dir)
            plot_k_shot_comparisons(dataset=dataset, metric='loss', results_dir=csv_dir)
            plot_k_shot_improvement(dataset=dataset, metric='accuracy', results_dir=csv_dir)
            plot_k_shot_improvement(dataset=dataset, metric='loss', results_dir=csv_dir)
        else:
            logging.warning("--dataset not specified; skipping k-shot comparison and improvement plots")

        plot_plasticity_comparisons(dataset=dataset, results_dir=csv_dir)

        logging.info("Multirun plotting completed successfully")
    except Exception as e:
        logging.error(f"Multirun plotting failed: {e}")
        raise


def run_pipeline(
    dataset: str,
    model: str,
    do_train: bool = True,
    do_eval: bool = True,
    do_plot: bool = True,
    do_plot_per_seed: bool = False,
    # Training args
    lr: float = 0.1,
    n_epochs: int = 50,
    batch_size: int = 32,
    buffer_size: Optional[int] = None,
    savecheck: str = 'task',
    device: str = 'cuda',
    # Evaluation args
    k_values: str = '0,1,2,5,10',
    adapt_lr: float = 0.01,
    num_adapt_steps: int = 5,
    max_subprocesses: int = 10,
    seeds: str = '42',
    multirun_csv_dir: str = '',
    # Plotting args
    plot_metric: str = 'accuracy',
    plasticity_metric: str = 'loss',
    # Additional model args (passed via **kwargs)
    **kwargs
) -> None:
    """Run the complete continual learning pipeline.
    
    Args:
        dataset: Dataset name (e.g., 'seq-cifar100', 'rot-mnist')
        model: Model name (e.g., 'der', 'meta_sgd')
        do_train: Whether to train a model
        do_eval: Whether to evaluate trained checkpoints
        do_plot: Whether to generate plots
        do_plot_per_seed: Whether to generate multirun plots
        multirun_output_dir: Output directory for multirun plots
        lr: Training learning rate
        n_epochs: Number of training epochs
        batch_size: Training batch size
        buffer_size: Replay buffer size (optional)
        savecheck: Checkpoint save frequency
        device: Device to use
        k_values: K-values for few-shot evaluation
        adapt_lr: Adaptation learning rate
        num_adapt_steps: Adaptation steps
        max_subprocesses: Max concurrent GPU evaluations
        seeds: Comma-separated random seeds for multirun evaluation (e.g., '42,1,2')
        plot_metric: Metric to plot
        plasticity_metric: Metric for plasticity computation
        **kwargs: Additional model-specific arguments
    """
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting pipeline for {model} on {dataset}")
    logger.info(f"  Train: {do_train}, Eval: {do_eval}, Plot: {do_plot}")
    
    try:
        # Step 1: Training
        if do_train:
            logger.info(f"Step 1/3: Training {model} on {dataset}")
            run_training(
                dataset=dataset,
                model=model,
                lr=lr,
                n_epochs=n_epochs,
                batch_size=batch_size,
                buffer_size=buffer_size,
                savecheck=savecheck,
                **kwargs
            )
            logger.info("✓ Training completed")
        else:
            logger.info("Step 1/3: Skipping training")
        
        # Step 2: Evaluation
        if do_eval:
            logger.info(f"Step 2/3: Evaluating {model} on {dataset}")
            model_name_for_eval = model.replace('_', '-')                
            run_evaluation(
                models=model_name_for_eval,
                datasets=dataset,
                k_values=k_values,
                adapt_lr=adapt_lr,
                seeds=seeds,
                multirun_csv_dir=multirun_csv_dir,
                num_adapt_steps=num_adapt_steps,
                max_subprocesses=max_subprocesses,
                meta_method=kwargs['meta_method'] if 'meta' in model else None,
                meta_strategy=kwargs['meta_strategy'] if 'meta' in model else None
            )
            logger.info("✓ Evaluation completed")
            
            # Compute plasticity
            logger.info(f"Step 2b/3: Computing plasticity metrics")
            compute_plasticity(metric=plasticity_metric)
            logger.info("✓ Plasticity computation completed")
        else:
            logger.info("Step 2/3: Skipping evaluation")
        
        # Step 3: Plotting
        if do_plot and not do_plot_per_seed:
            logger.info(f"Step 3/3: Plotting results")
            run_plotting(dataset=dataset)
            logger.info("✓ Plotting completed")
        elif do_plot_per_seed:
            logger.info(f"Step 3/3: Plotting results, grouped by random seed")
            run_plotting_multirun(dir=multirun_output_dir, dataset=dataset)
            logger.info(f"Step 3b/3: Plotting aggregated results (mean and stdev) over each random seed")
            run_plotting_multirun(dir=f"{multirun_output_dir}/aggregated", dataset=dataset)
            logger.info("✓ Plotting completed")
        else:
            logger.info("Step 3/3: Skipping plotting")
        
        logger.info("=" * 60)
        logger.info("Pipeline completed successfully!")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


def main() -> None:
    """Parse command-line arguments and run the pipeline."""
    parser = argparse.ArgumentParser(
        description='Run complete continual learning pipeline: train -> evaluate -> plot'
    )
    
    # Required arguments
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset name (e.g., seq-cifar100, rot-mnist)')
    parser.add_argument('--model', type=str, required=True,
                        help='Model name (e.g., der, meta_sgd)')
    
    # Pipeline flags
    parser.add_argument('--do_train', action='store_true', default=True,
                        help='Run training phase')
    parser.add_argument('--skip_train', action='store_false', dest='do_train',
                        help='Skip training phase')
    parser.add_argument('--do_eval', action='store_true', default=True,
                        help='Run evaluation phase')
    parser.add_argument('--skip_eval', action='store_false', dest='do_eval',
                        help='Skip evaluation phase')
    parser.add_argument('--do_plot', action='store_true', default=True,
                        help='Run plotting phase')
    parser.add_argument('--skip_plot', action='store_false', dest='do_plot',
                        help='Skip plotting phase')
    parser.add_argument('--do_plot_per_seed', action='store_false', dest='do_plot_per_seed',
                        help='Group by seed during plotting phase')
    
    # Training arguments
    parser.add_argument('--lr', type=float, default=0.1,
                        help='Training learning rate')
    parser.add_argument('--n_epochs', type=int, required=False,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, required=False,
                        help='Batch size for training')
    parser.add_argument('--buffer_size', type=int, default=None,
                        help='Replay buffer size (for methods like DER, ER)')
    parser.add_argument('--savecheck', type=str, default='task',
                        help='Checkpoint save frequency')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    
    # Evaluation arguments
    parser.add_argument('--k_values', type=str, default='0,1,2,5,10',
                        help='K-values for few-shot adaptation')
    parser.add_argument('--adapt_lr', type=float, default=0.01,
                        help='Adaptation learning rate')
    parser.add_argument('--num_adapt_steps', type=int, default=5,
                        help='Number of adaptation steps')
    parser.add_argument('--max_subprocesses', type=int, default=10,
                        help='Max concurrent evaluations per GPU')
    parser.add_argument('--seeds', type=str, default='42',
                        help='random seeds for evaluation (comma-separated, e.g., "0,42,123,1234,1")')
    parser.add_argument('--multirun_csv_dir', type=str, default='results/k_shot_evaluation_multirun',
                        help='Directory to save results csv generated from multiple runs with different seeds')
    
    # Plotting arguments
    parser.add_argument('--plot_metric', type=str, default='accuracy',
                        choices=['accuracy', 'loss'],
                        help='Metric to plot')
    parser.add_argument('--plasticity_metric', type=str, default='loss',
                        choices=['accuracy', 'loss'],
                        help='Metric for plasticity computation')
    
    # Parse known args and capture any additional model-specific arguments
    args, unknown_args = parser.parse_known_args()
    
    # Convert unknown args to keyword arguments for training
    # e.g., ['--meta-method', 'maml', '--alpha', '0.3'] -> {'meta_method': 'maml', 'alpha': '0.3'}
    kwargs = {}
    i = 0
    while i < len(unknown_args):
        if unknown_args[i].startswith('--'):
            key = unknown_args[i][2:].replace('-', '_')  # Remove -- and convert - to _
            # Check if next arg is a value or another flag
            if i + 1 < len(unknown_args) and not unknown_args[i + 1].startswith('--'):
                value = unknown_args[i + 1]
                # Try to convert to appropriate type
                try:
                    # Try int first
                    kwargs[key] = int(value)
                except ValueError:
                    try:
                        # Try float
                        kwargs[key] = float(value)
                    except ValueError:
                        # Keep as string
                        kwargs[key] = value
                i += 2
            else:
                # Flag without value (boolean flag)
                kwargs[key] = True
                i += 1
        else:
            i += 1
    
    # Run pipeline
    run_pipeline(
        dataset=args.dataset,
        model=args.model,
        do_train=args.do_train,
        do_eval=args.do_eval,
        do_plot=args.do_plot,
        do_plot_per_seed=args.do_plot_per_seed,
        multirun_csv_dir=args.multirun_csv_dir,
        lr=args.lr,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        savecheck=args.savecheck,
        device=args.device,
        k_values=args.k_values,
        adapt_lr=args.adapt_lr,
        seeds=args.seeds,
        num_adapt_steps=args.num_adapt_steps,
        max_subprocesses=args.max_subprocesses,
        plot_metric=args.plot_metric,
        plasticity_metric=args.plasticity_metric,
        **kwargs
    )


if __name__ == '__main__':
    main()