# Copyright 2026-present, Amogh Inamdar, Vici Milenia, Richard Zemel.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Results storage and aggregation utilities for few-shot checkpoint evaluation.

This module provides data structures and methods for storing evaluation results
and computing custom aggregation metrics.
"""

import csv
import json
import logging
import os
from dataclasses import dataclass, asdict
from typing import Any, Callable, Dict, List, Optional, Union
from pathlib import Path


@dataclass
class EvaluationResult:
    """
    Represents a single evaluation result for a checkpoint-task-k combination.
    """
    checkpoint_id: str
    eval_task_id: int
    k_value: int
    accuracy: float
    loss: Optional[float] = None
    num_adapt_steps: Optional[int] = None
    adapt_lr: Optional[float] = None
    num_examples_used: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        # Handle metadata separately to avoid JSON serialization issues
        if self.metadata:
            result['metadata'] = self.metadata
        return result


class EvaluationResults:
    """
    Collection of evaluation results with methods for storage and aggregation.
    """

    def __init__(self, results: Optional[List[EvaluationResult]] = None):
        self.results = results or []

    def add_result(self, result: EvaluationResult) -> None:
        """Add a single evaluation result."""
        self.results.append(result)

    def add_results(self, results: List[EvaluationResult]) -> None:
        """Add multiple evaluation results."""
        self.results.extend(results)

    def save_to_csv(self, filepath: Union[str, Path]) -> None:
        """
        Save results to CSV format (flat structure, one row per result).

        Args:
            filepath: Path to save the CSV file
        """
        if not self.results:
            logging.warning("No results to save")
            return

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Get all possible metadata keys for CSV headers
        all_metadata_keys = set()
        for result in self.results:
            if result.metadata:
                all_metadata_keys.update(result.metadata.keys())

        # Define CSV columns
        base_columns = ['checkpoint_id', 'eval_task_id', 'k_value', 'accuracy', 'loss',
                       'num_adapt_steps', 'adapt_lr', 'num_examples_used']
        metadata_columns = sorted(list(all_metadata_keys))
        columns = base_columns + metadata_columns

        with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=columns)
            writer.writeheader()

            for result in self.results:
                row = {
                    'checkpoint_id': result.checkpoint_id,
                    'eval_task_id': result.eval_task_id,
                    'k_value': result.k_value,
                    'accuracy': result.accuracy,
                    'loss': result.loss,
                    'num_adapt_steps': result.num_adapt_steps,
                    'adapt_lr': result.adapt_lr,
                    'num_examples_used': result.num_examples_used,
                }

                # Add metadata fields
                if result.metadata:
                    for key in metadata_columns:
                        row[key] = result.metadata.get(key, '')

                writer.writerow(row)

        logging.info(f"Saved {len(self.results)} results to {filepath}")

    def save_to_json(self, filepath: Union[str, Path]) -> None:
        """
        Save results to JSON format (hierarchical structure).

        Structure:
        {
            "checkpoint_id": {
                "eval_task_id": {
                    "k_value": {
                        "accuracy": float,
                        "loss": float,
                        ...
                    }
                }
            }
        }

        Args:
            filepath: Path to save the JSON file
        """
        if not self.results:
            logging.warning("No results to save")
            return

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Build hierarchical structure
        hierarchical_results = {}

        for result in self.results:
            ckpt_key = result.checkpoint_id
            task_key = str(result.eval_task_id)
            k_key = str(result.k_value)

            if ckpt_key not in hierarchical_results:
                hierarchical_results[ckpt_key] = {}

            if task_key not in hierarchical_results[ckpt_key]:
                hierarchical_results[ckpt_key][task_key] = {}

            hierarchical_results[ckpt_key][task_key][k_key] = result.to_dict()

        with open(filepath, 'w', encoding='utf-8') as jsonfile:
            json.dump(hierarchical_results, jsonfile, indent=2, default=str)

        logging.info(f"Saved hierarchical results to {filepath}")

    @classmethod
    def load_from_csv(cls, filepath: Union[str, Path]) -> 'EvaluationResults':
        """
        Load results from CSV file.

        Args:
            filepath: Path to the CSV file

        Returns:
            EvaluationResults instance
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"CSV file not found: {filepath}")

        results = []

        with open(filepath, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)

            for row in reader:
                # Parse basic fields
                metadata = {}

                # Extract metadata fields (everything beyond base columns)
                base_columns = {'checkpoint_id', 'eval_task_id', 'k_value', 'accuracy', 'loss',
                               'num_adapt_steps', 'adapt_lr', 'num_examples_used'}

                for key, value in row.items():
                    if key not in base_columns and value.strip():
                        metadata[key] = value

                # Create result object
                result = EvaluationResult(
                    checkpoint_id=row['checkpoint_id'],
                    eval_task_id=int(row['eval_task_id']),
                    k_value=int(row['k_value']),
                    accuracy=float(row['accuracy']),
                    loss=float(row['loss']) if row['loss'] and row['loss'].strip() else None,
                    num_adapt_steps=int(row['num_adapt_steps']) if row['num_adapt_steps'] and row['num_adapt_steps'].strip() else None,
                    adapt_lr=float(row['adapt_lr']) if row['adapt_lr'] and row['adapt_lr'].strip() else None,
                    num_examples_used=int(row['num_examples_used']) if row['num_examples_used'] and row['num_examples_used'].strip() else None,
                    metadata=metadata if metadata else None
                )
                results.append(result)

        logging.info(f"Loaded {len(results)} results from {filepath}")
        return cls(results)

    def aggregate_by_metric(self, custom_metric_fn: Callable[[List[EvaluationResult]], Any],
                           group_by_checkpoint: bool = True) -> Dict[str, Any]:
        """
        Apply a custom aggregation metric to the results.

        Args:
            custom_metric_fn: Function that takes a list of EvaluationResult and returns aggregated metric
            group_by_checkpoint: Whether to group results by checkpoint before aggregation

        Returns:
            Dictionary mapping checkpoint IDs (or 'all') to aggregated metrics
        """
        if group_by_checkpoint:
            # Group results by checkpoint
            checkpoint_groups = {}
            for result in self.results:
                if result.checkpoint_id not in checkpoint_groups:
                    checkpoint_groups[result.checkpoint_id] = []
                checkpoint_groups[result.checkpoint_id].append(result)

            # Apply metric to each checkpoint group
            aggregated = {}
            for ckpt_id, ckpt_results in checkpoint_groups.items():
                try:
                    aggregated[ckpt_id] = custom_metric_fn(ckpt_results)
                except Exception as e:
                    logging.error(f"Error computing metric for checkpoint {ckpt_id}: {e}")
                    aggregated[ckpt_id] = None

            return aggregated
        else:
            # Apply metric to all results
            try:
                return {'all': custom_metric_fn(self.results)}
            except Exception as e:
                logging.error(f"Error computing metric for all results: {e}")
                return {'all': None}

    def get_results_for_checkpoint(self, checkpoint_id: str) -> List[EvaluationResult]:
        """Get all results for a specific checkpoint."""
        return [r for r in self.results if r.checkpoint_id == checkpoint_id]

    def get_results_for_task(self, eval_task_id: int) -> List[EvaluationResult]:
        """Get all results for a specific evaluation task."""
        return [r for r in self.results if r.eval_task_id == eval_task_id]

    def get_results_for_k(self, k_value: int) -> List[EvaluationResult]:
        """Get all results for a specific k value."""
        return [r for r in self.results if r.k_value == k_value]

    def __len__(self) -> int:
        return len(self.results)

    def __iter__(self):
        return iter(self.results)