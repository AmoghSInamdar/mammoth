# Copyright 2026-present, Amogh Inamdar, Vici Milenia, Richard Zemel.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Results storage and aggregation utilities for multirun few-shot checkpoint evaluation.
"""

import csv
import json
import logging
import os
from dataclasses import dataclass, asdict
from typing import Any, Callable, Dict, List, Optional, Union
from pathlib import Path

import pandas as pd

from utils.eval_results import EvaluationResult, EvaluationResults


@dataclass
class MultirunEvaluationResult(EvaluationResult):
    seed: int = 42

class MultirunEvaluationResults(EvaluationResults):
    results: List[MultirunEvaluationResult]

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to a flat DataFrame with one row per result."""
        all_metadata_keys = set()
        for r in self.results:
            if r.metadata:
                all_metadata_keys.update(r.metadata.keys())

        rows = []
        for r in self.results:
            row = {
                'seed': r.seed,
                'checkpoint_id': r.checkpoint_id,
                'eval_task_id': r.eval_task_id,
                'k_value': r.k_value,
                'accuracy': r.accuracy,
                'loss': r.loss,
                'num_adapt_steps': r.num_adapt_steps,
                'adapt_lr': r.adapt_lr,
                'num_examples_used': r.num_examples_used,
            }
            for key in all_metadata_keys:
                val = r.metadata.get(key, None) if r.metadata else None
                row[key] = pd.to_numeric(val, errors='coerce') if val is not None else None
            rows.append(row)

        return pd.DataFrame(rows)

    def to_aggregated_dataframe(self) -> pd.DataFrame:
        """Aggregate results across seeds by computing mean and std."""
        df = self.to_dataframe()

        all_metadata_keys = set()
        for r in self.results:
            if r.metadata:
                all_metadata_keys.update(r.metadata.keys())

        base_agg = {
            'accuracy': ('accuracy', 'mean'),
            'accuracy_std': ('accuracy', 'std'),
            'loss': ('loss', 'mean'),
            'loss_std': ('loss', 'std'),
            'n_seeds': ('seed', 'count'),
        }
        metadata_agg = {key: (key, 'mean') for key in all_metadata_keys}

        return df.groupby(['checkpoint_id', 'eval_task_id', 'k_value']).agg(
            **base_agg, **metadata_agg
        ).reset_index()

    def save_to_csv(self, filepath: Union[str, Path]) -> None:
        if not self.results:
            logging.warning("No results to save")
            return

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Save raw results
        self.to_dataframe().to_csv(filepath, index=False)
        logging.info(f"Saved {len(self.results)} results to {filepath}")

        # Save aggregated results
        agg_path = filepath.parent / 'aggregated' / filepath.name
        agg_path.parent.mkdir(parents=True, exist_ok=True)
        self.to_aggregated_dataframe().to_csv(agg_path, index=False)
        logging.info(f"Saved aggregated results to {agg_path}")

    @classmethod
    def load_from_csv(cls, filepath: Union[str, Path]) -> 'MultirunEvaluationResults':
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"CSV file not found: {filepath}")

        results = []

        with open(filepath, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)

            for row in reader:
                metadata = {}
                base_columns = {'seed', 'checkpoint_id', 'eval_task_id', 'k_value', 'accuracy', 'loss',
                               'num_adapt_steps', 'adapt_lr', 'num_examples_used'}

                for key, value in row.items():
                    if key not in base_columns and value.strip():
                        metadata[key] = value

                result = MultirunEvaluationResult(
                    checkpoint_id=row['checkpoint_id'],
                    eval_task_id=int(row['eval_task_id']),
                    k_value=int(row['k_value']),
                    accuracy=float(row['accuracy']),
                    loss=float(row['loss']) if row['loss'] and row['loss'].strip() else None,
                    num_adapt_steps=int(row['num_adapt_steps']) if row['num_adapt_steps'] and row['num_adapt_steps'].strip() else None,
                    adapt_lr=float(row['adapt_lr']) if row['adapt_lr'] and row['adapt_lr'].strip() else None,
                    num_examples_used=int(row['num_examples_used']) if row['num_examples_used'] and row['num_examples_used'].strip() else None,
                    metadata=metadata if metadata else None,
                    seed=int(row['seed']),
                )
                results.append(result)

        logging.info(f"Loaded {len(results)} results from {filepath}")
        return cls(results)
    
    def aggregate_by_metric(self, custom_metric_fn: Callable[[List[EvaluationResult]], Any],
                            group_by_checkpoint: bool = True) -> Dict[str, Any]:
        """
        Apply a custom aggregation metric to the results, grouped by seed first,
        then delegating to the parent's aggregate_by_metric.
        """
        seed_groups: Dict[int, List[MultirunEvaluationResult]] = {}
        for result in self.results:
            seed_groups.setdefault(result.seed, []).append(result)

        aggregated = {}
        for seed, seed_results in seed_groups.items():
            seed_eval_results = EvaluationResults(seed_results)
            aggregated[seed] = seed_eval_results.aggregate_by_metric(custom_metric_fn, group_by_checkpoint)

        return aggregated