# Copyright 2026-present, Amogh Inamdar, Vici Milenia, Richard Zemel.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
CSV writers with the same format as the repo multirun pipeline
(utils/multirun_eval_results.py + eval_checkpoints.py) so the existing plot
scripts read the files unchanged:

- raw:        seed, checkpoint_id, eval_task_id, k_value, accuracy, loss,
              num_adapt_steps, adapt_lr, num_examples_used, digit_<c>_acc...
- aggregated: mean + sem over seeds per (checkpoint_id, eval_task_id, k_value)
              in out_dir/aggregated/, then SAUCE columns appended in place by
              utils/per_shot_plasticity.add_plasticity_scores_to_csv.
- transfer:   one small extra CSV with backward/forward transfer per
              (k, seed) — a new metric with no repo precedent, so it gets its
              own file instead of being forced into the plot format.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd

from utils.per_shot_plasticity import (add_plasticity_scores_to_csv,
                                       compute_per_shot_plasticity_all_variants_csv)

# Non-primary menus that get their own suffixed SAUCE/AUAC columns
# (SAUCE_til, SAUCE_as_learned, ...). The unsuffixed columns stay on the
# primary CIL 'accuracy'.
_SAUCE_MENU_METRICS = ('accuracy_til', 'accuracy_as_learned')
_PLASTICITY_COLS = ('AUAC', 'Clipped_AUAC', 'SAUCE', 'Clipped_SAUCE')


@dataclass
class ResultRow:
    """One (seed, checkpoint, task, k) evaluation — multirun CSV row.
    ``accuracy``/``loss`` = the primary measure (CIL raw); every other
    menu x measure combo (til / pmi / gen / gaps) rides in ``extra`` as its
    own columns."""
    seed: int
    checkpoint_id: str
    eval_task_id: int
    k_value: int
    accuracy: float
    loss: Optional[float] = None
    num_adapt_steps: int = 0
    adapt_lr: Optional[float] = None
    num_examples_used: int = 0
    extra: Dict[str, Optional[float]] = field(default_factory=dict)
    per_class_acc: Dict[int, float] = field(default_factory=dict)


def rows_to_dataframe(rows: List[ResultRow]) -> pd.DataFrame:
    """Flatten rows to the multirun CSV layout: base columns in repo order,
    then the extra measure columns (til / pmi / gen / gaps), then per-class
    digit_<c>_acc metadata columns.

    Args:
        rows: collected ResultRow list.

    Returns:
        DataFrame, one row per result.
    """
    records = []
    for r in rows:
        rec = {
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
        rec.update(r.extra)
        for c, acc in r.per_class_acc.items():
            rec[f'digit_{c}_acc'] = acc
        records.append(rec)
    return pd.DataFrame(records)


# Config columns: identity of the run, never averaged as metrics.
_KEY_COLS = ('seed', 'checkpoint_id', 'eval_task_id', 'k_value',
             'num_adapt_steps', 'adapt_lr', 'num_examples_used')


def save_results_csv(rows: List[ResultRow], out_dir: Union[str, Path],
                     group_name: str) -> Path:
    """Write raw + aggregated CSVs from ResultRow objects — thin wrapper over
    save_dataframe_csv. Kept as the runner's entry point.

    Args:
        rows: collected ResultRow list.
        out_dir: output directory (created).
        group_name: file stem suffix -> evaluation_results_<group_name>.csv.

    Returns:
        path of the raw CSV.
    """
    return save_dataframe_csv(rows_to_dataframe(rows), out_dir, group_name)


def save_dataframe_csv(df: pd.DataFrame, out_dir: Union[str, Path],
                       group_name: str) -> Path:
    """Write raw + aggregated CSVs (a matching pair) and append the plasticity
    columns, following eval_checkpoints.main: SAUCE goes on the aggregated file
    when several seeds exist, else on the raw one. This is also the merge path
    for fanned-out SLURM jobs (concat the per-part CSVs, then call this).

    Args:
        df: results DataFrame in the multirun layout.
        out_dir: output directory (created).
        group_name: file stem suffix -> evaluation_results_<group_name>.csv.

    Returns:
        path of the raw CSV.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f'evaluation_results_{group_name}.csv'
    df.to_csv(csv_path, index=False)
    logging.info(f"Saved {len(df)} rows to {csv_path}")

    multirun = df['seed'].nunique() > 1
    agg_path = None
    if multirun:
        # Every metric column (accuracy, loss, til/pmi/gen extras, digit_*)
        # gets mean + sem over seeds — same recipe as
        # MultirunEvaluationResults, extended to whatever measures the run
        # produced.
        agg_spec = {}
        for c in df.columns:
            if c in _KEY_COLS:
                continue
            agg_spec[c] = (c, 'mean')
            agg_spec[f'{c}_ste'] = (c, 'sem')
            if c == 'loss':   # n_seeds right after loss_ste, repo column order
                agg_spec['n_seeds'] = ('seed', 'count')
        agg = df.groupby(['checkpoint_id', 'eval_task_id', 'k_value']).agg(
            **agg_spec).reset_index()
        agg_path = out_dir / 'aggregated' / csv_path.name
        agg_path.parent.mkdir(parents=True, exist_ok=True)
        agg.to_csv(agg_path, index=False)
        logging.info(f"Saved aggregated results to {agg_path}")

    try:
        target = agg_path if multirun else csv_path
        add_plasticity_scores_to_csv(target, metric_for_sauce='accuracy')
        _add_menu_plasticity(target)
    except Exception as e:
        logging.warning(f"Failed to add plasticity scores: {e}")
    return csv_path


def _add_menu_plasticity(csv_path: Path) -> None:
    """SAUCE/AUAC for the non-primary menus, appended as suffixed columns
    (SAUCE_til, SAUCE_as_learned, ...) next to the repo's unsuffixed CIL
    ones. The repo scorer ties higher_is_better to the column name
    'accuracy', so each menu's values pass through a temp frame under that
    name — same behavior as the primary call.

    Args:
        csv_path: results CSV already carrying the unsuffixed columns.

    Outputs: none — rewrites the CSV in place with the extra columns.
    """
    df = pd.read_csv(csv_path)
    changed = False
    for metric in _SAUCE_MENU_METRICS:
        if metric not in df.columns or df[metric].isna().all():
            continue
        suffix = metric.replace('accuracy', '')          # '_til' / '_as_learned'
        for col in _PLASTICITY_COLS:
            df[f'{col}{suffix}'] = 0.0
        for _, idx in df.groupby(['checkpoint_id', 'eval_task_id']).groups.items():
            tmp = pd.DataFrame({'k_value': df.loc[idx, 'k_value'].values,
                                'accuracy': df.loc[idx, metric].values})
            scores = compute_per_shot_plasticity_all_variants_csv(
                tmp, metric_for_sauce='accuracy')
            for col, val in zip(_PLASTICITY_COLS, scores):
                df.loc[idx, f'{col}{suffix}'] = val
        changed = True
    if changed:
        df.to_csv(csv_path, index=False)
        logging.info(f"Menu plasticity columns added to {csv_path}")
