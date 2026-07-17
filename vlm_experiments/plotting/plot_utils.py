#!/usr/bin/env python3
"""
Shared aggregation and CSV-loading helpers for the plotting scripts.

These are the pieces every plotting script in this directory reuses: locating
the right evaluation CSV, adding a numeric checkpoint column, and averaging an
accuracy column over the checkpoint x task grid per seed. Keeping them here
avoids copying the same logic into each figure script.
"""

import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

# results_for_paper lives at the repo root, which is not on sys.path when a
# script is run from this directory.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from results_for_paper import (  # noqa: E402  (needs the sys.path edit above)
    extract_checkpoint_num,
    load_evaluation_results,
)

ACCURACY_COLUMNS = {
    'cil': ('accuracy', 'accuracy-cil', 'Accuracy (CIL)'),
    'til': ('accuracy_til', 'accuracy-til', 'Accuracy (TIL)'),
    'as-learned': ('accuracy_as_learned', 'accuracy-aslearned', 'Accuracy (as learned)'),
    'pmi': ('accuracy_pmi', 'accuracy-pmi', 'Accuracy (CIL, PMI)'),
    'til-pmi': ('accuracy_til_pmi', 'accuracy-til-pmi', 'Accuracy (TIL, PMI)'),
    'as-learned-pmi': ('accuracy_as_learned_pmi', 'accuracy-aslearned-pmi',
                       'Accuracy (as learned, PMI)'),
    'gen': ('accuracy_gen', 'accuracy-gen', 'Accuracy (generative)'),
    'til-gen': ('accuracy_til_gen', 'accuracy-til-gen', 'Accuracy (TIL, generative)'),
    'as-learned-gen': ('accuracy_as_learned_gen', 'accuracy-aslearned-gen',
                       'Accuracy (as learned, generative)'),
}

BACKBONE_LABELS = {
    'smolvlm-256m': 'SmolVLM-256M',
    'smolvlm-500m': 'SmolVLM-500M',
    'resnet18': 'ResNet-18',
}

# These subdirs hold partial or baseline runs, not the full grid, so they must
# be skipped when locating a method's evaluation CSV.
VLM_EXCLUDE_DIRS = {'parts', 'parts_base', 'parts_icl_q200', 'parts_base_icl_q200',
                    'base', 'base_q200', 'baseline', 'prompt_imageof'}


def k_alpha(k_idx: int, n_k: int) -> float:
    """Compute the bar transparency for the k-th shot count. Shades run light
    to dark so more shots read as a stronger bar.

    Args:
        k_idx: Index of the k-value within the plotted k-values.
        n_k: Total number of plotted k-values.

    Returns:
        float: Alpha value in (0, 1].
    """
    if n_k <= 1:
        return 1.0
    return 0.4 + 0.6 * k_idx / (n_k - 1)


def load_with_checkpoint_num(csv_path: Path) -> pd.DataFrame:
    """Load an evaluation CSV and add a numeric checkpoint column. The
    checkpoint number is parsed from the trailing integer of checkpoint_id.

    Args:
        csv_path: Path to an evaluation_results CSV.

    Returns:
        pd.DataFrame: Loaded results with an extra 'checkpoint_num' column.
    """
    results = load_evaluation_results(csv_path)
    results = results.copy()
    results['checkpoint_num'] = results['checkpoint_id'].apply(extract_checkpoint_num)
    return results


def find_vlm_csv(vlm_dir: Path, method: str, mode: str = 'gradient') -> Optional[Path]:
    """Locate the evaluation CSV for one VLM method and adaptation mode. Walks
    the VLM results directory recursively, skipping partial/baseline subdirs
    and preferring the 256M backbone when more than one backbone matches.

    Args:
        vlm_dir: Root directory of VLM adaptation results for one dataset.
        method: VLM method name as it appears in filenames (e.g. 'vlm-sgd').
        mode: Adaptation mode suffix, e.g. 'gradient' (default: 'gradient').

    Returns:
        Optional[Path]: Path to the CSV, or None if no match was found.
    """
    pattern = f'evaluation_results_*_{method}_*_{mode}.csv'
    candidates = [p for p in vlm_dir.rglob(pattern)
                  if not (set(p.parts) & VLM_EXCLUDE_DIRS)]
    if not candidates:
        return None
    preferred = [p for p in candidates if 'smolvlm-256m' in p.stem]
    candidates = preferred or candidates
    candidates.sort(key=lambda p: (len(p.parts), str(p)))
    if len(candidates) > 1:
        print(f"Warning: multiple CSVs for {method}/{mode}, using {candidates[0]}")
    return candidates[0]


def vlm_backbone_from_csv(csv_path: Path, method: str) -> str:
    """Extract the backbone display label from a VLM evaluation CSV filename.
    The backbone is the filename token that follows the method name.

    Args:
        csv_path: Path to the VLM evaluation CSV.
        method: VLM method name used in the filename (e.g. 'vlm-sgd').

    Returns:
        str: Display label for the backbone (e.g. 'SmolVLM-256M').
    """
    parts = csv_path.stem.split(f'_{method}_')
    if len(parts) < 2:
        return 'VLM'
    token = parts[1].split('_')[0]
    return BACKBONE_LABELS.get(token, token)


def aggregate_metric(results: pd.DataFrame, k: int, column: str,
                     direction: str = 'all') -> Tuple[float, float]:
    """Aggregate one accuracy column over the checkpoint x task grid at a given
    k. Averages per seed first, then reports mean and standard error across
    seeds.

    Args:
        results: Evaluation results with checkpoint_num/eval_task_id/k_value.
        k: Number of adaptation shots to select.
        column: Accuracy column to aggregate. Falls back to 'accuracy' when the
            requested column is absent (e.g. ResNet CSVs lack the menu variants).
        direction: Cells to keep: 'all', 'backward' (eval task before the
            checkpoint) or 'forward' (eval task after the checkpoint)
            (default: 'all').

    Returns:
        Tuple[float, float]: Mean and standard error (NaN when < 2 seeds).
    """
    if column not in results.columns:
        column = 'accuracy'
    subset = results[results['k_value'] == k]
    if direction == 'backward':
        subset = subset[subset['eval_task_id'] < subset['checkpoint_num']]
    elif direction == 'forward':
        subset = subset[subset['eval_task_id'] > subset['checkpoint_num']]
    if subset.empty:
        return np.nan, np.nan
    seed_means = subset.groupby('seed')[column].mean()
    ste = seed_means.std(ddof=1) / np.sqrt(len(seed_means)) if len(seed_means) > 1 else np.nan
    return seed_means.mean(), ste


# Matches the '_grad_adapted_t{t}_k{K}_' token in a checkpoint id.
_GRAD_ADAPTED_RE = re.compile(r'grad_adapted_t(\d+)_k(\d+)')
# Matches the trained-checkpoint index that precedes that token.
_BASE_CKPT_RE = re.compile(r'_(\d+)_grad_adapted')


def find_grad_adapted_csvs(vlm_dir: Path, method: str, mode: str,
                           backbone: str = 'smolvlm-256m') -> List[Path]:
    """Collect the grad-adapted evaluation CSVs for one VLM method and mode.
    Returns the merged 'adapted' CSV (all adapted tasks x gradient shots) plus a
    sibling 'base' CSV for the k=0 point, falling back to the per-task 'parts'
    shards only when no merged file exists.

    Args:
        vlm_dir: Root directory of VLM adaptation results for one dataset.
        method: VLM method name as it appears in filenames (e.g. 'vlm-sgd').
        mode: Adaptation mode suffix ('prototypical' or 'protonet_cil').
        backbone: Backbone token to keep (e.g. 'smolvlm-256m', 'smolvlm-500m').
            When no file matches the token, all backbones are kept
            (default: 'smolvlm-256m').

    Returns:
        List[Path]: Matching CSVs for the selected backbone.
    """
    pattern = f'evaluation_results_*_{method}_*_{mode}.csv'
    all_match = [p for p in vlm_dir.rglob(pattern)
                 if backbone in p.stem] or list(vlm_dir.rglob(pattern))

    merged = [p for p in all_match
              if 'parts' not in p.parts and p.parent.name != 'base']
    base = [p for p in all_match if p.parent.name == 'base']
    if merged:
        return merged + base

    return [p for p in all_match
            if 'grad_adapted' in p.stem or p.parent.name == 'gradeval_base']


def find_unadapted_csv(unadapted_dir: Path, method: str, mode: str,
                       backbone: str = 'smolvlm-256m') -> Optional[Path]:
    """Locate the protonet CSV for one method evaluated without gradient
    adaptation, which supplies the k=0 point. Skips the per-checkpoint shards
    and any grad-adapted files.

    Args:
        unadapted_dir: Root directory holding the unadapted protonet results.
        method: VLM method name as it appears in filenames (e.g. 'vlm-sgd').
        mode: Adaptation mode suffix ('prototypical' or 'protonet_cil').
        backbone: Backbone token to keep (default: 'smolvlm-256m').

    Returns:
        Optional[Path]: Path to the CSV, or None when the method has no
            unadapted run for this mode.
    """
    pattern = f'evaluation_results_*_{method}_*_{mode}.csv'
    candidates = [p for p in unadapted_dir.rglob(pattern)
                  if backbone in p.stem
                  and 'grad_adapted' not in p.stem
                  and not (set(p.parts) & {'parts', 'baseline', 'base',
                                           'prompt_imageof'})
                  and not any(part.startswith('parts_') for part in p.parts)]
    if not candidates:
        return None
    candidates.sort(key=lambda p: (len(p.parts), str(p)))
    return candidates[0]


# Matches the backbone token in a readapt filename.
_BACKBONE_RE = re.compile(r'(smolvlm-\d+m)')


def find_merged_eval_csv(eval_dir: Path, mode: str) -> Optional[Path]:
    """Locate the merged evaluation CSV for one mode, which holds every method's
    checkpoints in a single file. Skips the per-cell shards and the base-only
    file.

    Args:
        eval_dir: Dataset directory of an eval run (holds the merged CSVs and a
            parts/ subdir).
        mode: Adaptation mode suffix ('prototypical' or 'protonet_cil').

    Returns:
        Optional[Path]: Path to the merged CSV, or None when absent.
    """
    candidates = [p for p in eval_dir.glob(f'evaluation_results_*_{mode}.csv')
                  if 'parts' not in p.parts and '_base_' not in p.stem]
    if not candidates:
        return None
    candidates.sort(key=lambda p: len(p.stem))
    return candidates[0]


def select_method_rows(results: pd.DataFrame, method: str) -> pd.DataFrame:
    """Keep the rows whose checkpoint belongs to one method. The merged eval CSV
    mixes every method plus the base model, tagged in the checkpoint id.

    Args:
        results: Rows loaded from a merged eval CSV.
        method: VLM method name as it appears in checkpoint ids (e.g. 'vlm-sgd').

    Returns:
        pd.DataFrame: Rows for that method only.
    """
    return results[results['checkpoint_id'].str.contains(f'{method}_', regex=False)]


def find_readapt_csv(readapt_dir: Path, mode: str) -> Optional[Path]:
    """Locate the merged base-model readapt CSV for one mode. These runs
    gradient-adapt the base model at each shot count and read it out with
    ProtoNet, so the merged file at the dataset root holds every shot count.

    Args:
        readapt_dir: Dataset directory of a readapt run (holds the merged CSV
            and a parts/ subdir).
        mode: Adaptation mode suffix ('prototypical' or 'protonet_cil').

    Returns:
        Optional[Path]: Path to the merged CSV, or None when absent.
    """
    candidates = [p for p in readapt_dir.rglob(f'evaluation_results_*_base_{mode}.csv')
                  if 'parts' not in p.parts]
    if not candidates:
        return None
    candidates.sort(key=lambda p: (len(p.parts), str(p)))
    return candidates[0]


def readapt_backbone(csv_path: Path) -> str:
    """Read the backbone display label out of a readapt CSV filename.

    Args:
        csv_path: Path to a readapt evaluation CSV.

    Returns:
        str: Display label for the backbone (e.g. 'SmolVLM-256M').
    """
    match = _BACKBONE_RE.search(csv_path.stem)
    if not match:
        return 'VLM'
    return BACKBONE_LABELS.get(match.group(1), match.group(1))


def backbone_from_checkpoint_id(ckpt_id: str) -> str:
    """Read the backbone display label out of a checkpoint id. Merged eval CSVs
    name the backbone in the checkpoint id rather than the filename.

    Args:
        ckpt_id: Checkpoint id from a merged eval CSV.

    Returns:
        str: Display label for the backbone (e.g. 'SmolVLM-256M').
    """
    match = _BACKBONE_RE.search(ckpt_id)
    if not match:
        return 'VLM'
    return BACKBONE_LABELS.get(match.group(1), match.group(1))


def load_readapt(csv_path: Path) -> pd.DataFrame:
    """Load a base-model readapt CSV, where the k_value column already holds the
    gradient-adaptation shot count. The base model has no task index, so the
    checkpoint column is left flat and only 'all' cells are meaningful.

    Args:
        csv_path: Merged readapt evaluation CSV.

    Returns:
        pd.DataFrame: Rows with a 'grad_k' column mirroring k_value.
    """
    df = load_evaluation_results(csv_path).copy()
    df['grad_k'] = df['k_value'].astype(int)
    df['adapted_task'] = np.nan
    df['checkpoint_num'] = -1
    return df


def load_grad_adapted(csv_paths: List[Path]) -> pd.DataFrame:
    """Load and stack grad-adapted evaluation CSVs, parsing the gradient shot
    count and adapted task from each checkpoint id. Base-model rows (no adapted
    token) are tagged as grad_k=0 with no adapted task.

    Args:
        csv_paths: Grad-adapted (and base) evaluation CSVs to concatenate.

    Returns:
        pd.DataFrame: Stacked rows with extra 'grad_k' and 'adapted_task'
            columns; empty frame when nothing loads.
    """
    frames = []
    for path in csv_paths:
        df = load_evaluation_results(path).copy()
        matches = df['checkpoint_id'].str.extract(_GRAD_ADAPTED_RE)
        df['adapted_task'] = pd.to_numeric(matches[0], errors='coerce')
        df['grad_k'] = pd.to_numeric(matches[1], errors='coerce').fillna(0).astype(int)
        base = pd.to_numeric(df['checkpoint_id'].str.extract(_BASE_CKPT_RE)[0],
                             errors='coerce')
        df['checkpoint_num'] = base.fillna(
            df['checkpoint_id'].apply(extract_checkpoint_num)).astype(int)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    stacked = pd.concat(frames, ignore_index=True)
    return stacked.drop_duplicates(
        subset=['checkpoint_id', 'eval_task_id', 'seed', 'k_value'])


def aggregate_grad_adapted(results: pd.DataFrame, grad_k: int, column: str,
                           direction: str = 'all',
                           proto_k: Optional[List[int]] = None) -> Tuple[float, float]:
    """Aggregate one accuracy column at a given gradient-adaptation shot count.
    Direction is measured against the trained checkpoint, so unadapted (k=0)
    rows use the same cells, and values are averaged per seed before reporting
    the mean and standard error across seeds.

    Args:
        results: Rows from load_grad_adapted with grad_k/checkpoint_num columns.
        grad_k: Gradient-adaptation shot count to select.
        column: Accuracy column to aggregate; falls back to 'accuracy' when the
            requested column is absent.
        direction: Cells to keep: 'all', 'backward' (eval task before the
            checkpoint) or 'forward' (eval task after it) (default: 'all').
        proto_k: ProtoNet support sizes to average over; None keeps every
            support size present (default: None).

    Returns:
        Tuple[float, float]: Mean and standard error (NaN when < 2 seeds).
    """
    if column not in results.columns:
        column = 'accuracy'
    subset = results[results['grad_k'] == grad_k]
    if proto_k is not None and 'k_value' in subset.columns:
        subset = subset[subset['k_value'].isin(proto_k)]
    if direction == 'backward':
        subset = subset[subset['eval_task_id'] < subset['checkpoint_num']]
    elif direction == 'forward':
        subset = subset[subset['eval_task_id'] > subset['checkpoint_num']]
    if subset.empty:
        return np.nan, np.nan
    seed_means = subset.groupby('seed')[column].mean()
    ste = seed_means.std(ddof=1) / np.sqrt(len(seed_means)) if len(seed_means) > 1 else np.nan
    return seed_means.mean(), ste
