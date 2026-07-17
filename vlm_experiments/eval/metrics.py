# Copyright 2026-present, Amogh Inamdar, Vici Milenia, Richard Zemel.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Transfer + plasticity metrics for the adaptation grid. These are thin wrappers
— the actual math lives in the repo (utils/metrics.py for backward/forward
transfer; utils/per_shot_plasticity.py for SAUCE) so the numbers match every
other pipeline in the repo exactly.
"""

from typing import Dict, List, Optional, Tuple

import torch

from utils.per_shot_plasticity import compute_plasticity_score


def slice_metrics(scores: torch.Tensor, labels: torch.Tensor, lo: int,
                  hi: int) -> Tuple[float, float, Dict[int, float]]:
    """Metrics over one candidate menu [lo, hi): percent accuracy of the
    argmax, candidate cross-entropy loss, per-class accuracy fractions. lo=0
    matches utils/evaluate.py's [:, :n_classes] slice; lo>0 is the
    task-identity menu.

    Args:
        scores: [N, n_classes] candidate scores.
        labels: [N] true class ids (must lie inside [lo, hi)).
        lo: menu start class id.
        hi: menu end class id (exclusive).

    Returns:
        (accuracy_percent, ce_loss, {class_id: acc_fraction}).
    """
    assert bool(((labels >= lo) & (labels < hi)).all()), \
        f"labels outside menu [{lo}, {hi})"
    sub = scores[:, lo:hi].float()
    preds = sub.argmax(dim=1) + lo
    total = labels.numel()
    acc = (preds == labels).sum().item() / total * 100.0 if total else 0.0
    loss = (torch.nn.functional.cross_entropy(sub, labels - lo).item()
            if total else 0.0)
    per_class = {}
    for c in labels.unique().tolist():
        m = labels == c
        per_class[int(c)] = (preds[m] == c).float().mean().item()
    return acc, loss, per_class


def accuracy_loss_per_class(scores: torch.Tensor, labels: torch.Tensor,
                            width: int) -> Tuple[float, float, Dict[int, float]]:
    """Older alias: metrics over the [0, width) menu. Kept so older callers
    keep working; slice_metrics is the real function.

    Args:
        scores: [N, n_classes] candidate scores.
        labels: [N] true class ids.
        width: candidate width.

    Returns:
        (accuracy_percent, ce_loss, {class_id: acc_fraction}).
    """
    return slice_metrics(scores, labels, 0, width)


def ckpt_task_index(checkpoint_id: str) -> Optional[int]:
    """Task index from a checkpoint stem like ..._adamw1e-5_3. None when the
    stem does not end in an int (the transfer grid is then skipped).

    Args:
        checkpoint_id: checkpoint file stem.

    Returns:
        int task index or None.
    """
    try:
        return int(checkpoint_id.split('_')[-1])
    except ValueError:
        return None


def sauce_from_curve(accuracies: List[float], k_values: List[int]) -> float:
    """SAUCE for one accuracy-vs-k curve, using the same call the CSV pipeline
    makes (do_clip off, scale_losses on). Convenience for logging; the CSV
    columns still come from add_plasticity_scores_to_csv.

    Args:
        accuracies: metric values ordered by increasing k.
        k_values: matching k values.

    Returns:
        SAUCE score (float).
    """
    return compute_plasticity_score(
        torch.tensor(accuracies, dtype=torch.float32),
        torch.tensor(k_values, dtype=torch.float32),
        do_clip=False, scale_losses=True, higher_is_better=True)
