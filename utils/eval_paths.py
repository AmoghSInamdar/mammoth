# Copyright 2026-present, Amogh Inamdar, Vici Milenia, Richard Zemel.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Output-path resolution for k-shot evaluation runs.

Runs that only adapt part of the network (``layer_min > 0``) write their results
and plots to separate directories, so that they do not overwrite the
full-adaptation (``layer_min = 0``) results.
"""

from pathlib import Path
from typing import Optional

RESULTS_DIR = Path('results/k_shot_evaluation')
PLOTS_DIR = Path('plots')


def get_output_dir(layer_min: Optional[int] = 0, base_dir: Path = RESULTS_DIR) -> Path:
    """Return the evaluation-results directory for a given layer_min.

    ``layer_min=0``  -> ``results/k_shot_evaluation``
    ``layer_min=16`` -> ``results/k_shot_evaluation_16``
    """
    if layer_min and layer_min > 0:
        return base_dir.with_name(f"{base_dir.name}_{layer_min}")
    return base_dir


def get_plots_dir(layer_min: Optional[int] = 0, base_dir: Path = PLOTS_DIR) -> Path:
    """Return the plots root for a given layer_min.

    The structure below the root is unchanged (``<dataset>/`` and
    ``paper_plots/<dataset>/``), so only the root is namespaced:

    ``layer_min=0``  -> ``plots``
    ``layer_min=16`` -> ``plots/layer_min_16``
    """
    if layer_min and layer_min > 0:
        return base_dir / f"layer_min_{layer_min}"
    return base_dir
