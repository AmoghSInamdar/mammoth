# Copyright 2026-present, Amogh Inamdar, Vici Milenia, Richard Zemel.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Evaluation loop + metrics (accuracy, SAUCE) for the VLM adaptation
modes."""

from vlm_experiments.eval.evaluator import evaluate_adapter
from vlm_experiments.eval.metrics import (accuracy_loss_per_class,
                                          sauce_from_curve, slice_metrics)
