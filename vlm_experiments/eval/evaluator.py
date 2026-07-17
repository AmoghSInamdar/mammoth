# Copyright 2026-present, Amogh Inamdar, Vici Milenia, Richard Zemel.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Task evaluation loop for the adaptation modes.

One call = one (adapter, eval_task) cell. Every query is scored ONCE over the
widest needed candidate range, then three menus are sliced out of the same
matrix:

- CIL: classes seen at the checkpoint, [0, class_hi(ckpt_task)) — the same
  menu for every eval task at that checkpoint. Eval tasks the checkpoint has
  not seen yet (task > ckpt_task) instead use their own classes, since the
  seen menu cannot contain the true label.
- TIL: the eval task's own classes [class_lo, class_hi) — the setting where
  the task identity is given (utils/evaluate.mask_classes).
- AS-LEARNED: [0, class_hi(eval_task)) — the menu the task had when it was
  the current task, kept fixed across checkpoints so the model faces the same
  choices over time. This is also the old k-shot pipeline's convention and
  the link to the ResNet CSVs. For future tasks it keeps the old behavior
  where correct classes sit outside the menu, on purpose.

Three measures per menu:

- raw (primary, the ``accuracy``/``loss`` columns): argmax of the
  length-normalized candidate log-probs — what the model would answer given
  the menu.
- PMI: raw minus the same context's prior scores, with the query image
  replaced by a plain gray image (PMI prior; Holtzman et al. 2021, Zhao et
  al. 2021) — what the image contributed, with menu bias removed. The
  raw - PMI accuracy gap is a derived column.
- free generation + parse: greedy decode, longest-prefix match to a menu
  name; unparseable answers count wrong.

The first 5 prompts + generated answers of each task go to the log.
"""

import logging
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader, Subset

from vlm_experiments.adaptation.scoring import parse_generation
from vlm_experiments.adaptation.support import EvalState
from vlm_experiments.eval.metrics import slice_metrics


def _task_loader(state: EvalState, task_id: int, max_queries: int,
                 seed: int) -> DataLoader:
    """Test loader for one task, optionally subsampled with a seed when the
    mode is too slow for the full test set (ICL). The subsample is a Subset
    over the underlying dataset, keeping the same batching.

    Args:
        state: EvalState with per-task test loaders.
        task_id: eval task.
        max_queries: query cap (0 = full test set).
        seed: subsample seed.

    Returns:
        DataLoader over the (sub)sampled queries.
    """
    loader = state.test_loaders[task_id]
    if max_queries <= 0 or len(loader.dataset) <= max_queries:
        return loader
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(loader.dataset), generator=g)[:max_queries].tolist()
    return DataLoader(Subset(loader.dataset, idx), batch_size=loader.batch_size,
                      shuffle=False, num_workers=0)


def _gen_accuracy(answers, labels, names, menu):
    """Accuracy of generation+parse against one menu, plus unparsed fraction.
    Unparseable answers (-1) can never match, so they count wrong and are
    tracked separately.

    Args:
        answers: generated strings, one per query.
        labels: [N] true class ids.
        names: label-order class names.
        menu: candidate class ids.

    Returns:
        (accuracy_percent, unparsed_fraction).
    """
    preds = [parse_generation(a, names, menu) for a in answers]
    total = len(preds)
    correct = sum(int(p == int(t)) for p, t in zip(preds, labels))
    unparsed = sum(int(p == -1) for p in preds)
    return (correct / total * 100.0 if total else 0.0,
            unparsed / total if total else 0.0)


def evaluate_adapter(adapter, state: EvalState, task_id: int,
                     ckpt_task: Optional[int] = None,
                     max_queries: int = 0, query_seed: int = 0,
                     log_first: int = 0, log_tag: str = '',
                     measure_pmi: bool = True, measure_gen: bool = True,
                     all_classes: bool = False) -> Dict:
    """Evaluate one prepared adapter on one task: CIL + TIL menus, raw / PMI /
    generation measures, per-class accuracies, first-N prompt+answer log.

    Args:
        adapter: prepared adaptation mode (scores/prior_scores/generate ready).
        state: EvalState.
        task_id: eval task.
        ckpt_task: task index of the checkpoint (defines the CIL seen menu);
            None falls back to CIL menu = [0, class_hi(task_id)) with a warning.
        max_queries: query cap (0 = full test set; use for ICL).
        query_seed: subsample seed.
        log_first: how many (prompt, answer) pairs to print (0 = none).
        log_tag: prefix for the log lines (mode/ckpt/k identifiers).
        measure_pmi: score the gray-image prior and PMI columns.
        measure_gen: run generation + parse columns.
        all_classes: the model under eval trained on no task, so it has seen no
            class more than any other and the CIL menu is the full class range
            [0, n_classes) rather than the own-task fallback.

    Returns:
        dict: accuracy/loss/per_class (CIL raw, primary) + accuracy_til/
        accuracy_as_learned (+ losses) + *_pmi + *_gen + gap columns
        (missing measures -> None).
    """
    lo_u, hi_u = state.offsets[task_id]
    if all_classes:
        hi_seen = state.n_classes
        unseen = False
    elif ckpt_task is None:
        logging.warning(f"{log_tag} no ckpt_task; CIL menu falls back to [0, {hi_u})")
        hi_seen = hi_u
        unseen = False
    else:
        hi_seen = state.offsets[ckpt_task][1]
        unseen = task_id > ckpt_task
    # Unseen task: the seen menu cannot contain the true label, so use the
    # task's own menu.
    cil_lo, cil_hi = (lo_u, hi_u) if unseen else (0, hi_seen)
    width = max(hi_seen, hi_u)

    adapter.set_width(width)
    loader = _task_loader(state, task_id, max_queries, query_seed)

    prior = adapter.prior_scores(width) if measure_pmi else None
    if prior is not None:
        prior = prior.cpu()

    all_scores, all_labels, all_answers = [], [], []
    logged = 0
    for data in loader:
        inputs = data[0].to(adapter.model.device)
        labels = data[1].to(adapter.model.device, dtype=torch.long)
        all_scores.append(adapter.scores(inputs).cpu())
        all_labels.append(labels.cpu())

        answers = adapter.generate(inputs) if measure_gen else None
        if answers is not None:
            all_answers.extend(answers)

        if logged < log_first:
            n = min(log_first - logged, inputs.shape[0])
            pairs = adapter.explain(inputs[:n], state)
            batch_scores = all_scores[-1][:n]
            preds = batch_scores[:, cil_lo:cil_hi].argmax(dim=1) + cil_lo
            for j, (prompt, answer) in enumerate(pairs):
                true = int(labels[j])
                gen_id = parse_generation(answer, state.class_names,
                                          list(range(cil_lo, cil_hi)))
                gen_name = state.class_names[gen_id] if gen_id >= 0 else '<unparsed>'
                logging.info(
                    f"[first5]{log_tag} task={task_id} #{logged + j + 1}\n"
                    f"  prompt: {prompt!r}\n"
                    f"  answer: {answer!r} | "
                    f"tf_pred={state.class_names[int(preds[j])]} "
                    f"gen_pred={gen_name} true={state.class_names[true]}")
            logged += n

    scores = torch.cat(all_scores, dim=0)
    labels = torch.cat(all_labels, dim=0)

    # Raw scores, all three menus, from the same matrix.
    acc, loss, per_class = slice_metrics(scores, labels, cil_lo, cil_hi)
    acc_til, loss_til, _ = slice_metrics(scores, labels, lo_u, hi_u)
    acc_al, loss_al, _ = slice_metrics(scores, labels, 0, hi_u)
    out = {'accuracy': acc, 'loss': loss, 'per_class': per_class,
           'accuracy_til': acc_til, 'loss_til': loss_til,
           'accuracy_as_learned': acc_al, 'loss_as_learned': loss_al,
           'accuracy_pmi': None, 'loss_pmi': None,
           'accuracy_til_pmi': None, 'loss_til_pmi': None,
           'accuracy_as_learned_pmi': None, 'loss_as_learned_pmi': None,
           'raw_pmi_gap': None, 'til_raw_pmi_gap': None,
           'as_learned_raw_pmi_gap': None,
           'accuracy_gen': None, 'accuracy_til_gen': None,
           'accuracy_as_learned_gen': None,
           'gen_unparsed_frac': None}

    # PMI: subtract the gray-image prior, slice the same menus.
    if prior is not None:
        pmi = scores - prior.unsqueeze(0)
        out['accuracy_pmi'], out['loss_pmi'], _ = slice_metrics(pmi, labels, cil_lo, cil_hi)
        out['accuracy_til_pmi'], out['loss_til_pmi'], _ = slice_metrics(pmi, labels, lo_u, hi_u)
        out['accuracy_as_learned_pmi'], out['loss_as_learned_pmi'], _ = \
            slice_metrics(pmi, labels, 0, hi_u)
        out['raw_pmi_gap'] = out['accuracy'] - out['accuracy_pmi']
        out['til_raw_pmi_gap'] = out['accuracy_til'] - out['accuracy_til_pmi']
        out['as_learned_raw_pmi_gap'] = (out['accuracy_as_learned']
                                         - out['accuracy_as_learned_pmi'])

    # Generation + parse, all three menus.
    if measure_gen and all_answers:
        out['accuracy_gen'], out['gen_unparsed_frac'] = _gen_accuracy(
            all_answers, labels, state.class_names, list(range(cil_lo, cil_hi)))
        out['accuracy_til_gen'], _ = _gen_accuracy(
            all_answers, labels, state.class_names, list(range(lo_u, hi_u)))
        out['accuracy_as_learned_gen'], _ = _gen_accuracy(
            all_answers, labels, state.class_names, list(range(0, hi_u)))

    def _f(v):
        return 'NA' if v is None else f"{v:.2f}"
    logging.info(f"[eval]{log_tag} task={task_id} n={labels.numel()} "
                 f"cil[raw={_f(out['accuracy'])} pmi={_f(out['accuracy_pmi'])} "
                 f"gen={_f(out['accuracy_gen'])}] "
                 f"til[raw={_f(out['accuracy_til'])} pmi={_f(out['accuracy_til_pmi'])} "
                 f"gen={_f(out['accuracy_til_gen'])}] "
                 f"al[raw={_f(out['accuracy_as_learned'])} "
                 f"pmi={_f(out['accuracy_as_learned_pmi'])} "
                 f"gen={_f(out['accuracy_as_learned_gen'])}] loss={out['loss']:.4f}")
    return out
