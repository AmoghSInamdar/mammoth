#!/usr/bin/env python3
# Copyright 2026-present, Amogh Inamdar, Vici Milenia, Richard Zemel.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Estimate the Anthropic API cost of the in-context SAUCE sweep with
``--icl_backend vlm`` (Claude multimodal classification) BEFORE running it.

The VLM backend is the only API-billed backend (clip/dinov2/vit are local). It
sends, per query: a short text scaffold + (k support images per candidate class)
+ 1 query image, and reads back a short integer label. We price that against the
current Claude model table and the vision token rule (tokens ~= w*h / 750).

Usage:
    python estimate_vlm_cost.py                       # full submission grid, defaults
    python estimate_vlm_cost.py --max_queries 10 --n_seeds 1
    python estimate_vlm_cost.py --datasets rot-mnist seq-cifar100 --model claude-haiku-4-5
"""

import argparse

# Per-1M-token pricing (input, output). Source: Claude model catalog.
PRICING = {
    'claude-opus-4-8':   (5.00, 25.00),
    'claude-sonnet-4-6': (3.00, 15.00),
    'claude-haiku-4-5':  (1.00, 5.00),
}

# (N_TASKS, N_CLASSES_PER_TASK, is_domain_il) per submission dataset. The VLM
# scores a query against the task's candidate label set: the fixed 10 digits for
# domain-il (rot/smooth-mnist), else the task's own classes.
DATASETS = {
    'seq-mnist':            (5,  2,  False),
    'rot-mnist':            (20, 10, True),
    'smooth-mnist':         (20, 10, True),
    'seq-cifar100':         (10, 10, False),
    'seq-cifar100-20task':  (20, 5,  False),
    'struct-cifar100':      (20, 5,  False),
    'seq-tinyimg':          (10, 20, False),
}

IMG_PX = 96  # images are PNG-resized to 96x96 before base64 (see in_context_eval._png_b64)
SCAFFOLD_TOKENS = 120
OUTPUT_TOKENS = 5


def img_tokens(px: int = IMG_PX) -> float:
    return (px * px) / 750.0


def call_cost(k: int, n_classes: int, in_per_m: float, out_per_m: float) -> float:
    n_example_imgs = k * n_classes
    in_tok = SCAFFOLD_TOKENS + (n_example_imgs + 1) * img_tokens()
    return (in_tok * in_per_m + OUTPUT_TOKENS * out_per_m) / 1e6


def main() -> None:
    p = argparse.ArgumentParser(description='Estimate VLM in-context SAUCE sweep cost')
    p.add_argument('--model', default='claude-opus-4-8', choices=list(PRICING))
    p.add_argument('--datasets', nargs='*', default=list(DATASETS),
                   help='Subset of datasets (default: full submission grid)')
    p.add_argument('--k_values', default='0,1,2,5,10')
    p.add_argument('--n_seeds', type=int, default=5)
    p.add_argument('--max_queries', type=int, default=50,
                   help='Matches eval_checkpoints.py --vlm_max_queries')
    args = p.parse_args()

    in_per_m, out_per_m = PRICING[args.model]
    k_values = [int(x) for x in args.k_values.split(',')]

    print(f"Model: {args.model}  (input ${in_per_m}/1M, output ${out_per_m}/1M)")
    print(f"k_values={k_values}  n_seeds={args.n_seeds}  max_queries={args.max_queries}\n")
    print(f"{'dataset':22} {'calls':>9} {'$cost':>9}")

    total_cost, total_calls = 0.0, 0
    for name in args.datasets:
        ntasks, ncls, is_dom = DATASETS[name]
        ds_cost, ds_calls = 0.0, 0
        for k in k_values:
            n_classes = 10 if is_dom else ncls
            calls = ntasks * args.n_seeds * args.max_queries
            ds_calls += calls
            ds_cost += calls * call_cost(k, n_classes, in_per_m, out_per_m)
        total_cost += ds_cost
        total_calls += ds_calls
        print(f"{name:22} {ds_calls:>9,} ${ds_cost:>7.2f}")

    print(f"{'TOTAL':22} {total_calls:>9,} ${total_cost:>7.2f}")
    print(f"\n~${total_cost:.0f} and {total_calls:,} API calls "
          f"(~1-2s/call sequential => plan for hours; lower --max_queries to cut cost/time).")


if __name__ == '__main__':
    main()
