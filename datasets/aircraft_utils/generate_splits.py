# Copyright 2026-present, Amogh Inamdar, Vici Milenia, Richard Zemel.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Generate seq-aircraft split yamls from the official FGVC-Aircraft splits.

Labels 0-99 follow a seeded shuffle (rng 42) of variants.txt, so contiguous
label blocks (= mammoth tasks) mix manufacturers instead of grouping all
737 variants into one task. Per class: official test kept as test, 10
trainval images reserved as the held-out k-shot support source, rest train.
"""

import pickle
from pathlib import Path

import numpy as np
import yaml

ROOT = Path('data/FGVC_AIRCRAFT/fgvc-aircraft-2013b/data')
OUT = Path(__file__).resolve().parent
HELDOUT_RESERVE = 10
SEED = 42


def read_split(name):
    """images_variant_{name}.txt lines are '<image_id> <variant name...>'."""
    examples = []
    for line in (ROOT / f'images_variant_{name}.txt').read_text().splitlines():
        img_id, variant = line.split(' ', 1)
        examples.append((f'data/FGVC_AIRCRAFT/fgvc-aircraft-2013b/data/images/{img_id}.jpg',
                         variant))
    return examples


def main():
    variants = (ROOT / 'variants.txt').read_text().splitlines()
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(variants))
    label = {variants[i]: int(np.where(order == i)[0][0]) for i in range(len(variants))}

    trainval, test = read_split('trainval'), read_split('test')
    by_class = {}
    for path, v in trainval:
        by_class.setdefault(label[v], []).append(path)

    splits = {'train': {'data': [], 'targets': []},
              'heldout': {'data': [], 'targets': []},
              'test': {'data': [], 'targets': []}}
    for lb in sorted(by_class):
        paths = sorted(by_class[lb])
        perm = np.random.default_rng(SEED + lb).permutation(len(paths))
        held = [paths[i] for i in perm[:HELDOUT_RESERVE]]
        train = [paths[i] for i in perm[HELDOUT_RESERVE:]]
        splits['heldout']['data'] += held
        splits['heldout']['targets'] += [lb] * len(held)
        splits['train']['data'] += train
        splits['train']['targets'] += [lb] * len(train)
    for path, v in test:
        splits['test']['data'].append(path)
        splits['test']['targets'].append(label[v])

    for name, d in splits.items():
        with open(OUT / f'aircraft_{name}.yaml', 'w') as f:
            yaml.dump(d, f)
        counts = np.bincount(d['targets'], minlength=100)
        print(f"{name}: {len(d['data'])} images, per-class min/max {counts.min()}/{counts.max()}")

    with open(OUT / 'label_to_class_name.pkl', 'wb') as f:
        pickle.dump({lb: v for v, lb in sorted(label.items(), key=lambda kv: kv[1])}, f)
    print("task 0 classes:", [v for v, lb in sorted(label.items(), key=lambda kv: kv[1])][:10])


if __name__ == '__main__':
    main()
