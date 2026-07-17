#!/usr/bin/env python3
# Copyright 2026-present, Amogh Inamdar, Vici Milenia, Richard Zemel.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Move existing checkpoints onto the sidecar scheme in vlm_experiments/manifest.py:
rename each .pt to its new name and write it a full sidecar. Prints every change
and writes nothing unless --execute is passed.

The adaptation learning rate and step count were never recorded for the
checkpoints already on disk (not in the .pt args, not in the logs, not in the
result CSVs), so they are filled in from --adapt_lr / --adapt_steps and marked
inferred in the sidecar. Everything else is read from each checkpoint's own args.

A rename map is written alongside so the checkpoint_id column of already-written
result CSVs can be remapped afterwards.

Usage:
    python vlm_experiments/migrate_checkpoints.py --checkpoint_dir <dir>
    python vlm_experiments/migrate_checkpoints.py --checkpoint_dir <dir> --execute
"""

import argparse
import glob
import io
import json
import os
import pickle
import re
import sys
import zipfile

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vlm_experiments.manifest import (adaptation_hyperparams, adapted_manifest,
                                      base_manifest, checkpoint_name,
                                      finetuned_manifest, sidecar_path,
                                      write_manifest)

ADAPTED_RE = re.compile(r'^(?P<donor>.+?)(?P<is_base>_base)?_grad_adapted'
                        r'_t(?P<task>\d+)_k(?P<k>\d+)_s(?P<seed>\d+)$')

FINETUNED_RE = re.compile(r'^(?P<prefix>.+)_(?P<task>\d+)$')


def parse_args() -> argparse.Namespace:
    """Read the command line for the migration.

    Args: none (reads sys.argv).

    Returns:
        the parsed namespace.
    """
    p = argparse.ArgumentParser(description='Move checkpoints onto the sidecar scheme')
    p.add_argument('--checkpoint_dir', required=True,
                   help='directory of .pt files to migrate (not recursive)')
    p.add_argument('--adapt_lr', type=float, default=1e-5,
                   help='adaptation learning rate to record for adapted '
                        'checkpoints; never stored on disk, so marked inferred')
    p.add_argument('--adapt_steps', type=int, default=5,
                   help='adaptation steps to record; marked inferred')
    p.add_argument('--adapt_batch_size', type=int, default=32,
                   help='adaptation batch cap to record; marked inferred')
    p.add_argument('--execute', action='store_true',
                   help='apply the renames and sidecar writes (default: print only)')
    p.add_argument('--rename_map', type=str, default=None,
                   help='where to write the old->new name map '
                        '(default: <checkpoint_dir>/rename_map.json)')
    return p.parse_args()


def _skip(*args, **kwargs):
    """Stand in for any torch class the pickle asks for. Reading args needs no
    tensors.

    Args: ignored.

    Returns:
        None, always.
    """
    return None


class _ArgsOnlyUnpickler(pickle.Unpickler):
    """Unpickler that reads a checkpoint's plain-python parts and skips every
    tensor."""

    def persistent_load(self, pid):
        """Skip the storage a tensor points at.

        Args:
            pid: the persistent id being loaded.

        Returns:
            None, always.
        """
        return None

    def find_class(self, module: str, name: str):
        """Resolve a class named by the pickle, standing in for torch ones.

        Args:
            module: module the pickle names.
            name: class name the pickle names.

        Returns:
            the resolved class, or a stand-in for torch classes.
        """
        if module.startswith('torch'):
            return _skip
        return super().find_class(module, name)


def read_args(path: str) -> dict:
    """Read a checkpoint's args without reading its weights. A checkpoint is a
    zip of one pickle plus the tensor data, so the pickle alone answers this.

    Args:
        path: path to the .pt file.

    Returns:
        the args dict.

    Raises:
        KeyError: the checkpoint has no args.
    """
    try:
        with zipfile.ZipFile(path) as z:
            name = next(n for n in z.namelist() if n.endswith('data.pkl'))
            obj = _ArgsOnlyUnpickler(io.BytesIO(z.read(name))).load()
    except (zipfile.BadZipFile, StopIteration):
        obj = torch.load(path, map_location='cpu', weights_only=False)
    a = obj['args']
    return a if isinstance(a, dict) else vars(a)


def build_manifest(path: str, args) -> dict:
    """Work out the sidecar for one existing checkpoint from its filename and
    its own args. The filename says whether it was adapted and from what; the
    args supply the dataset, method, model, and training settings.

    Args:
        path: path to the .pt file.
        args: the migration settings, for the inferred adaptation values.

    Returns:
        the sidecar dict.

    Raises:
        ValueError: the filename matches no known layout.
    """
    stem = os.path.splitext(os.path.basename(path))[0]
    a = read_args(path)

    m = ADAPTED_RE.match(stem)
    if m:
        hp = adaptation_hyperparams(args.adapt_lr, args.adapt_steps,
                                    args.adapt_batch_size, method=a['model'],
                                    vlm_prompt=a.get('vlm_prompt'),
                                    inferred=('lr', 'steps', 'batch_size'))
        if m.group('is_base'):
            base = base_manifest(a['vlm_model_id'], a['backbone'])
        else:
            donor_task = FINETUNED_RE.match(m.group('donor'))
            if not donor_task:
                raise ValueError(f'donor stem has no task: {m.group("donor")}')
            base = finetuned_manifest(a, int(donor_task.group('task')))
            base['checkpoint'] = checkpoint_name(base)
        return adapted_manifest(base, int(m.group('task')), int(m.group('k')),
                                int(m.group('seed')), hp, dataset=a['dataset'])

    m = FINETUNED_RE.match(stem)
    if m:
        return finetuned_manifest(a, int(m.group('task')))

    raise ValueError(f'unrecognized checkpoint name: {stem}')


def main():
    """Walk the checkpoint directory, work out each new name and sidecar, print
    the plan, and apply it only when asked.

    Args: none (CLI).

    Outputs: a printed plan; under --execute, renamed .pt files, new sidecars,
    and a rename map.
    """
    args = parse_args()
    paths = sorted(glob.glob(os.path.join(args.checkpoint_dir, '*.pt')))
    if not paths:
        raise SystemExit(f'no .pt files in {args.checkpoint_dir}')

    plan, failed, collisions = [], [], {}
    for path in paths:
        try:
            m = build_manifest(path, args)
        except (ValueError, KeyError) as e:
            failed.append((os.path.basename(path), str(e)))
            continue
        new_stem = checkpoint_name(m)
        new_path = os.path.join(os.path.dirname(path), new_stem + '.pt')
        collisions.setdefault(new_path, []).append(path)
        plan.append((path, new_path, m))

    clashing = {n: o for n, o in collisions.items() if len(o) > 1}

    print(f'{len(paths)} checkpoints in {args.checkpoint_dir}')
    print(f'  {len(plan)} planned, {len(failed)} unrecognized, '
          f'{len(clashing)} name collisions')
    print()
    for old, new, m in plan[:3]:
        print(f'  {os.path.basename(old)}')
        print(f'    -> {os.path.basename(new)}')
        print('    sidecar:')
        for line in json.dumps(m, indent=2).splitlines():
            print(f'      {line}')
        print()
    if len(plan) > 3:
        print(f'  ... and {len(plan) - 3} more')
        print()

    for name, err in failed[:5]:
        print(f'  UNRECOGNIZED {name}: {err}')
    for new, olds in list(clashing.items())[:5]:
        print(f'  COLLISION {os.path.basename(new)} <- {len(olds)} files:')
        for o in olds:
            print(f'      {os.path.basename(o)}')

    if clashing:
        raise SystemExit('\nrefusing to migrate: distinct checkpoints would '
                         'share a name and one would overwrite the other')
    if not args.execute:
        print('\ndry run, nothing written. pass --execute to apply.')
        return

    # Written before any file moves: a job that dies partway still leaves the
    # map needed to undo what it did.
    rename_map = {os.path.splitext(os.path.basename(old))[0]:
                  os.path.splitext(os.path.basename(new))[0]
                  for old, new, _ in plan}
    map_path = args.rename_map or os.path.join(args.checkpoint_dir, 'rename_map.json')
    with open(map_path, 'w') as f:
        json.dump(rename_map, f, indent=2)

    for old, new, m in plan:
        if old != new:
            os.replace(old, new)
        write_manifest(new, m)
        old_sidecar = sidecar_path(old)
        if os.path.exists(old_sidecar) and old_sidecar != sidecar_path(new):
            os.remove(old_sidecar)
    print(f'\nmigrated {len(plan)} checkpoints; rename map at {map_path}')


if __name__ == '__main__':
    main()
