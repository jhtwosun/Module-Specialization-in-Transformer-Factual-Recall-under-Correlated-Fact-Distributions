"""Slurm-only entry point. --validate is safe on a login node."""
import argparse
import json
import os
from pathlib import Path


def validate(c):
    if c.get('experiment') not in {'synthetic_transfer', 'natural_counterfact', 'two_hop'}:
        raise ValueError('experiment must be synthetic_transfer, natural_counterfact, or two_hop')
    if not c.get('model'):
        raise ValueError('model is required')
    if c['experiment'] == 'natural_counterfact' and not c.get('dataset_path'):
        raise ValueError('dataset_path to original CounterFact JSON is required')
    for k in ('steps', 'batch_size', 'eval_examples', 'pairs', 'pairs_per_fact'):
        if k in c and (not isinstance(c[k], int) or c[k] <= 0):
            raise ValueError(f'{k} must be a positive integer')
    if not c.get('seeds', [0, 1, 2]):
        raise ValueError('at least one seed is required')
    if not isinstance(c.get('legacy_repeat_relation', False), bool):
        raise ValueError('legacy_repeat_relation must be boolean')
    if c.get('random_init') and c['experiment'] == 'natural_counterfact':
        raise ValueError('Random-initialized models are only supported for synthetic smoke tests')
    for k in ('source_layers', 'target_layers'):
        if k in c and (not c[k] or any(not isinstance(v, int) or v < 0 for v in c[k])):
            raise ValueError(f'{k} must be a nonempty list of zero-based nonnegative block indices')
    if c.get('min_module_effect_pp', 1) <= 0:
        raise ValueError('min_module_effect_pp must be positive')
    if c.get('sequence_length', 16) < 4:
        raise ValueError('sequence_length must be at least 4')
    if 'subject' in c.get('motifs', []) and c.get('n_relations', 128) > c.get('n_answers', 256):
        raise ValueError('subject motif requires n_relations <= n_answers')
    return c


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--validate', action='store_true')
    a = p.parse_args()
    c = validate(json.loads(Path(a.config).read_text()))
    if a.validate:
        print(json.dumps({'valid': True, 'config': c}, indent=2))
        return
    if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('SLURM_JOB_GPUS', os.environ.get('CUDA_VISIBLE_DEVICES')):
        raise SystemExit('Refusing model imports/execution outside a Slurm GPU allocation')
    from .engine import run
    run(c, Path(a.output))


if __name__ == '__main__':
    main()
