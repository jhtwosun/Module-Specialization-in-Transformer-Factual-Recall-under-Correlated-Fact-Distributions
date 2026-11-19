"""CUDA/Slurm-only experiment entrypoint. --validate imports no torch."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess

DEFAULTS = dict(n_subjects=128, n_relations=128, answers=256, n_noise=64, length=16,
                d_model=128, heads=4, layers=2, temperature=1.0, motif='subject',
                collision=2, answer_pool=4, correlation=.9, block_sampling='with_replacement',
                legacy_repeat_relation=False, seed=0, steps=8000, batch_size=128, lr=.001,
                weight_decay=.01, probe_mode='joint', probe_steps=4000, probe_lr=.001, probe_weight_decay=.01,
                eval_samples=4096, mean_samples=2048, pairs=200, checkpoint_every=1000, value_init='default', value_init_scale=1.0,
                value_init_std=.02, identity_noise=.001, diagnostic_every=1000, warmup_steps=0,
                attention='bidirectional', positional='learned', final_norm=False,
                fact_sampling='fixed_table', probe_input='residual', combined_base='pool',
                probe_detach=True)


def resolve(config):
    extra = set(config)-set(DEFAULTS)
    if extra:
        raise ValueError(f'Unknown configuration keys: {sorted(extra)}')
    c = {**DEFAULTS, **config}
    for k in ('n_subjects', 'n_relations', 'answers', 'n_noise', 'length', 'd_model', 'heads',
              'layers', 'collision', 'answer_pool', 'steps', 'batch_size', 'probe_steps',
              'eval_samples', 'mean_samples', 'pairs', 'checkpoint_every'):
        if not isinstance(c[k], int) or c[k] < 1:
            raise ValueError(f'{k} must be a positive integer')
    if c['value_init'] not in ('default','small_normal','scaled_identity','orthogonal'):
        raise ValueError('Unknown value_init')
    if c['attention'] not in ('bidirectional', 'causal') or c['positional'] not in ('learned', 'none'):
        raise ValueError('Unknown attention or positional setting')
    if not isinstance(c['final_norm'], bool):
        raise ValueError('final_norm must be boolean')
    if c['fact_sampling'] not in ('fixed_table', 'per_example', 'pool_resampled'):
        raise ValueError('Unknown fact_sampling')
    if c['probe_input'] not in ('residual', 'module_output'):
        raise ValueError('Unknown probe_input')
    if c['combined_base'] not in ('pool', 'full'):
        raise ValueError('Unknown combined_base')
    if not isinstance(c['probe_detach'], bool):
        raise ValueError('probe_detach must be boolean')
    for k in ('value_init_scale','value_init_std','identity_noise'):
        if not isinstance(c[k], (int,float)) or not math.isfinite(c[k]) or c[k] < 0:
            raise ValueError(f'Invalid {k}')
    for k in ('diagnostic_every','warmup_steps'):
        if not isinstance(c[k], int) or c[k] < 0:
            raise ValueError(f'Invalid {k}')
    if c['length'] < 3 or c['n_relations'] < 2 or c['d_model'] % c['heads']:
        raise ValueError('Require length >=3, relations >=2, d_model divisible by heads')
    if c['motif'] not in ('independent', 'subject', 'relation', 'answer', 'combined'):
        raise ValueError('Unknown motif')
    bound = {'subject': c['n_subjects'], 'combined': c['n_subjects'], 'relation': c['n_relations'], 'answer': c['answers']}.get(c['motif'])
    if bound and c['collision'] > bound:
        raise ValueError('collision exceeds axis cardinality')
    if c['answer_pool'] > c['answers'] or not 0 <= c['correlation'] <= 1:
        raise ValueError('Invalid answer_pool/correlation')
    if c['block_sampling'] not in ('with_replacement', 'legacy_without_replacement'):
        raise ValueError('Invalid block_sampling')
    if c['block_sampling'] == 'legacy_without_replacement' and c['motif'] == 'relation' and c['collision'] > c['answers']:
        raise ValueError('Cannot sample relation block without replacement')
    if c['probe_mode'] not in ('joint', 'frozen', 'none') or c['lr'] <= 0 or c['probe_lr'] <= 0:
        raise ValueError('Invalid probe mode or learning rate')
    if not isinstance(c['seed'], int) or c['seed'] < 0 or not isinstance(c['legacy_repeat_relation'], bool):
        raise ValueError('seed must be nonnegative integer and legacy_repeat_relation boolean')
    for key in ('temperature', 'lr', 'probe_lr', 'weight_decay', 'probe_weight_decay', 'correlation'):
        if not isinstance(c[key], (int, float)) or not math.isfinite(c[key]):
            raise ValueError(f'{key} must be finite')
    if c['temperature'] <= 0 or c['weight_decay'] < 0 or c['probe_weight_decay'] < 0:
        raise ValueError('Temperature must be positive and decays nonnegative')
    return c


def guard():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Refusing compute outside Slurm allocation. Submit through reproduction/slurm.')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path)
    p.add_argument('--validate', action='store_true')
    p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    c = resolve(json.loads(a.config.read_text()))
    if a.validate:
        print(json.dumps(c, indent=2))
        return
    if not a.output:
        p.error('--output is required unless --validate')
    guard()
    import torch
    from torch.nn import functional as F
    from .data import facts, Sampler
    from .model import Transformer
    from .evaluate import evaluate
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required: CPU fallback is intentionally disabled')
    torch.set_num_threads(min(int(os.environ.get('SLURM_CPUS_PER_TASK', '4')), 8))
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(c['seed'])
    torch.cuda.manual_seed_all(c['seed'])
    out = a.output
    out.mkdir(parents=True, exist_ok=True)
    if (out/'config.json').exists() and resolve(json.loads((out/'config.json').read_text())) != c:
        raise ValueError('Output directory belongs to a different configuration')
    if (out/'checkpoint.pt').exists() and not a.resume:
        raise FileExistsError('Checkpoint exists: use --resume or a new output directory')
    dump(out/'config.json', c)
    git = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True)
    sources = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in Path(__file__).parent.glob('*.py')}
    dump(out/'provenance.json', dict(torch=torch.__version__, cuda=torch.version.cuda,
         gpu=torch.cuda.get_device_name(), python=platform.python_version(), git_commit=git.stdout.strip(),
         job_id=os.environ['SLURM_JOB_ID'], sources=sources))
    table, classes, pool, structured = facts(c, c['seed'])
    torch.save(dict(table=table, classes=classes, pool=pool, structured=structured), out/'facts.pt')
    model = Transformer(c).cuda()
    opt = torch.optim.AdamW([
        dict(params=[p for n,p in model.named_parameters() if not n.startswith('probes.')], lr=c['lr'], weight_decay=c['weight_decay']),
        dict(params=model.probes.parameters(), lr=c['probe_lr'], weight_decay=c['probe_weight_decay'])])
    from .diagnostics import value_diagnostics, probe_diagnostic
    if not a.resume:
        dump(out/'initial_value_spectrum.json', value_diagnostics(model))
    sampler = Sampler(table, c, c['seed']*7+11, structured, pool)
    probe_source = 'module_states' if c['probe_input'] == 'module_output' else 'residuals'
    start = 0
    frozen_done = False
    if a.resume:
        state = torch.load(out/'checkpoint.pt', map_location='cpu', weights_only=False)
        model.load_state_dict(state['model'])
        if len(state['optimizer']['param_groups']) == 1:
            # Backward compatibility for pre-rebuttal single-optimizer-group checkpoints.
            opt = torch.optim.AdamW(model.parameters(), lr=c['lr'], weight_decay=c['weight_decay'])
        opt.load_state_dict(state['optimizer'])
        start = state['step']
        frozen_done = state.get('frozen_done', False)
        sampler.rng.set_state(state['sampler_rng'])
        torch.set_rng_state(state['rng'])
        torch.cuda.set_rng_state_all(state['cuda_rng'])

    def checkpoint(step, frozen_done=False):
        temp = out/'checkpoint.tmp'
        torch.save(dict(model=model.state_dict(), optimizer=opt.state_dict(), config=c,
                        step=step, frozen_done=frozen_done, sampler_rng=sampler.rng.get_state(),
                        rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all()), temp)
        temp.replace(out/'checkpoint.pt')

    model.train()
    with (out/'train.jsonl').open('a') as log:
        for step in range(start, c['steps']):
            if c['warmup_steps']:
                factor = min(1., (step+1)/c['warmup_steps'])
                for group, base_lr in zip(opt.param_groups, (c['lr'],c['probe_lr'])):
                    group['lr'] = base_lr*factor
            x, y = sampler.sample(c['batch_size'])
            y = y.cuda()
            logits, cache = model(x.cuda())
            loss = F.cross_entropy(logits, y)
            if c['probe_mode'] == 'joint':
                # Section 3 blocks the probe's input gradient, which makes the joint and
                # CE-only backbones identical and leaves no room for the frozen-probe
                # appendix's claim that joint training inflates the non-subject gaps by
                # 1.5-3x. probe_detach=False lets the probe loss reach the backbone, which
                # is the only way that inflation can arise.
                loss = loss + sum(F.cross_entropy(probe(h if not c['probe_detach'] else h.detach()), y)
                                  for probe, h in zip(model.probes, cache[probe_source]))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if step % 100 == 0 or step+1 == c['steps']:
                row = dict(step=step+1, loss=loss.item())
                log.write(json.dumps(row)+'\n')
                log.flush()
                print(json.dumps(row), flush=True)
            if c['diagnostic_every'] and (step+1) % c['diagnostic_every'] == 0:
                diag = probe_diagnostic(model, table, c)
                diag.update(step=step+1, value_spectrum=value_diagnostics(model))
                with (out/'diagnostics.jsonl').open('a') as f:
                    f.write(json.dumps(diag, allow_nan=False)+'\n')
            if (step+1) % c['checkpoint_every'] == 0:
                checkpoint(step+1)
    checkpoint(c['steps'], frozen_done)
    if c['probe_mode'] == 'frozen' and not frozen_done:
        # Fresh probe optimizer only; detached activations keep backbone fixed.
        model.eval()
        probe_opt = torch.optim.AdamW(model.probes.parameters(), lr=c['probe_lr'], weight_decay=c['probe_weight_decay'])
        for _ in range(c['probe_steps']):
            x, y = sampler.sample(c['batch_size'])
            with torch.no_grad():
                _, cache = model(x.cuda())
            loss = sum(F.cross_entropy(probe(h.detach()), y.cuda()) for probe, h in zip(model.probes, cache[probe_source]))
            probe_opt.zero_grad(set_to_none=True)
            loss.backward()
            probe_opt.step()
        checkpoint(c['steps'], True)
    dump(out/'final_value_spectrum.json', value_diagnostics(model))
    evaluate(model, table, classes, c, out, structured, pool)


if __name__ == '__main__':
    main()
