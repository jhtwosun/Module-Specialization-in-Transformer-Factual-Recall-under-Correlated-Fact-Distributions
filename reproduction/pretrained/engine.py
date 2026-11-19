"""GPU implementation. Imported only after the launcher validates Slurm allocation."""
import hashlib
import importlib.metadata
import json
import os
import random
import subprocess
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from .data import Synthetic
from .model import Instrumented


def write(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def append(path, data):
    with path.open('a') as f:
        f.write(json.dumps(data, allow_nan=False) + '\n')


def probability(logits, target):
    return logits.float().softmax(-1).gather(1, target[:, None]).squeeze(1)


def aggregate(rows, c):
    groups = defaultdict(list)
    for row in rows:
        groups[(row['source'], row['receiver'])].append(row)
    result = []
    for (source, receiver), group in sorted(groups.items()):
        module = 100 * sum(r['module_probability'] - r['corrupt_probability'] for r in group) / len(group)
        edge = 100 * sum(r['edge_probability'] - r['corrupt_probability'] for r in group) / len(group)
        result.append(dict(source=source, receiver=receiver, n_pairs=len(group), module_effect_pp=module,
                           edge_effect_pp=edge, eligible=module >= c.get('min_module_effect_pp', 1),
                           edge_module_ratio=edge / module if module >= c.get('min_module_effect_pp', 1) else None))
    valid = [r for r in result if r['eligible']]
    return {'layers': result, 'maximum': max(valid, key=lambda r: r['edge_module_ratio']) if valid else None}


@torch.no_grad()
def patch_pair(model, clean, corrupt, target, c, metadata, destination):
    lc, cc = model(clean.cuda())
    lb, cb = model(corrupt.cuda())
    target = target.cuda()
    pc, pb = probability(lc, target).item(), probability(lb, target).item()
    depth = len(model.layers)
    sources = c.get('source_layers', list(range(depth - 1)))
    receivers = c.get('target_layers', list(range(max(1, depth - 3), depth)))
    if any(i < 0 or i >= depth for i in sources + receivers):
        raise ValueError(f'Layer sweep exceeds model depth {depth}: {sources}, {receivers}')
    if not any(r > s for s in sources for r in receivers):
        raise ValueError('Layer sweep must include at least one later receiver')
    records = []
    for source in [-1] + list(range(depth)):
        state = cc['in'][0] if source == -1 else cc['out'][source]
        lm, _ = model(corrupt.cuda(), module=(source, state))
        pm = probability(lm, target).item()
        append(destination.with_name('module_patching.jsonl'), dict(metadata, source=source, target=target.item(),
               clean_probability=pc, corrupt_probability=pb, module_probability=pm,
               module_correct=lm.argmax(-1).item() == target.item()))
        if source == -1 or source not in sources:
            continue
        # Contribution-only delta: isolate source's own addition to the residual stream.
        # README.md specifies (clean_out[s]-clean_in[s])-(corrupt_out[s]-corrupt_in[s]).
        # The accumulated-residual formula (cc['out']-cb['out']) makes every edge ~equal
        # to its module (ratio ~100%) because adjacent-layer injection sends the same signal
        # as the module patch, destroying the metric's discrimination across strata.
        delta = (cc['out'][source] - cc['in'][source]) - (cb['out'][source] - cb['in'][source])
        for receiver in receivers:
            if receiver <= source:
                continue
            le, _ = model(corrupt.cuda(), edge=(receiver, delta))
            row = dict(metadata, source=source, receiver=receiver, target=target.item(), clean_probability=pc,
                       corrupt_probability=pb, module_probability=pm, edge_probability=probability(le, target).item(),
                       clean_correct=lc.argmax(-1).item() == target.item(),
                       module_correct=lm.argmax(-1).item() == target.item(),
                       edge_correct=le.argmax(-1).item() == target.item())
            append(destination, row)
            records.append(row)
    return records


@torch.no_grad()
def evaluate(model, data, c, seed, out):
    data.rng.manual_seed(seed + 99999)
    model.eval()
    records = []
    n = c.get('eval_examples', 2000)
    batch = c.get('eval_batch_size', 32)
    for start in range(0, n, batch):
        x, y, _ = data.sample(min(batch, n - start))
        logits, cache = model(x.cuda())
        preds = [logits.argmax(-1).cpu()] + [z.argmax(-1).cpu() for z in model.probe_logits(cache)]
        for j in range(len(y)):
            row = {'example': start + j, 'target': y[j].item(), 'input_ids': x[j].tolist(),
                   'prediction': preds[0][j].item(), 'probes': [p[j].item() for p in preds[1:]]}
            append(out / 'evaluation.jsonl', row)
            records.append(row)
    def metrics(get):
        by = defaultdict(list)
        for row in records:
            by[row['target']].append(int(get(row) == row['target']))
        return {'accuracy': sum(sum(v) for v in by.values()) / n,
                'macro_accuracy': sum(sum(v) / len(v) for v in by.values()) / len(by)}
    return {'full': metrics(lambda r: r['prediction']),
            'probes': [metrics(lambda r, i=i: r['probes'][i]) for i in range(len(model.probes))]}


def train_cell(c, seed, motif, out):
    # A carried-in checkpoint means this cell is continuing, so the directory is
    # expected to exist; without one it must not, which keeps two runs from writing
    # into the same place.
    out.mkdir(parents=True, exist_ok=(out / 'checkpoint.pt').is_file())
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    data = Synthetic(c, motif, seed)
    model = Instrumented(c, classes=data.na).cuda()
    write(out / 'resolved_model.json', {'commit': getattr(model.base.config, '_commit_hash', None),
                                      'model_config': model.base.config.to_dict()})
    torch.save({'first_hop': data.facts, 'second_hop': data.second}, out / 'facts.pt')
    opt = torch.optim.AdamW([
        {'params': [p for p in model.body.parameters() if p.requires_grad], 'lr': c.get('body_lr', 5e-5)},
        {'params': model.head.parameters(), 'lr': c.get('head_lr', 5e-3)},
        {'params': model.probes.parameters(), 'lr': c.get('probe_lr', 5e-3), 'weight_decay': 0.0},
    ], weight_decay=c.get('weight_decay', .01))
    steps = c.get('steps', 12000)
    # Three of eight seeds diverge early on the answer motif at the paper's head
    # learning rate. A warmup is the usual guard and the paper states none, so it is
    # off by default and recorded when used.
    warmup = c.get('warmup_steps', 0)
    base_lrs = [group['lr'] for group in opt.param_groups]
    start = 0
    saved = out / 'checkpoint.pt'
    if saved.is_file():
        # Load on CPU: load_state_dict copies into the parameters already on the GPU,
        # while the RNG states have to stay CPU ByteTensors to be restorable.
        state = torch.load(saved, map_location='cpu', weights_only=False)
        model.load_state_dict(state['model'])
        opt.load_state_dict(state['optimizer'])
        data.rng.set_state(state['sampler_rng'])
        torch.set_rng_state(state['cpu_rng'])
        torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda_rng']])
        start = state['step']
        print(json.dumps({'resumed_from_step': start}), flush=True)
    model.train()
    for step in range(start + 1, steps + 1):
        if warmup:
            scale = min(1.0, step / warmup)
            for group, base in zip(opt.param_groups, base_lrs):
                group['lr'] = base * scale
        x, y, _ = data.sample(c.get('batch_size', 32))
        logits, cache = model(x.cuda())
        y = y.cuda()
        main = F.cross_entropy(logits.float(), y)
        probe = sum(F.cross_entropy(p.float(), y) for p in model.probe_logits(cache))
        opt.zero_grad(set_to_none=True)
        (main + probe).backward()
        opt.step()
        if step % c.get('log_every', 100) == 0 or step == steps:
            row = {'step': step, 'loss': main.item(), 'probe_loss': probe.item()}
            append(out / 'training.jsonl', row)
            print(json.dumps(dict(seed=seed, motif=motif, **row)), flush=True)
        if step % c.get('checkpoint_every', 1000) == 0 or step == steps:
            # Atomic replacement leaves the preceding checkpoint intact until save completes.
            temp = out / 'checkpoint.tmp'
            torch.save({'step': step, 'model': model.state_dict(), 'optimizer': opt.state_dict(), 'config': c,
                        'seed': seed, 'motif': motif, 'sampler_rng': data.rng.get_state(),
                        'cpu_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all()}, temp)
            temp.replace(out / 'checkpoint.pt')
    metrics = evaluate(model, data, c, seed, out)
    rows = []
    for i, (clean, corrupt, target, meta) in enumerate(data.pairs(c.get('pairs', 200), seed)):
        meta.update(pair=i, clean_ids=clean[0].tolist(), corrupt_ids=corrupt[0].tolist())
        rows.extend(patch_pair(model, clean, corrupt, target, c, meta, out / 'patching.jsonl'))
    metrics['path_patching'] = aggregate(rows, c)
    write(out / 'metrics.json', metrics)
    del model, opt
    torch.cuda.empty_cache()
    return metrics


def load_counterfact(path):
    rows = json.loads(Path(path).read_text())
    if not isinstance(rows, list):
        raise ValueError('Expected original CounterFact JSON list')
    return rows


@torch.no_grad()
def natural(c, out):
    model = Instrumented(c).cuda().eval()
    model.requires_grad_(False)
    tok = AutoTokenizer.from_pretrained(c['model'], revision=c.get('revision', 'main'),
                                       local_files_only=c.get('local_files_only', False))
    write(out / 'resolved_model.json', {'commit': getattr(model.base.config, '_commit_hash', None),
                                      'model_config': model.base.config.to_dict()})
    source = Path(c['dataset_path'])
    write(out / 'dataset.json', {'path': str(source.resolve()), 'sha256': hashlib.sha256(source.read_bytes()).hexdigest()})
    facts = defaultdict(list)
    for row in load_counterfact(source):
        r = row['requested_rewrite']
        relation = r['relation_id']
        if relation not in c.get('relations', ['P103', 'P30', 'P19']):
            continue
        prompt = r['prompt'].format(r['subject']).rstrip()
        answer = ' ' + r['target_true']['str'].strip()
        ids = tok.encode(prompt, add_special_tokens=False)
        full = tok.encode(prompt + answer, add_special_tokens=False)
        if full[:len(ids)] != ids or len(full) == len(ids):
            append(out / 'selection.jsonl', {'case_id': row['case_id'], 'known': False, 'reason': 'token_boundary'})
            continue
        target = full[len(ids):]
        known = True
        for j, token in enumerate(target):
            logits, _ = model(torch.tensor([ids + target[:j]], device='cuda'))
            if logits.argmax(-1).item() != token:
                known = False
                break
        item = {'case_id': row['case_id'], 'relation': relation, 'subject': r['subject'], 'template': r['prompt'],
                'prompt': prompt, 'answer': answer, 'ids': ids, 'answer_ids': target, 'known': known}
        append(out / 'selection.jsonl', item)
        if known:
            facts[relation].append(item)
    if c.get('known_scan_only'):
        # Pairing needs facts the model answers greedily, and that count varies enormously by
        # relation and model: Pythia-1.4B knows 302 of 919 P103 facts and 0 of 924 P1412 facts.
        # This mode stops after the scan so relation availability can be measured before a
        # matrix is committed to a relation the model cannot support.
        counts = [{'relation': relation, 'known_facts': len(facts[relation])}
                  for relation in c.get('relations', ['P103', 'P30', 'P19'])]
        write(out / 'known_scan.json', {'model': c['model'], 'revision': c.get('revision'), 'relations': counts})
        return counts
    summaries = []
    for relation in c.get('relations', ['P103', 'P30', 'P19']):
        group = facts[relation]
        if len(group) < 2:
            raise RuntimeError(f'{relation}: only {len(group)} fully known facts, cannot pair')
        for seed in c.get('seeds', [0, 1, 2]):
            rng = random.Random(seed)
            records = []
            cell = out / f'{relation}-seed{seed}'
            cell.mkdir(exist_ok=False)
            # The caption's "200 pairs per fact" over every known fact costs ~93 forwards per
            # pair across the layer sweep, which is far beyond a practical allocation for the
            # larger models. max_facts subsamples the known facts and max_pairs_total caps the
            # emitted pairs; both default to no cap, and whichever is used is recorded here and
            # in the metrics so a reduced budget can never be read as the full caption budget.
            selected = group
            if c.get('max_facts') and len(group) > c['max_facts']:
                selected = sorted(random.Random(seed).sample(group, c['max_facts']), key=lambda f: f['case_id'])
            # Backfill preempts, and a 2,000-pair cell on an 8B model runs for hours.
            # Splitting the selected facts into chunks keeps each cell short enough to
            # finish; chunk k takes every chunks-th fact, so the chunks partition the
            # same selection and their pair sets are disjoint and mergeable.
            chunks, chunk = c.get('pair_chunks', 1), c.get('pair_chunk', 0)
            if chunks > 1:
                if not 0 <= chunk < chunks:
                    raise ValueError(f'pair_chunk {chunk} outside 0..{chunks-1}')
                selected = selected[chunk::chunks]
            budget = c.get('max_pairs_total')
            write(cell / 'selected_facts.json', {'known_facts': len(group), 'selected_facts': len(selected),
                                                'max_facts': c.get('max_facts'), 'max_pairs_total': budget,
                                                'pairs_per_fact': c.get('pairs_per_fact', 200),
                                                'pair_chunks': chunks, 'pair_chunk': chunk,
                                                'case_ids': [f['case_id'] for f in selected]})
            emitted = 0
            for fact in selected:
                if budget is not None and emitted >= budget:
                    break
                donors = [d for d in group if d['subject'] != fact['subject'] and d['answer_ids'][0] != fact['answer_ids'][0]]
                if not donors:
                    append(cell / 'excluded.jsonl', {'case_id': fact['case_id'], 'reason': 'no_different_answer_donor'})
                    continue
                for i in range(c.get('pairs_per_fact', 200)):
                    if budget is not None and emitted >= budget:
                        break
                    emitted += 1
                    donor = rng.choice(donors)
                    corrupt_ids = tok.encode(fact['template'].format(donor['subject']).rstrip(), add_special_tokens=False)
                    meta = {'case_id': fact['case_id'], 'donor_id': donor['case_id'], 'pair': i, 'seed': seed,
                            'relation': relation, 'clean_ids': fact['ids'], 'corrupt_ids': corrupt_ids}
                    records.extend(patch_pair(model, torch.tensor([fact['ids']]), torch.tensor([corrupt_ids]),
                                              torch.tensor([fact['answer_ids'][0]]), c, meta, cell / 'patching.jsonl'))
            summary = aggregate(records, c)
            summary.update(relation=relation, seed=seed, known_facts=len(group), selected_facts=len(selected),
                           pairs_emitted=emitted, max_facts=c.get('max_facts'), max_pairs_total=budget,
                           pairs_per_fact=c.get('pairs_per_fact', 200), pair_chunks=chunks, pair_chunk=chunk)
            write(cell / 'metrics.json', summary)
            summaries.append(summary)
    return summaries


def run(c, out):
    if not torch.cuda.is_available():
        raise RuntimeError('A Slurm GPU allocation with working CUDA is required')
    out.mkdir(parents=True, exist_ok=True)
    # The manifest executor writes the current job's status before invoking us.
    status_name = f"execution-{os.environ['SLURM_JOB_ID']}.json"
    # A preempted fine-tuning cell leaves a checkpoint behind. Continuing from it is
    # the only way a long cell finishes on a preemptible partition, so a directory
    # that holds checkpoints for the SAME configuration is resumed rather than
    # refused. Any other leftover still refuses, which is what keeps results from
    # different configurations out of one directory.
    leftovers = [p for p in out.iterdir() if p.name != status_name]
    saved = out / 'config.json'
    # The saved configuration carries the defaults this function applies, so the
    # comparison is over the keys the caller actually supplied.
    # Without a saved configuration the leftover is a checkpoint carried in to continue
    # training under a longer budget, which is allowed; with one, it has to agree on
    # every key the caller supplied except the budget, which is what extends.
    # steps extends the run; pairs and the layer grids only shape the evaluation that
    # follows it, so a continuation may change them without being a different training.
    evaluation_only = {'steps', 'pairs', 'source_layers', 'target_layers', 'min_module_effect_pp'}
    budget_free = {k: v for k, v in c.items() if k not in evaluation_only}
    same = not saved.is_file() or all(json.loads(saved.read_text()).get(k) == v
                                      for k, v in budget_free.items())
    directories = [cell for cell in leftovers if cell.is_dir()]
    resumable = bool(directories) and same and all((cell / 'checkpoint.pt').is_file()
                                                   for cell in directories)
    if leftovers and not resumable:
        raise FileExistsError(f'Refusing to mix results with existing output directory: {out}')
    defaults = dict(revision='main', seeds=[0, 1, 2], min_module_effect_pp=1,
                    local_files_only=False, random_init=False, legacy_repeat_relation=False)
    if c['experiment'] == 'natural_counterfact':
        defaults.update(dtype='bfloat16', relations=['P103', 'P30', 'P19'], pairs_per_fact=200,
                        max_facts=None, max_pairs_total=None, pair_chunks=1, pair_chunk=0)
    else:
        defaults.update(dtype='float32', warmup_steps=0, steps=12000, body_lr=5e-5, head_lr=5e-3,
                        probe_lr=5e-3, batch_size=32, eval_examples=2000, eval_batch_size=32,
                        pairs=200, checkpoint_every=1000, log_every=100, n_subjects=128,
                        n_relations=128, n_answers=256, sequence_length=16, n_noise=64,
                        collision=.9, subject_groups=2, answer_pool=4, weight_decay=.01,
                        motifs=['independent', 'subject', 'answer'] if c['experiment'] == 'synthetic_transfer' else ['independent'])
    c = dict(defaults, **c)
    write(out / 'config.json', c)
    rev = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(['git', 'status', '--porcelain'], capture_output=True, text=True).stdout
    write(out / 'provenance.json', {'git_revision': rev, 'git_status': dirty, 'slurm_job_id': os.environ['SLURM_JOB_ID'],
                                   'torch': torch.__version__, 'transformers': importlib.metadata.version('transformers'),
                                   'gpu': torch.cuda.get_device_name(), 'assumptions': 'reproduction/pretrained/README.md',
                                   'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                                     for p in Path(__file__).parent.glob('*.py')}})
    if c['experiment'] == 'natural_counterfact':
        results = natural(c, out)
    else:
        results = []
        for seed in c.get('seeds', [0, 1, 2]):
            for motif in c.get('motifs', ['independent', 'subject', 'answer'] if c['experiment'] == 'synthetic_transfer' else ['independent']):
                metrics = train_cell(c, seed, motif, out / f'{motif}-seed{seed}')
                results.append(dict(seed=seed, motif=motif, metrics=metrics))
    write(out / 'summary.json', results)
