"""Aggregate manifest-backed fresh measurements; stdlib unless plotting in Slurm.

Examples:
  python -m reproduction.report inspect --run reproduction/runs/pilot
  python -m reproduction.report aggregate --run RUN --output REPORT
  python -m reproduction.report plot --run RUN --output REPORT

Measurements are loaded from the supplied run manifests.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
from collections import defaultdict


TABLES = {
    'hierarchy': ('probe_macro.', 'delta2_macro_pp'),
    'regime': ('delta2_macro_pp',),
    'causal': ('mean_ablation_', 'patching.module_'),
    'pythia-natural': ('natural.layers.', 'natural.maximum.'),
    'path-patch': ('patching.strict_edge_', 'patching.legacy_edge_', 'patching.module_', 'edge_sum_gap_pp.'),
    'pythia-edge': ('full.', 'path_patching.layers.'),
    'causal-scrubbing': ('patching.clean.accuracy_to_clean', 'patching.scrub_', 'scrub_gap_pp'),
}


def read(path):
    return json.loads(Path(path).read_text())


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def flatten(value, prefix=''):
    """Numeric leaves only; preserve index, metric name and intervention convention."""
    result = {}
    if isinstance(value, dict):
        for key, item in value.items():
            if key in ('source', 'receiver', 'seed'):
                continue
            result.update(flatten(item, f'{prefix}.{key}' if prefix else key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            # Edge identifiers survive list reordering and omitted ineligible edges.
            key = f"{item['source']}->{item['receiver']}" if isinstance(item, dict) and 'source' in item and 'receiver' in item else str(index)
            leaves = flatten(item, f'{prefix}.{key}')
            if set(result) & set(leaves):
                raise ValueError(f'Duplicate metric key in {prefix}')
            result.update(leaves)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        if not math.isfinite(value):
            raise ValueError(f'Nonfinite measurement {prefix}')
        result[prefix] = float(value)
    return result


def derive(metrics, backend):
    result = flatten(metrics)
    if backend == 'synthetic':
        probes = metrics.get('probe_macro')
        if probes and len(probes) >= 4:
            result['delta2_macro_pp'] = 100 * (probes[3] - probes[2])
        for kind, baseline in [('macro', 'accuracy'), ('micro', 'micro_accuracy')]:
            for i, value in enumerate(metrics.get('mean_ablation_' + kind, [])):
                if baseline in metrics:
                    result[f'mean_ablation_{kind}_change_pp.{i}'] = 100 * (value - metrics[baseline])
        patches = metrics.get('patching', {})
        for convention in ('strict', 'legacy'):
            for source in (0, 1):
                module = patches.get(f'module_{source}', {}).get('effect_pp')
                edges = [v['effect_pp'] for k, v in patches.items()
                         if k.startswith(f'{convention}_edge_{source}_') and 'effect_pp' in v]
                if module is not None and edges:
                    result[f'edge_sum_gap_pp.{convention}.{source}'] = sum(edges) - module
    return result


def subcells(cfg, backend):
    if backend == 'twohop':
        return [(cfg.get('seed', 0), cfg['task'])]
    if backend == 'profile':
        return [(0, 'counterfact_profile')]
    if backend == 'synthetic':
        return [(cfg.get('seed', 0), cfg.get('motif', 'subject'))]
    axes = cfg.get('relations', ['P103', 'P30', 'P19']) if cfg['experiment'] == 'natural_counterfact' else cfg.get('motifs', ['independent', 'subject', 'answer'])
    return [(seed, axis) for seed in cfg.get('seeds', [0, 1, 2]) for axis in axes]


def group_config(cfg, backend, axis):
    c = {k: v for k, v in cfg.items() if k not in ('seed', 'seeds', 'motifs', 'relations')}
    c['relation' if backend == 'pretrained' and cfg['experiment'] == 'natural_counterfact' else 'motif'] = axis
    return c


def safe_child(root, relative):
    target = (root / relative).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError(f'Manifest path escapes run: {relative}')
    return target


def load_runs(runs):
    """Validate completion/provenance, returning per-seed records and explicit gaps."""
    records, coverage, configurations = [], [], {}
    seen = set()
    for path in runs:
        run = Path(path).resolve()
        manifest = read(run / 'manifest.json')
        source = fingerprint(manifest['source_sha256'])
        # A manifest alone cannot establish that its immutable snapshot survived.
        for rel, sha in manifest['source_sha256'].items():
            target = safe_child(run / 'source', rel)
            if not target.is_file() or hashlib.sha256(target.read_bytes()).hexdigest() != sha:
                raise ValueError(f'Source hash mismatch: {target}')
        for cell in manifest['cells']:
            config_path = safe_child(run, cell['config'])
            if hashlib.sha256(config_path.read_bytes()).hexdigest() != cell['config_sha256']:
                raise ValueError(f'Config hash mismatch: {config_path}')
            cfg = read(config_path)
            backend = cell['family']
            out = safe_child(run, cell['output'])
            status_files = sorted(out.glob('execution-*.json'), key=lambda p: p.stat().st_mtime_ns)
            error = None
            if not status_files:
                error = 'missing execution record'
            else:
                status = read(status_files[-1])
                if not status.get('finished_utc'):
                    error = 'execution unfinished'
                elif status.get('returncode') != 0:
                    error = f"execution failed: returncode={status.get('returncode')}"
            resolved = read(out / 'config.json') if (out / 'config.json').is_file() else cfg
            if any(resolved.get(k) != v for k, v in cfg.items()):
                error = 'resolved configuration differs from manifest'
            if not (out / 'config.json').is_file():
                error = error or 'missing resolved configuration'
            summary = None
            if backend == 'pretrained' and error is None:
                if (out / 'summary.json').is_file():
                    summary = read(out / 'summary.json')
                else:
                    error = 'missing pretrained summary'
            for seed, axis in subcells(cfg, backend):
                protocol = group_config(resolved, backend, axis)
                if backend == 'pretrained':
                    leaf = out if cfg['experiment'] == 'natural_counterfact' else out / f'{axis}-seed{seed}'
                    # `revision=main` is mutable; pin measured model/data identity.
                    for name, location in [('resolved_model', leaf / 'resolved_model.json'),
                                           ('resolved_dataset', out / 'dataset.json')]:
                        if location.is_file():
                            protocol[name] = read(location)
                group = fingerprint({'source': source, 'backend': backend, 'config': protocol})
                configurations[group] = {'source_fingerprint': source, 'backend': backend, 'config': protocol}
                identity = (group, seed)
                if identity in seen:
                    raise ValueError(f'Duplicate seed for identical source/config: {group[:12]}, seed {seed}; select one run explicitly')
                seen.add(identity)
                issue = error
                metrics = None
                if issue is None and backend in ('synthetic','twohop','profile'):
                    if (out / 'metrics.json').is_file():
                        metrics = read(out / 'metrics.json')
                    else:
                        issue = 'missing metrics.json'
                elif issue is None:
                    natural = cfg['experiment'] == 'natural_counterfact'
                    matching = [r for r in summary if r.get('seed') == seed and r.get('relation' if natural else 'motif') == axis]
                    if len(matching) != 1:
                        issue = 'missing or duplicate pretrained subcell'
                    elif natural:
                        metrics = {'natural': matching[0]}
                    else:
                        metrics = matching[0]['metrics']
                row = {'run': str(run), 'cell': cell['cell'], 'group': group, 'seed': seed,
                       'axis': axis, 'backend': backend, 'status': issue or 'complete'}
                coverage.append(row)
                if metrics is not None:
                    values = derive(metrics, backend)
                    if not values:
                        row['status'] = 'no numeric measurements'
                        continue
                    selections = {name: metrics[name].get('maximum') for name in ('natural','path_patching') if name in metrics}
                    records.append(dict(row, metrics=values, selections=selections))
    return records, coverage, configurations


def aggregate(records, coverage, expected_seeds=(0, 1, 2)):
    groups = defaultdict(list)
    for row in records:
        groups[row['group']].append(row)
    summaries, gaps = [], [dict(c) for c in coverage if c['status'] != 'complete']
    for group in sorted({c['group'] for c in coverage}):
        rows = groups[group]
        seeds = {r['seed'] for r in rows}
        if seeds != set(expected_seeds):
            gaps.append({'group': group, 'status': 'seed coverage mismatch',
                         'expected': list(expected_seeds), 'present': sorted(seeds)})
        names = sorted({k for r in rows for k in r['metrics']})
        for metric in names:
            measured = [(r['seed'], r['metrics'][metric]) for r in rows if metric in r['metrics']]
            values = [v for _, v in measured]
            complete = {s for s, _ in measured} == set(expected_seeds)
            if len(measured) != len(rows):
                gaps.append({'group': group, 'metric': metric, 'status': 'metric missing in some seeds'})
            summaries.append({'group': group, 'metric': metric, 'n': len(values),
                              'seeds': ','.join(str(s) for s, _ in sorted(measured)),
                              'mean': statistics.mean(values),
                              'std_population': statistics.pstdev(values),
                              'std_sample': statistics.stdev(values) if len(values) > 1 else None,
                              'complete': complete})
    return summaries, gaps


def write_csv(path, rows, fields):
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def select_table(name, summaries, configs):
    rows = []
    for r in summaries:
        c = configs[r['group']]['config']
        backend = configs[r['group']]['backend']
        if name in ('pythia-natural', 'pythia-edge'):
            if backend != 'pretrained':
                continue
            if name == 'pythia-edge' and c.get('experiment') != 'synthetic_transfer':
                continue
        elif backend != 'synthetic':
            continue
        if name in ('hierarchy', 'regime') and (c.get('layers') != 2 or c.get('probe_mode') != 'joint'):
            continue
        if name == 'hierarchy' and c.get('d_model') != 128:
            continue
        if any(r['metric'].startswith(p) for p in TABLES[name]):
            rows.append(r)
    return rows


def emit(output, records, coverage, configs, summaries, gaps):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'coverage.json').write_text(json.dumps({'cells': coverage, 'gaps': gaps}, indent=2) + '\n')
    (output / 'configurations.json').write_text(json.dumps(configs, indent=2, sort_keys=True) + '\n')
    (output / 'selected_edges.json').write_text(json.dumps([
        {k:v for k,v in r.items() if k != 'metrics'} for r in records if r.get('selections')
    ], indent=2) + '\n')
    raw = [dict(r, metric=k, value=v) for r in records for k, v in r['metrics'].items()]
    write_csv(output / 'per_seed.csv', raw, ['run', 'cell', 'group', 'seed', 'axis', 'backend', 'metric', 'value'])
    fields = ['group', 'metric', 'n', 'seeds', 'mean', 'std_population', 'std_sample', 'complete']
    write_csv(output / 'aggregate.csv', summaries, fields)
    lines = ['# Fresh-run measurements', '',
             f'{len(records)} completed seed records; {len(gaps)} coverage/protocol gaps.', '',
             'Values come from the supplied run manifests.', '',
             'Group IDs resolve to complete protocols in configurations.json; different source snapshots/configurations never pool. Units remain explicit in metric names: *_pp are percentage points; accuracy and ratios otherwise are fractions. Synthetic `accuracy` is macro; `micro_accuracy` is micro. Strict and legacy edges remain separate.', '',
             'Both population and sample standard deviations are exported. A single seed has no sample standard deviation. An incomplete mean is not a three-seed result.', '']
    lines += ['Natural/path maximum ratios are selected within each seed, then averaged; this convention may differ from maximizing an across-seed mean. Selected source/receiver IDs are retained in selected_edges.json, never numerically averaged.', '']
    for name in TABLES:
        subset = select_table(name, summaries, configs)
        write_csv(output / f'table_{name}.csv', subset, fields)
        lines += [f'## tab:{name}', '']
        if not subset:
            lines += ['MISSING: no matching measured fields in supplied runs.', '']
            continue
        lines += ['Fields collected from the supplied runs.', '',
                  '| Group | Metric | Mean | Sample std | Seeds | Complete |',
                  '|---|---|---:|---:|---|---|']
        for r in subset:
            sd = 'undefined' if r['std_sample'] is None else f"{r['std_sample']:.6g}"
            lines.append(f"| {r['group'][:12]} | {r['metric']} | {r['mean']:.6g} | {sd} | {r['seeds']} | {r['complete']} |")
        lines.append('')
    lines += ['## Outstanding coverage', '', 'Coverage is checked against the supplied manifests and requested seeds.', '']
    lines += [f'- {json.dumps(g, sort_keys=True)}' for g in gaps] or ['No gaps within supplied manifests.']
    (output / 'tables.md').write_text('\n'.join(lines) + '\n')


def plot(output, summaries, configs):
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Plotting requires Slurm allocation; use inspect/aggregate on login node')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output = Path(output)
    plotted_series = {}
    # These panels visualize measurements only. Every curve preserves full config
    # identity except its explicitly declared x coordinate; no cross-protocol line.
    def panel(ax, metric_prefix, backend, xkey=None, predicate=lambda c: True, title=''):
        series = defaultdict(list)
        for r in summaries:
            meta = configs[r['group']]
            c = meta['config']
            if meta['backend'] != backend or not predicate(c) or not r['metric'].startswith(metric_prefix):
                continue
            if not r['complete']:
                continue
            if xkey:
                if xkey not in c:
                    continue
                x = c[xkey]
                label = fingerprint({'source': meta['source_fingerprint'], 'config': {k:v for k,v in c.items() if k != xkey}, 'metric': r['metric']})[:8]
            else:
                suffix = r['metric'][len(metric_prefix):].split('.')[0]
                if not suffix.lstrip('-').isdigit():
                    continue
                x = int(suffix)
                label = r['group'][:8]
            plotted_series.setdefault(label, {'groups': [], 'metric_prefix':metric_prefix, 'x_coordinate':xkey or 'module index'})
            if r['group'] not in plotted_series[label]['groups']:
                plotted_series[label]['groups'].append(r['group'])
            series[label].append((x, r['mean'], r['std_sample'] or 0))
        if series:
            for text in list(ax.texts):
                if text.get_text() == 'No complete measured data':
                    text.remove()
        for label, points in series.items():
            points.sort()
            ax.errorbar([p[0] for p in points], [p[1] for p in points], yerr=[p[2] for p in points], marker='o', label=label)
        ax.set_title(title)
        if not series and not ax.lines:
            ax.text(.5, .5, 'No complete measured data', ha='center', transform=ax.transAxes)
        elif len(series) <= 12:
            ax.legend(fontsize=5)
        ax.set_xlabel(xkey or 'module index (zero based)')
        ax.set_ylabel(metric_prefix)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    panel(axes[0, 0], 'delta2_macro_pp', 'synthetic', 'collision', lambda c: c.get('motif') in ('subject','relation') and c.get('d_model') == 128, 'Collision hierarchy (pp)')
    panel(axes[0, 0], 'delta2_macro_pp', 'synthetic', 'answer_pool', lambda c: c.get('motif') == 'answer' and c.get('d_model') == 128, 'Collision hierarchy (pp); x=axis K')
    panel(axes[0, 1], 'delta2_macro_pp', 'synthetic', 'd_model', title='Capacity (pp)')
    panel(axes[1, 0], 'mean_ablation_macro_change_pp.', 'synthetic', title='Mean ablation macro change (pp)')
    # probability effect lives beneath module keys; numeric component extraction
    # is supported after choosing the precise suffix below.
    converted = [dict(r, metric=r['metric'].replace('patching.module_', 'module_effect.').replace('.effect_pp', ''))
                 for r in summaries if r['metric'].startswith('patching.module_') and r['metric'].endswith('.effect_pp')]
    saved = summaries
    summaries = converted
    panel(axes[1, 1], 'module_effect.', 'synthetic', title='Module patch effect (pp)')
    summaries = saved
    fig.tight_layout(); fig.savefig(output / 'fig_hero.pdf'); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5))
    panel(ax, 'probe_macro.', 'synthetic', predicate=lambda c: c.get('layers') == 4, title='Depth: residual probe macro accuracy')
    fig.tight_layout(); fig.savefig(output / 'fig_depth.pdf'); plt.close(fig)
    conventions = ('strict', 'legacy')
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, convention in zip(axes, conventions):
        panel(ax, f'chain_fraction.{convention}', 'synthetic', 'collision', lambda c:c.get('motif') == 'subject', f'Subject chain fraction ({convention})')
        panel(ax, f'chain_fraction.{convention}', 'synthetic', 'answer_pool', lambda c:c.get('motif') == 'answer', f'Chain fraction ({convention}); x=axis K')
    grid_rows = [r for r in summaries if r['metric'] == 'delta2_macro_pp' and r['complete'] and configs[r['group']]['config'].get('motif') == 'combined']
    grids = defaultdict(list)
    for r in grid_rows:
        meta = configs[r['group']]
        grids[fingerprint({'source':meta['source_fingerprint'], 'config':{k:v for k,v in meta['config'].items() if k not in ('collision','answer_pool')}})].append(r)
    if len(grids) == 1:
        rows = next(iter(grids.values()))
        ks = sorted({configs[r['group']]['config']['collision'] for r in rows}, reverse=True)
        ka = sorted({configs[r['group']]['config']['answer_pool'] for r in rows}, reverse=True)
        values = {(configs[r['group']]['config']['collision'],configs[r['group']]['config']['answer_pool']):r['mean'] for r in rows}
        mesh = [[values.get((s,a), math.nan) for a in ka] for s in ks]
        im = axes[2].imshow(mesh, aspect='auto')
        axes[2].set_xticks(range(len(ka)), ka); axes[2].set_yticks(range(len(ks)), ks)
        axes[2].set_xlabel('Ka'); axes[2].set_ylabel('Ks'); fig.colorbar(im, ax=axes[2], label='Δ2 macro pp')
    else:
        axes[2].text(.5,.5,'Missing or multiple protocols; select one grid',ha='center',transform=axes[2].transAxes)
    axes[2].set_title('Combined collision; missing cells masked')
    fig.tight_layout(); fig.savefig(output / 'fig_path_grid.pdf'); plt.close(fig)
    fig, axes = plt.subplots(1,2, figsize=(12,5))
    converted = [dict(r, metric=r['metric'].replace('.macro_accuracy','')) for r in summaries
                 if r['metric'].startswith('probes.') and r['metric'].endswith('.macro_accuracy')]
    summaries = converted
    panel(axes[0], 'probes.', 'pretrained', predicate=lambda c:c.get('experiment') == 'synthetic_transfer', title='Pythia probe macro trajectories')
    summaries = saved
    converted = [dict(r, metric=r['metric'].replace('.accuracy_to_clean','')) for r in summaries
                 if r['metric'].startswith('module_patching.') and r['metric'].endswith('.accuracy_to_clean')]
    summaries = converted
    panel(axes[1], 'module_patching.', 'pretrained', predicate=lambda c:c.get('experiment') == 'synthetic_transfer', title='Pythia module patch accuracy-to-clean')
    summaries = saved
    fig.tight_layout(); fig.savefig(output / 'fig_pythia.pdf'); plt.close(fig)
    (output / 'plot_series.json').write_text(json.dumps(plotted_series, indent=2) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('inspect', 'aggregate', 'plot'))
    p.add_argument('--run', type=Path, action='append', required=True)
    p.add_argument('--output', type=Path)
    p.add_argument('--seeds', default='0,1,2', help='Required seed set (comma-separated)')
    p.add_argument('--allow-incomplete', action='store_true')
    args = p.parse_args()
    if args.mode == 'plot' and not os.environ.get('SLURM_JOB_ID'):
        p.error('Plotting requires Slurm allocation')
    records, coverage, configs = load_runs(args.run)
    summaries, gaps = aggregate(records, coverage, tuple(int(s) for s in args.seeds.split(',')))
    if args.mode == 'inspect':
        print(json.dumps({'completed_seed_records':len(records), 'manifest_subcells':len(coverage), 'gaps':gaps}, indent=2))
        return
    if not args.output:
        p.error('--output required')
    emit(args.output, records, coverage, configs, summaries, gaps)
    if gaps and not args.allow_incomplete:
        p.exit(2, f'Coverage incomplete; diagnostics written to {args.output}; no figures generated\n')
    if args.mode == 'plot':
        plot(args.output, summaries, configs)
    print(f'Measured report: {args.output}; {len(gaps)} gaps')


if __name__ == '__main__':
    main()
