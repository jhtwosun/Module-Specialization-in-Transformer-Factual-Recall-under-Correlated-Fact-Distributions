"""Save raw predictions, pair identities, donors, and aggregate measurements."""
import json
import math
import statistics
import torch
from .data import Sampler


def macro(pred, y):
    return sum((pred[y == k] == k).float().mean().item() for k in y.unique())/len(y.unique())


@torch.no_grad()
def evaluate(model, table, classes, c, out, structured=None, pool=None):
    model.eval()
    device = next(model.parameters()).device
    sample = Sampler(table, c, c['seed']*101+19, structured, pool)
    nmod = len(model.probes)
    # Independent calibration examples; primary ablation is query-only per paper.
    # Retain the recovered all-position intervention separately.
    means = torch.zeros(nmod, c['d_model'], device=device)
    for begin in range(0, c['mean_samples'], c['batch_size']):
        x, _ = sample.sample(min(c['batch_size'], c['mean_samples']-begin))
        _, cache = model(x.to(device))
        means += torch.stack([a[:, -1].sum(0) for a in cache['outputs']])
    means /= c['mean_samples']
    raw = []
    for begin in range(0, c['eval_samples'], c['batch_size']):
        x, y = sample.sample(min(c['batch_size'], c['eval_samples']-begin))
        logits, cache = model(x.to(device))
        source = 'module_states' if c.get('probe_input') == 'module_output' else 'residuals'
        probes = torch.stack([p(h).argmax(-1).cpu() for p, h in zip(model.probes, cache[source])], 1)
        ablated = torch.stack([model(x.to(device), mean_replacements={i: means[i]}, mean_scope='query')[0].argmax(-1).cpu() for i in range(nmod)], 1)
        legacy = torch.stack([model(x.to(device), mean_replacements={i: means[i]}, mean_scope='all')[0].argmax(-1).cpu() for i in range(nmod)], 1)
        raw.append(dict(x=x, y=y, pred=logits.argmax(-1).cpu(), probes=probes, ablated=ablated, legacy_ablated=legacy))
    raw = {k: torch.cat([batch[k] for batch in raw]) for k in raw[0]}
    torch.save(dict(**raw, module_means=means.cpu()), out/'evaluation.pt')
    summary = dict(accuracy=macro(raw['pred'], raw['y']), micro_accuracy=(raw['pred'] == raw['y']).float().mean().item(),
                   probe_macro=[macro(raw['probes'][:, i], raw['y']) for i in range(nmod)] if c['probe_mode'] != 'none' else None,
                   mean_ablation_macro=[macro(raw['ablated'][:, i], raw['y']) for i in range(nmod)])
    # The causal table's ablation column turned out to be micro while the hierarchy
    # caption's gap is macro, so both conventions are recorded for the probes too.
    if c['probe_mode'] != 'none':
        summary['probe_micro'] = [(raw['probes'][:, i] == raw['y']).float().mean().item() for i in range(nmod)]
        summary['probe_layer_gap_micro_pp'] = [100*(summary['probe_micro'][i+1]-summary['probe_micro'][i])
                                               for i in range(0, nmod, 2)]
    summary['mean_ablation_scope'] = 'query'
    summary['mean_ablation_micro'] = [(raw['ablated'][:, i] == raw['y']).float().mean().item() for i in range(nmod)]
    summary['legacy_all_position_ablation_macro'] = [macro(raw['legacy_ablated'][:, i], raw['y']) for i in range(nmod)]
    summary['class_support'] = {str(k.item()): int((raw['y'] == k).sum()) for k in raw['y'].unique()}
    summary['mean_ablation_drop_pp'] = [100*(summary['accuracy']-value) for value in summary['mean_ablation_macro']]
    summary['probe_layer_gap_pp'] = ([100*(summary['probe_macro'][i+1]-summary['probe_macro'][i]) for i in range(0, nmod, 2)]
                                      if summary['probe_macro'] is not None else None)
    # Evaluation-level standard error of the same gaps, by resampling evaluation
    # examples. A published deviation can be this or a seed deviation, and the two
    # differ by an order of magnitude, so neither is assumed.
    if summary['probe_macro'] is not None:
        boot = torch.Generator().manual_seed(c['seed']+777)
        draws = {i: [] for i in range(0, nmod, 2)}
        for _ in range(200):
            idx = torch.randint(len(raw['y']), (len(raw['y']),), generator=boot)
            y_b = raw['y'][idx]
            for i in draws:
                draws[i].append(100*(macro(raw['probes'][idx, i+1], y_b)-macro(raw['probes'][idx, i], y_b)))
        summary['probe_layer_gap_bootstrap_se_pp'] = [statistics.stdev(draws[i]) for i in sorted(draws)]
        summary['probe_bootstrap_draws'] = 200
    pair = sample.matched(c['pairs'])
    clean, corrupt, y, yc, s, r, other, pos = pair
    # Batched pair processing limits memory even at larger d and depth.
    pair_rows, donors_saved = [], []
    for begin in range(0, c['pairs'], c['batch_size']):
        end = min(begin+c['batch_size'], c['pairs'])
        cx, bx, yy = clean[begin:end].to(device), corrupt[begin:end].to(device), y[begin:end].to(device)
        cl, cc = model(cx)
        bl, bc = model(bx)
        results = {'clean': cl, 'corrupt': bl}
        for i in range(nmod):
            results[f'module_{i}'] = model(bx, replacements={i: cc['outputs'][i][:, -1]})[0]
        for source in range(min(2, nmod)):
            for receiver in list(range(source+1, nmod))+['direct']:
                for convention in ('strict', 'legacy'):
                    results[f'{convention}_edge_{source}_{receiver}'] = model.edge(bc, cc, source, receiver, convention)
        if c['layers'] == 2:
            for name, logits in model.joint_chain_readings(bc, cc).items():
                results['joint_'+name] = logits
            results['joint_chain'] = results['joint_replace_last_only']
            results['joint_simultaneous'] = model(bx, replacements={i: cc['outputs'][i][:, -1]
                                                                   for i in (0, 1, 3)})[0]
            for label, hypothesis in [('hypothesis', True), ('null', False)]:
                donors = sample.scrub_donors(s[begin:end], r[begin:end], y[begin:end], classes, hypothesis)
                replacements = {}
                for index, (dx, ids) in zip((0, 1, 3), donors):
                    _, dc = model(dx.to(device))
                    replacements[index] = dc['outputs'][index][:, -1]
                    donors_saved.append(dict(begin=begin, condition=label, module=index, ids=ids, x=dx))
                results['scrub_'+label] = model(cx, replacements=replacements)[0]
        wrong = yc[begin:end].to(device)
        vals = {key: dict(margin=(logit.gather(1, yy[:, None])-logit.gather(1, wrong[:, None])).squeeze(1).cpu().tolist(), pred=logit.argmax(-1).cpu().tolist(), p=logit.softmax(-1).gather(1, yy[:, None]).squeeze(1).cpu().tolist())
                for key, logit in results.items()}
        for j in range(end-begin):
            i = begin+j
            pair_rows.append(dict(pair=i, subject=s[i].item(), relation=r[i].item(), corrupt_subject=other[i].item(),
                                  answer=y[i].item(), corrupt_answer=yc[i].item(), subject_position=pos[i].item(),
                                  results={k: dict(pred=v['pred'][j], p_clean=v['p'][j], logit_margin=v['margin'][j]) for k, v in vals.items()}))
    torch.save(dict(clean=clean, corrupt=corrupt, target=y, corrupt_target=yc, donors=donors_saved), out/'pairs.pt')
    with (out/'pairs.jsonl').open('w') as f:
        for row in pair_rows:
            f.write(json.dumps(row)+'\n')
    metrics = {}
    for key in pair_rows[0]['results']:
        metrics[key] = dict(logit_margin=sum(row['results'][key]['logit_margin'] for row in pair_rows)/len(pair_rows), accuracy_to_clean=sum(row['results'][key]['pred'] == row['answer'] for row in pair_rows)/len(pair_rows),
                            probability_clean=sum(row['results'][key]['p_clean'] for row in pair_rows)/len(pair_rows))
    base = metrics['corrupt']['probability_clean']
    for key, value in metrics.items():
        value['logit_margin_effect'] = value['logit_margin']-metrics['corrupt']['logit_margin']
        value['effect_pp'] = 100*(value['probability_clean']-base)
        # Standard error of the effect over the matched pairs. The rebuttal's own
        # deviations are this quantity, not a seed sample deviation: R05 reports
        # 57.3 +/- 2.8 pp, and the matching condition measures a pair standard error
        # of 2.803 pp against a seed deviation of 12.3 pp. Reports must say which of
        # the two they quote, so both are recorded.
        paired = [100*(row['results'][key]['p_clean']-row['results']['corrupt']['p_clean']) for row in pair_rows]
        value['effect_pair_se_pp'] = statistics.stdev(paired)/math.sqrt(len(paired)) if len(paired) > 1 else None
    summary['patching'] = metrics
    summary['pairs_measured'] = len(pair_rows)
    if c['layers'] == 2:
        summary['chain_fraction'] = {}
        for convention in ('strict', 'legacy'):
            a, b = metrics['module_0']['effect_pp'], metrics['module_1']['effect_pp']
            numerator = metrics[f'{convention}_edge_0_1']['effect_pp'] * metrics[f'{convention}_edge_1_3']['effect_pp']
            summary['chain_fraction'][convention] = numerator/(a*b) if abs(a*b) > 1e-12 else None
        denom = metrics['module_0']['effect_pp']
        summary['joint_chain_fraction'] = metrics['joint_chain']['effect_pp']/denom if abs(denom) > 1e-12 else None
        summary['joint_chain_variants_pp'] = {key[len('joint_'):]: value['effect_pp']
                                             for key, value in metrics.items()
                                             if key.startswith('joint_')}
        summary['joint_chain_variants_pp']['serial'] = summary['joint_chain_variants_pp']['replace_last_only']
        summary['joint_chain_variants_pp']['cumulative'] = summary['joint_chain_variants_pp']['accumulate_all']
        # Recovery is the rebuttal's own convention: the joint effect over the A1 module effect.
        summary['joint_chain_recovery'] = {name: value/denom if abs(denom) > 1e-12 else None
                                          for name, value in summary['joint_chain_variants_pp'].items()}
        summary['scrub_gap_pp'] = 100*(metrics['scrub_hypothesis']['accuracy_to_clean']-metrics['scrub_null']['accuracy_to_clean'])
    (out/'metrics.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
