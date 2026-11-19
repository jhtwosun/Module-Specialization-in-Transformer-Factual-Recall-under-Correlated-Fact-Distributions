"""Compute-node invariant tests: python -m reproduction.synthetic.tests_gpu."""
from .run import guard, resolve


def main():
    guard()
    import torch
    from .data import facts, Sampler
    from .model import Transformer
    if not torch.cuda.is_available():
        raise RuntimeError('GPU tests require CUDA')
    c = resolve(dict(n_subjects=8, n_relations=4, answers=8, n_noise=4, length=6, d_model=16))
    torch.manual_seed(42)
    table, groups, _, _ = facts(c, 0)
    assert torch.equal(table, facts(c, 0)[0])
    sampler = Sampler(table, c, 123)
    clean, corrupt, y, yc, s, r, other, pos = sampler.matched(16)
    assert (y != yc).all() and (s != other).all()
    assert ((clean != corrupt).sum(1) == 1).all()
    assert ((clean >= c['n_subjects']) & (clean < c['n_subjects']+c['n_relations'])).sum(1).eq(1).all()
    donors = sampler.scrub_donors(s, r, y, groups, True)
    assert (groups[donors[0][1][:, 0]] == groups[s]).all()
    assert (donors[0][1][:, 1] != r).all()
    assert (groups[donors[1][1][:, 0]] == groups[s]).all()
    assert (donors[1][1][:, 1] == r).all()
    assert (table[donors[2][1][:, 0], donors[2][1][:, 1]] == y).all()
    model = Transformer(c).cuda().eval()
    x = clean.cuda()
    logits, cache = model(x)
    # Probe-only objective cannot alter backbone gradients.
    sum(p(h.detach()).sum() for p, h in zip(model.probes, cache['residuals'])).backward()
    assert all(p.grad is None for name, p in model.named_parameters() if not name.startswith('probes.'))
    for i in range(4):
        patched, _ = model(x, replacements={i: cache['outputs'][i][:, -1]})
        torch.testing.assert_close(patched, logits)
    for convention in ('strict', 'legacy'):
        for source in (0, 1):
            for receiver in list(range(source+1, 4))+['direct']:
                torch.testing.assert_close(model.edge(cache, cache, source, receiver, convention), logits)
    for i in range(4):
        mean = cache['outputs'][i][:, -1].mean(0)
        actual, _ = model(x, mean_replacements={i: mean}, mean_scope='query')
        expected, _ = model(x, replacements={i: mean.expand(len(x), -1)})
        torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(model.joint_chain(cache, cache), logits)
    _, cc = model(corrupt.cuda())
    source_noop = dict(cache, outputs=[cc['outputs'][0]] + cache['outputs'][1:])
    torch.testing.assert_close(model.joint_chain(cc, source_noop), model(corrupt.cuda())[0])
    # Strict edge never carries source delta directly into readout.
    source, receiver = 0, 3
    h = cc['inputs'][receiver].clone()
    h[:, -1] += cache['outputs'][source][:, -1]-cc['outputs'][source][:, -1]
    expected = model.unembed(cc['final'][:, -1]+model.contribution(receiver, h)[:, -1]-cc['outputs'][receiver][:, -1])
    torch.testing.assert_close(model.edge(cc, cache, source, receiver), expected)
    print('synthetic GPU invariants passed', flush=True)


if __name__ == '__main__':
    main()
