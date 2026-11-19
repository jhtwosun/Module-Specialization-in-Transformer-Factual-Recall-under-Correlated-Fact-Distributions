"""Run only on Slurm GPU: python -m reproduction.pretrained.selftest."""
import os


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise SystemExit('Self-test requires a Slurm GPU allocation')
    import torch
    from .model import Instrumented
    from .data import Synthetic
    if not torch.cuda.is_available():
        raise SystemExit('Self-test requires CUDA')
    torch.manual_seed(42)
    for architecture in ('gpt_neox', 'llama'):
        c = {'model': 'random-smoke-only', 'random_init': True, 'architecture': architecture, 'dtype': 'float32'}
        model = Instrumented(c).cuda().eval()
        x = torch.randint(0, 100, (2, 8), device='cuda')
        with torch.no_grad():
            actual, cache = model(x)
            expected = model.base(x, use_cache=False).logits[:, -1]
            torch.testing.assert_close(actual, expected)
            noop, _ = model(x, module=(1, cache['out'][1]))
            torch.testing.assert_close(actual, noop)
            noop, _ = model(x, edge=(2, torch.zeros_like(cache['out'][0])))
            torch.testing.assert_close(actual, noop)
            altered = x.clone()
            altered[:, -1] = (altered[:, -1] + 1) % 100
            a = model.body(x, output_hidden_states=True).last_hidden_state[:, :-1]
            b = model.body(altered, output_hidden_states=True).last_hidden_state[:, :-1]
            torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)
            for layer in model.layers:
                assert not layer._forward_hooks and not layer._forward_pre_hooks
        del model
    for legacy in (False, True):
        data = Synthetic({'experiment': 'synthetic_transfer', 'legacy_repeat_relation': legacy}, 'independent', 0)
        x, _, _ = data.sample(64)
        relation_count = ((x >= data.ns) & (x < data.ns + data.nr)).sum(1)
        assert relation_count.eq(2 if legacy else 1).all()
        assert ((x[:, -1] >= data.ns) & (x[:, -1] < data.ns + data.nr)).all()
    for experiment in ('synthetic_transfer', 'two_hop'):
        data = Synthetic({'experiment': experiment}, 'independent', 0)
        for clean, corrupt, target, meta in data.pairs(5, 0):
            assert (clean != corrupt).sum().item() == 1
            assert target.item() != meta['corrupt_target']
    print('PASS: native-forward equivalence, no-op patches, causality, hook cleanup, matched pairs')


if __name__ == '__main__':
    main()
