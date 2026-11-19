"""Rebuttal trajectory diagnostics; no training sampler/RNG consumption."""
import torch
from .data import Sampler


@torch.no_grad()
def value_diagnostics(model):
    rows = []
    for block in model.blocks:
        s = torch.linalg.svdvals(block['v'].weight.double())
        energy = s.square()
        rows.append(dict(singular_values=s.cpu().tolist(),
                         numerical_rank=int((s > s.max()*1e-8).sum()),
                         participation_rank=(energy.sum().square()/energy.square().sum()).item(),
                         condition_number=(s.max()/s.min()).item() if s.min() > 1e-12 else None))
    return rows


@torch.no_grad()
def probe_diagnostic(model, table, cfg):
    sampler = Sampler(table, cfg, cfg['seed']*101+991)
    labels, predictions, probes = [], [], []
    # Fixed held-out noise draw, independent of the optimizer's sample stream.
    for _ in range(4):
        x,y = sampler.sample(128)
        logits, cache = model(x.cuda())
        labels.append(y); predictions.append(logits.argmax(-1).cpu())
        probes.append(torch.stack([p(h).argmax(-1).cpu() for p,h in zip(model.probes,cache['residuals'])],1))
    y,p,z = torch.cat(labels),torch.cat(predictions),torch.cat(probes)
    macro = [sum((z[y==k,i]==k).float().mean().item() for k in y.unique())/len(y.unique()) for i in range(z.shape[1])]
    return dict(micro_accuracy=(p==y).float().mean().item(), probe_macro=macro,
                delta2_pp=100*(macro[3]-macro[2]) if len(macro)>=4 else None)
