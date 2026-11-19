"""Pre-LN transformer with explicit additive-module interventions."""
import math
import torch
from torch import nn


class Transformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg['d_model']
        self.heads = cfg['heads']
        self.tau = cfg['temperature']
        # Architecture details the retained sources leave open. The surviving toy code
        # attends bidirectionally, adds learned positions and has no final norm; the
        # published accuracies are far below what that configuration reaches, so each
        # detail is a switch rather than a silent assumption.
        self.causal = cfg.get('attention', 'bidirectional') == 'causal'
        self.use_positions = cfg.get('positional', 'learned') == 'learned'
        self.final_norm = nn.LayerNorm(d) if cfg.get('final_norm', False) else None
        self.token_emb = nn.Embedding(cfg['n_subjects'] + cfg['n_relations'] + cfg['n_noise'], d)
        self.pos_emb = nn.Embedding(cfg['length'], d)
        self.blocks = nn.ModuleList([nn.ModuleDict({
            'ln1': nn.LayerNorm(d), 'ln2': nn.LayerNorm(d),
            **{k: nn.Linear(d, d, bias=False) for k in ('q', 'k', 'v', 'o')},
            'mlp': nn.Sequential(nn.Linear(d, 4*d), nn.ReLU(), nn.Linear(4*d, d)),
        }) for _ in range(cfg['layers'])])
        self.unembed = nn.Linear(d, cfg['answers'])
        self.probes = nn.ModuleList([nn.Linear(d, cfg['answers']) for _ in range(2*cfg['layers'])])

        # Rebuttal R02: initialization is an explicit reconstruction choice.
        mode = cfg.get('value_init', 'default')
        with torch.no_grad():
            for block in self.blocks:
                w = block['v'].weight
                if mode == 'small_normal':
                    nn.init.normal_(w, std=cfg['value_init_std'])
                elif mode == 'scaled_identity':
                    w.copy_(torch.eye(d)*cfg['value_init_scale'] + torch.randn_like(w)*cfg['identity_noise'])
                elif mode == 'orthogonal':
                    nn.init.orthogonal_(w, gain=cfg['value_init_scale'])

    def contribution(self, index, h):
        block = self.blocks[index // 2]
        if index % 2:
            return block['mlp'](block['ln2'](h))
        b, t, d = h.shape
        z = block['ln1'](h)
        q, k, v = [block[name](z).view(b, t, self.heads, d//self.heads).transpose(1, 2)
                   for name in ('q', 'k', 'v')]
        scores = (q @ k.transpose(-1, -2))/(math.sqrt(d//self.heads)*self.tau)
        if self.causal:
            mask = torch.ones(t, t, dtype=torch.bool, device=h.device).triu(1)
            scores = scores.masked_fill(mask, float('-inf'))
        weights = scores.softmax(-1)
        return block['o']((weights @ v).transpose(1, 2).contiguous().view(b, t, d))

    def forward(self, x, replacements=None, mean_replacements=None, mean_scope="all", residual_replacements=None):
        h = self.token_emb(x)
        if self.use_positions:
            h = h + self.pos_emb(torch.arange(x.shape[1], device=x.device))[None]
        inputs, outputs, residuals = [], [], []
        for i in range(len(self.probes)):
            inputs.append(h)
            out = self.contribution(i, h)
            if mean_replacements and i in mean_replacements:
                if mean_scope == 'all':
                    out = mean_replacements[i][None, None].expand_as(out)
                elif mean_scope == 'query':
                    out = out.clone()
                    out[:, -1] = mean_replacements[i]
                else:
                    raise ValueError(mean_scope)
            if replacements and i in replacements:
                out = out.clone()
                out[:, -1] = replacements[i]
            outputs.append(out)
            h = h + out
            if residual_replacements and i in residual_replacements:
                h = h.clone()
                h[:, -1] = residual_replacements[i]
            residuals.append(h[:, -1])
        final = self.final_norm(h) if self.final_norm is not None else h
        # 'residuals' is the cumulative stream after each module, which is what the
        # joint probes read; 'module_states' is each module's own output, which is
        # what the frozen-probe appendix describes fitting probes on.
        module_states = [out[:, -1] for out in outputs]
        return self.unembed(final[:, -1]), {'inputs': inputs, 'outputs': outputs,
                                            'residuals': residuals, 'module_states': module_states,
                                            'final': final}

    def edge(self, corrupt, clean, source, receiver, convention='strict'):
        """Strict: recompute only receiver; freeze every other module output.

        Legacy: reproduce cycle32 downstream recomputation and its final-MLP
        direct residual carry. Both conventions are reported independently.
        """
        delta = clean['outputs'][source][:, -1] - corrupt['outputs'][source][:, -1]
        if receiver == 'direct':
            return self.unembed(corrupt['final'][:, -1] + delta)
        h = corrupt['inputs'][receiver].clone()
        h[:, -1] += delta
        new = self.contribution(receiver, h)
        if convention == 'strict':
            final = corrupt['final'][:, -1] + new[:, -1] - corrupt['outputs'][receiver][:, -1]
            return self.unembed(final)
        if receiver == len(self.probes)-1:
            return self.unembed((h + new)[:, -1])
        h = corrupt['inputs'][receiver] + new
        for i in range(receiver+1, len(self.probes)):
            h = h + self.contribution(i, h)
        return self.unembed(h[:, -1])

    def joint_chain_deltas(self, corrupt, clean, carry):
        """Per-module deltas along A1 -> M1 -> M2 under one carry convention.

        carry='replace': each receiver sees only the previous module's recomputed
            delta, so the chain is isolated step by step.
        carry='accumulate': the deltas accumulate, so each receiver sees everything
            the chain has changed up to that point.
        """
        if len(self.probes) != 4:
            raise ValueError('joint_chain requires two layers')
        first = clean['outputs'][0][:, -1] - corrupt['outputs'][0][:, -1]
        deltas, carried = [first], first
        for receiver in (1, 3):
            h = corrupt['inputs'][receiver].clone()
            h[:, -1] += carried
            recomputed = self.contribution(receiver, h)[:, -1] - corrupt['outputs'][receiver][:, -1]
            deltas.append(recomputed)
            carried = recomputed if carry == 'replace' else carried + recomputed
        return deltas

    # Which of (A1, M1, M2) reach the readout. The rebuttal writes "joint
    # A1->M1->M2->direct patch" without saying which contributions arrive at the
    # unembedding, and the readings differ by tens of points, so all of them are
    # measured and none is chosen silently.
    JOINT_READOUTS = {'last_only': (2,), 'tail': (1, 2), 'a1_last': (0, 2), 'all': (0, 1, 2)}

    def joint_chain_readings(self, corrupt, clean):
        """Logits for every carry convention crossed with every readout subset."""
        out = {}
        for carry in ('replace', 'accumulate'):
            deltas = self.joint_chain_deltas(corrupt, clean, carry)
            for name, indices in self.JOINT_READOUTS.items():
                total = sum(deltas[i] for i in indices)
                out[f'{carry}_{name}'] = self.unembed(corrupt['final'][:, -1] + total)
        return out

    def joint_chain(self, corrupt, clean, mode='serial'):
        """Backwards-compatible names: serial is replace_last_only, cumulative is accumulate_all."""
        readings = self.joint_chain_readings(corrupt, clean)
        return readings['replace_last_only' if mode == 'serial' else 'accumulate_all']
