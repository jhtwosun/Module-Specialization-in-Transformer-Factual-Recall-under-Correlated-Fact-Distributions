"""Seeded fact maps and matched counterfactuals; torch only on compute nodes."""
import torch


def facts(cfg, seed):
    """Mixed fact table, subject classes and the answer pool.

    The structured component is also available through facts_detail, because the
    correlation can be applied once to build a fixed table or at every presentation;
    the two give very different ceilings and the sources do not state which was used.
    """
    rng = torch.Generator().manual_seed(seed)
    ns, nr, m = cfg['n_subjects'], cfg['n_relations'], cfg['answers']
    kind, k, c = cfg['motif'], cfg['collision'], cfg['correlation']
    classes = torch.arange(ns) % (k if kind in ('subject', 'combined') else ns)
    pool = torch.arange(m)
    if kind == 'independent':
        table = torch.randint(m, (ns, nr), generator=rng)
        return table, classes, pool, table
    if kind in ('answer', 'combined'):
        ka = cfg['answer_pool'] if kind == 'combined' else k
        pool = torch.randperm(m, generator=rng)[:ka]
    if kind in ('subject', 'combined'):
        if cfg['block_sampling'] == 'legacy_without_replacement' and nr <= len(pool):
            blocks = torch.stack([pool[torch.randperm(len(pool), generator=rng)[:nr]] for _ in range(k)])
        else:
            blocks = pool[torch.randint(len(pool), (k, nr), generator=rng)]
        structured = blocks[classes]
    elif kind == 'relation':
        if cfg['block_sampling'] == 'legacy_without_replacement':
            blocks = torch.stack([torch.randperm(m, generator=rng)[:k] for _ in range(ns)])
        else:
            blocks = torch.randint(m, (ns, k), generator=rng)
        structured = blocks[:, torch.arange(nr) % k]
    elif kind == 'answer':
        structured = pool[torch.randint(len(pool), (ns, nr), generator=rng)]
    else:
        raise ValueError(kind)
    # The combined grid draws its uncorrelated component inside the answer pool, so at
    # a small pool every answer is a pool member. Drawing it from the full answer set
    # instead matches the single-axis motifs and is the other reading of the paper's
    # joint sweep; the two differ exactly where the published claim says answer
    # collision leaves a gap that subject collision cannot suppress.
    base_pool = pool if (kind == 'combined' and cfg.get('combined_base', 'pool') == 'pool') else torch.arange(m)
    base = base_pool[torch.randint(len(base_pool), (ns, nr), generator=rng)]
    table = torch.where(torch.rand(ns, nr, generator=rng) < c, structured, base)
    return table, classes, pool, structured


def facts_detail(cfg, seed):
    return facts(cfg, seed)


class Sampler:
    def __init__(self, table, cfg, seed, structured=None, pool=None):
        self.table, self.cfg = table.cpu(), cfg
        self.structured = (structured if structured is not None else table).cpu()
        self.pool = pool if pool is not None else torch.arange(cfg['answers'])
        self.rng = torch.Generator().manual_seed(seed)

    def sequence(self, s, r):
        c, g, n = self.cfg, self.rng, len(s)
        off = c['n_subjects'] + c['n_relations']
        x = torch.randint(off, off+c['n_noise'], (n, c['length']), generator=g)
        pos = torch.randint(c['length']-1, (n,), generator=g)
        idx = torch.arange(n)
        x[idx, pos] = s
        if c['legacy_repeat_relation']:
            rp = torch.randint(c['length']-2, (n,), generator=g)
            rp += (rp >= pos).long()
            x[idx, rp] = r + c['n_subjects']
        x[:, -1] = r + c['n_subjects']
        return x, pos

    def answers(self, s, r):
        """Label for one presentation of (s, r).

        fixed_table: the correlation was applied once, so every fact including the
            uncorrelated ones is memorisable and accuracy can approach 100%.
        per_example: the correlation is applied at each presentation, so only the
            structured component is learnable and accuracy is capped near c. For the
            answer motif the structured component is itself a fresh draw from the
            pool, which caps accuracy at one over the pool size.
        """
        c, g = self.cfg, self.rng
        mode = c.get('fact_sampling', 'fixed_table')
        if mode == 'fixed_table':
            return self.table[s, r]
        if mode == 'pool_resampled' and c['motif'] != 'answer':
            # Only the answer motif draws fresh. The rebuttal's answer-pool accuracy
            # stays at the pool's chance level after 20,000 steps, which a fixed table
            # cannot produce, while its subject and relation gaps need the fixed table
            # those motifs are measured on.
            return self.table[s, r]
        n = len(s)
        if c['motif'] == 'answer':
            structured = self.pool[torch.randint(len(self.pool), (n,), generator=g)]
        else:
            structured = self.structured[s, r]
        base = torch.randint(c['answers'], (n,), generator=g)
        return torch.where(torch.rand(n, generator=g) < c['correlation'], structured, base)

    def sample(self, n):
        s = torch.randint(self.cfg['n_subjects'], (n,), generator=self.rng)
        r = torch.randint(self.cfg['n_relations'], (n,), generator=self.rng)
        x, _ = self.sequence(s, r)
        return x, self.answers(s, r)

    def matched(self, n):
        # Draw uniformly from eligible clean facts, then uniformly from
        # different-answer subjects at that relation. Never accept collisions.
        eligible = [(s, r) for s in range(self.table.shape[0]) for r in range(self.table.shape[1])
                    if (self.table[:, r] != self.table[s, r]).any()]
        if not eligible:
            raise ValueError('No same-relation different-answer counterfactual exists')
        chosen = torch.randint(len(eligible), (n,), generator=self.rng)
        sr = torch.tensor([eligible[i] for i in chosen.tolist()])
        s, r = sr[:, 0], sr[:, 1]
        other = []
        for si, ri in zip(s, r):
            candidates = torch.where(self.table[:, ri] != self.table[si, ri])[0]
            other.append(candidates[torch.randint(len(candidates), (), generator=self.rng)].item())
        other = torch.tensor(other)
        clean, pos = self.sequence(s, r)
        corrupt = clean.clone()
        corrupt[torch.arange(n), pos] = other
        return clean, corrupt, self.table[s, r], self.table[other, r], s, r, other, pos

    def scrub_donors(self, s, r, y, classes, hypothesis):
        donors = []
        ns, nr = self.table.shape
        for step in range(3):
            selected = []
            for si, ri, yi in zip(s.tolist(), r.tolist(), y.tolist()):
                if not hypothesis:
                    pair = torch.tensor([torch.randint(ns, (), generator=self.rng), torch.randint(nr, (), generator=self.rng)])
                else:
                    if step == 0:
                        mask = (classes[:, None] == classes[si]).expand(ns, nr).clone()
                        mask[:, ri] = False
                    elif step == 1:
                        mask = torch.zeros(ns, nr, dtype=torch.bool)
                        mask[:, ri] = classes == classes[si]
                    else:
                        mask = self.table == yi
                    candidates = torch.nonzero(mask)
                    pair = candidates[torch.randint(len(candidates), (), generator=self.rng)]
                selected.append(pair)
            sr = torch.stack(selected)
            x, _ = self.sequence(sr[:, 0], sr[:, 1])
            donors.append((x, sr))
        return donors
