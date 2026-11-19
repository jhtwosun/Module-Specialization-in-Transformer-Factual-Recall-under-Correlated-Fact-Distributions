"""Synthetic distributions preserve the surviving cycle29 torch RNG protocol."""
import torch


class Synthetic:
    def __init__(self, c, motif, seed):
        self.ns, self.nr, self.na = c.get('n_subjects', 128), c.get('n_relations', 128), c.get('n_answers', 256)
        self.legacy_repeat_relation = c.get('legacy_repeat_relation', False)
        self.T, self.noise = c.get('sequence_length', 16), c.get('n_noise', 64)
        self.two = c['experiment'] == 'two_hop'
        self.rng = torch.Generator().manual_seed(seed)
        g = self.rng
        shape = (self.ns, self.nr)
        strength = c.get('collision', .9)
        if motif == 'subject':
            k = c.get('subject_groups', 2)
            b = torch.stack([torch.randperm(self.na, generator=g)[:self.nr] for _ in range(k)])
            structured = b[torch.arange(self.ns) % k]
        elif motif == 'answer':
            k = c.get('answer_pool', 4)
            pool = torch.randperm(self.na, generator=g)[:k]
            structured = pool[torch.randint(k, shape, generator=g)]
        elif motif != 'independent':
            raise ValueError(motif)
        self.facts = torch.randint(self.na, shape, generator=g)
        if motif != 'independent':
            self.facts = torch.where(torch.rand(shape, generator=g) < strength, structured, self.facts)
        self.second = torch.randint(self.na, (self.na, self.nr), generator=g) if self.two else None
        self.rng.manual_seed(seed * 7 + 11)

    def sample(self, n):
        g = self.rng
        s = torch.randint(self.ns, (n,), generator=g)
        r = torch.randint(self.nr, (n,), generator=g)
        r2 = torch.randint(self.nr, (n,), generator=g) if self.two else None
        x = torch.randint(self.ns + self.nr, self.ns + self.nr + self.noise, (n, self.T), generator=g)
        rows = torch.arange(n)
        loc = torch.randint(self.T - 1, (n,), generator=g)
        rloc = torch.randint(self.T - 2, (n,), generator=g)
        rloc += (rloc >= loc).long()
        # NLB task.py, EOS_token=False: one relation at query, no duplicate.
        x[rows, loc], x[:, -1] = s, r + self.ns
        if self.legacy_repeat_relation:
            x[rows, rloc] = r + self.ns
        if self.two:
            # Ordered query suffix disambiguates composition; exclude it from subject location.
            loc = torch.randint(self.T - 2, (n,), generator=g)
            x = torch.randint(self.ns + self.nr, self.ns + self.nr + self.noise, (n, self.T), generator=g)
            x[rows, loc], x[:, -2], x[:, -1] = s, r + self.ns, r2 + self.ns
        y = self.facts[s, r]
        if self.two:
            y = self.second[y, r2]
        return x, y, (s, r, r2, loc)

    def pairs(self, n, seed):
        self.rng.manual_seed(seed + 7777)
        count = 0
        for _ in range(n * 100):
            x, y, (s, r, r2, loc) = self.sample(1)
            s2 = torch.randint(self.ns, (1,), generator=self.rng)
            z = self.facts[s2, r]
            if self.two:
                z = self.second[z, r2]
            if y.item() == z.item():
                continue
            corrupt = x.clone()
            corrupt[0, loc.item()] = s2.item()
            yield x, corrupt, y, {'subject': s.item(), 'corrupt_subject': s2.item(), 'relation': r.item(),
                                 'relation2': r2.item() if self.two else None, 'corrupt_target': z.item()}
            count += 1
            if count == n:
                return
        raise RuntimeError(f'Only {count}/{n} matched pairs after bounded rejection sampling')
