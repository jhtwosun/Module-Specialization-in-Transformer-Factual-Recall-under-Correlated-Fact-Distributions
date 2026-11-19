"""Generate explicit paper experiment cells, without importing ML libraries."""
from __future__ import annotations
import argparse
import itertools
import json
from pathlib import Path


SYNTHETIC_FAMILIES = ("hierarchy", "capacity", "depth", "causal", "path", "threshold", "combined", "frozen", "scale")
PRETRAINED_FAMILIES = ("pythia", "natural", "twohop")


def synthetic_cells(family):
    base = dict(n_subjects=128, n_relations=128, answers=256, n_noise=64,
                length=16, d_model=128, heads=4, layers=2, temperature=1.0,
                correlation=.9, steps=8000, batch_size=128, lr=.001,
                weight_decay=.01, probe_lr=.001, probe_weight_decay=.01,
                probe_mode="joint", probe_steps=4000, eval_samples=4096,
                mean_samples=2048, pairs=200, checkpoint_every=1000,
                legacy_repeat_relation=False, block_sampling="legacy_without_replacement")
    motifs = [dict(motif="independent"), dict(motif="subject", collision=2),
              dict(motif="relation", collision=2), dict(motif="answer", collision=4, answer_pool=4)]
    cells = []
    if family == "hierarchy":
        cells = [motifs[0]]
        for motif in ("subject", "relation"):
            cells += [dict(motif=motif, collision=k) for k in (128,64,32,16,8,4,2)]
        cells += [dict(motif="answer", collision=k, answer_pool=k) for k in (256,128,64,32,16,8,4,2)]
    elif family == "capacity":
        # Paper ranges Kr<=4 and Ka<=8 are ambiguous: expose BOTH, never choose
        # whichever reproduces the target after observing results.
        # Full sweeps retain axis-specific high-K baselines as well as iid.
        axis_cells = [dict(c) for c in synthetic_cells("hierarchy") if c['seed'] == 0]
        for d, motif in itertools.product((64,128,256), axis_cells):
            cells.append(dict(motif, d_model=d))
    elif family == "depth":
        cells = [dict(m, layers=l) for l, m in itertools.product((1,2,3,4), (motifs[0],motifs[1],motifs[3]))]
    elif family in ("causal", "path"):
        cells = [motifs[0],motifs[1],motifs[3]]
    elif family == "threshold":
        cells = [dict(motif=m, **{"collision": k, **({"answer_pool": k} if m == "answer" else {})})
                 for m,k in itertools.product(("subject","answer"),(128,64,32,8,2))]
    elif family == "combined":
        # combined_base="full" draws the uncorrelated component from the whole answer
        # set, as the single-axis motifs do. Drawing it inside the pool instead shrinks
        # the whole task to the pool at small Ka and erases the residual gap the paper
        # reports there; with the full range, max(subject, answer) becomes the best
        # closed form at 7.17 pp against 26.29 under the pool reading.
        cells = [dict(motif="combined", collision=ks, answer_pool=ka, block_sampling="with_replacement",
                      combined_base="full")
                 for ks,ka in itertools.product((128,64,32,16,8,4,2),(256,128,64,32,16,8,4,2))]
    elif family == "frozen":
        cells = [dict(m, probe_mode="frozen") for m in (motifs[0],motifs[1],motifs[3])]
    elif family == "scale":
        cells = [dict(m,d_model=d) for d,m in itertools.product((128,256,512,768),(motifs[0],motifs[1],motifs[3]))]
    else:
        raise ValueError(f"Unknown family {family}")
    return [dict(base, **dict(c,seed=seed)) for c,seed in itertools.product(cells, range(3))]


def pretrained_cells(family, dataset_path=None):
    base = dict(revision="main", steps=12000, body_lr=5e-5, head_lr=.005,
                batch_size=32, eval_examples=2000, pairs=200, checkpoint_every=1000)
    if family == "natural":
        if not dataset_path:
            raise ValueError("Natural transfer requires --dataset-path CounterFact JSON; no synthetic fallback")
        # Revisions come from the local weight cache so a run records which weights it used,
        # and the pair budget is the measured-cost budget documented in rebuttal/matrix.py.
        from reproduction.fetch_models import cached_revisions
        revisions = cached_revisions()
        return [dict(base, experiment="natural_counterfact", model=model, seeds=[seed],
                     revision=revisions.get(model, "main"), local_files_only=model in revisions,
                     dataset_path=str(Path(dataset_path).resolve()), relations=[relation],
                     max_facts=100, pairs_per_fact=20, max_pairs_total=2000, min_module_effect_pp=1)
                for model,relation,seed in itertools.product(
                    ("EleutherAI/pythia-1.4b", "EleutherAI/pythia-2.8b",
                     "meta-llama/Llama-3.2-1B-Instruct", "meta-llama/Llama-3.1-8B"),
                    ("P103","P30","P19"),range(3))]
    # Same revision pinning as the natural family, so a fine-tuning run records the
    # weights it started from.
    from reproduction.fetch_models import cached_revisions
    revisions = cached_revisions()
    if family == "pythia":
        model = "EleutherAI/pythia-160m"
        return [dict(base, experiment="synthetic_transfer", model=model,
                     revision=revisions.get(model, "main"), local_files_only=model in revisions,
                     motifs=[motif], seeds=[seed], source_layers=[3,5], target_layers=[11])
                for motif,seed in itertools.product(("independent","subject","answer"),range(3))]
    if family == "twohop":
        model = "EleutherAI/pythia-1.4b"
        return [dict(base, experiment="two_hop", model=model,
                     revision=revisions.get(model, "main"), local_files_only=model in revisions,
                     motifs=["independent"], seeds=[seed], source_layers=list(range(23)),
                     target_layers=list(range(1,24))) for seed in range(3)]
    raise ValueError(f"Unknown family {family}")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("family", choices=SYNTHETIC_FAMILIES + PRETRAINED_FAMILIES)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--dataset-path", type=Path)
    a=p.parse_args()
    cells = synthetic_cells(a.family) if a.family in SYNTHETIC_FAMILIES else pretrained_cells(a.family,a.dataset_path)
    a.output.mkdir(parents=True,exist_ok=False)
    for i,cfg in enumerate(cells):
        (a.output / f"{i:04d}.json").write_text(json.dumps(cfg,indent=2,sort_keys=True)+"\n")
    print(f"{len(cells)} cells written to {a.output}; no jobs submitted")


if __name__ == "__main__":
    main()
