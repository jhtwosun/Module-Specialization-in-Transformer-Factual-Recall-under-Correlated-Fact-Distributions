"""Generate explicit rebuttal configurations; no tensor libraries or submission."""
import argparse
import json
from pathlib import Path
from reproduction.matrix import synthetic_cells, pretrained_cells
from reproduction.fetch_models import cached_revisions

# The rebuttal names P102 for the subject bucket, but the verified official
# CounterFact snapshot (21919 facts, 34 relations) contains no P102. The research
# lead selected P1412 ('languages spoken, written or signed') as a disclosed
# substitute: it is the closest subject-axis analogue of P103 ('mother tongue'),
# with 924 facts, H(A|R)=3.573 bits and H(A|S,R)=0.002 bits. This is an explicit
# reconstruction choice, not recovery of whatever relation the rebuttal ran.
NATURAL_RELATIONS = ('P1412', 'P495', 'P17', 'P131', 'P27', 'P407')
NATURAL_SUBSTITUTIONS = {'P1412': 'P102'}


def cells(family, seeds=(0,1,2), dataset_path=None):
    base = next(c for c in synthetic_cells('causal') if c['motif']=='subject' and c['seed']==0)
    if family == 'decay':
        axes = [dict(weight_decay=x,probe_weight_decay=x) for x in (0.,.001,.01,.1)]
    elif family == 'init':
        axes = [dict(value_init=x,value_init_scale=1.,value_init_std=.02,identity_noise=.001) for x in ('small_normal','scaled_identity','orthogonal')]
    elif family == 'correlation':
        axes = [dict(motif=m,collision=2 if m=='subject' else 4,answer_pool=4,correlation=c,d_model=64)
                for m in ('subject','relation','answer') for c in (.5,.7,.9,1.)]
    elif family == 'relation':
        axes = [dict(motif='relation',collision=16,d_model=64,steps=2000)]
    elif family == 'joint':
        axes = [{}]
    elif family == 'answer_convergence':
        axes = [dict(motif='answer',collision=4,answer_pool=4,steps=20000,warmup_steps=1000)]
    elif family == 'natural':
        original = pretrained_cells('natural', dataset_path)
        models = sorted({c['model'] for c in original})
        revisions = cached_revisions()
        return [dict(experiment='natural_counterfact',model=m,revision=revisions.get(m,'main'),
                     local_files_only=m in revisions,seeds=[s],
                     dataset_path=str(Path(dataset_path).resolve()),relations=[r],
                     # Budget chosen by the research lead on 2026-10-01 from measured cost:
                     # 2.00 s/pair on Pythia-1.4B and 4.42 s/pair on LLaMA-3.1-8B across the
                     # layer sweep. The caption's 200 pairs for every known fact would be ~34 h
                     # per Pythia-1.4B cell. 100 facts x 20 pairs keeps fact coverage wide at
                     # 2,000 pairs per cell; the cap is recorded in every cell's metrics.
                     max_facts=100,pairs_per_fact=20,max_pairs_total=2000,min_module_effect_pp=1)
                for m in models for r in NATURAL_RELATIONS for s in seeds]
    elif family == 'twohop':
        return [dict(task=task,seed=s,steps=8000,batch_size=128,n_subjects=32,n_entities=32,
                     n_relations=16,answers=64,d_model=128,layers=4,length=16,heads=4,n_noise=64,
                     lr=.001,weight_decay=.01,eval_samples=4096,pairs=200,checkpoint_every=1000)
                for task in ('composed','direct_control','flat_control') for s in seeds]
    else:
        raise ValueError(family)
    return [dict(base,**dict(axis,seed=s)) for axis in axes for s in seeds]


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('family',choices=('decay','init','correlation','relation','joint','answer_convergence','twohop','natural'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seeds',help='Default 0,1,2, or 0..7 for twohop')
    p.add_argument('--dataset-path')
    a=p.parse_args()
    seeds=tuple(map(int,a.seeds.split(','))) if a.seeds else tuple(range(8 if a.family=='twohop' else 3))
    configs=cells(a.family,seeds,a.dataset_path)
    a.output.mkdir(parents=True,exist_ok=False)
    for i,c in enumerate(configs): (a.output/f'{i:04d}.json').write_text(json.dumps(c,indent=2)+'\n')
    print(len(configs),'configurations; no jobs submitted')


if __name__=='__main__':main()
