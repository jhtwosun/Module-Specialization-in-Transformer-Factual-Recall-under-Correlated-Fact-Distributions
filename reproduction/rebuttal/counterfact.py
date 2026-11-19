"""Full-table CounterFact entropy profile; original JSON schema, stdlib only."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import urllib.request

SOURCE='https://rome.baulab.info/data/dsets/counterfact.json'
# P102 stays listed so the profile keeps reporting it as absent from the official
# dataset; P1412 is the disclosed substitute actually executed (see rebuttal/matrix.py).
AUTHOR_BUCKETS={'P103':'subject','P102':'subject','P1412':'subject','P495':'subject','P30':'answer','P17':'answer','P131':'answer','P19':'independent','P27':'independent','P407':'independent'}


def entropy(counts):
    n=sum(counts.values())
    return -sum((v/n)*math.log2(v/n) for v in counts.values()) if n else 0.


def profile(records):
    groups=defaultdict(list)
    ids=set()
    for record in records:
        if record['case_id'] in ids:raise ValueError('Duplicate case_id')
        ids.add(record['case_id'])
        rewrite=record['requested_rewrite']
        groups[rewrite['relation_id']].append((rewrite['subject'],str(rewrite['target_true'].get('id',rewrite['target_true']['str']))))
    result=[]
    for relation, pairs in sorted(groups.items()):
        marginal=Counter(a for _,a in pairs)
        conditional=defaultdict(Counter)
        for s,a in pairs:conditional[s][a]+=1
        h=entropy(marginal)
        cond=sum(sum(c.values())*entropy(c) for c in conditional.values())/len(pairs)
        result.append(dict(relation=relation,facts=len(pairs),subjects=len(conditional),answers=len(marginal),
            eligible_ge50=len(pairs)>=50,author_bucket=AUTHOR_BUCKETS.get(relation),
            entropy_bits=h,normalized_by_observed_answers=h/math.log2(len(marginal)) if len(marginal)>1 else 0.,
            normalized_by_fact_count=h/math.log2(len(pairs)) if len(pairs)>1 else 0.,
            conditional_entropy_given_subject_bits=cond,
            singleton_subject_fraction=sum(sum(c.values())==1 for c in conditional.values())/len(conditional),
            answer_counts=dict(marginal)))
    return result


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Use Slurm for download/profile')
    c=json.loads(a.config.read_text());source=Path(c['dataset_path'])
    if not source.exists():
        source.parent.mkdir(parents=True,exist_ok=True)
        with urllib.request.urlopen(SOURCE,timeout=90) as response:
            data=response.read()
        # Validate downloaded JSON before publishing into the shared dataset path.
        if not isinstance(json.loads(data),list):raise ValueError('CounterFact must be a JSON list')
        temp=source.with_suffix('.download');temp.write_bytes(data);temp.replace(source)
    records=json.loads(source.read_text())
    rows=profile(records)
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/'config.json').write_text(json.dumps(c,indent=2)+'\n')
    provenance=dict(source_url=SOURCE,sha256=hashlib.sha256(source.read_bytes()).hexdigest(),job=os.environ['SLURM_JOB_ID'],
                    record_count=len(records),relation_count=len(rows),eligible_relations=sum(r['eligible_ge50'] for r in rows),
                    answer_identity='target_true.id, falling back to target_true.str',subject_identity='requested_rewrite.subject')
    (a.output/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    (a.output/'relations.json').write_text(json.dumps(rows,indent=2)+'\n')
    metric=dict(record_count=len(records),relation_count=len(rows),eligible_relations=sum(r['eligible_ge50'] for r in rows),
                selected={r['relation']:{k:v for k,v in r.items() if isinstance(v,(int,float))} for r in rows if r['relation'] in AUTHOR_BUCKETS})
    (a.output/'metrics.json').write_text(json.dumps(metric,indent=2)+'\n')
    print(json.dumps(provenance),flush=True)
    print('COUNTERFACT_PROFILE_OK',flush=True)


if __name__=='__main__':main()
