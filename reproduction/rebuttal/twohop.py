"""Rebuttal two-hop probe and hop-separated patching reconstruction, Slurm only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from reproduction.synthetic.run import guard, resolve, dump


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    c=json.loads(a.config.read_text())
    if c['task'] not in ('composed','direct_control','flat_control'):
        raise ValueError('Invalid task')
    guard()
    import torch
    from torch import nn
    from torch.nn import functional as F
    from reproduction.synthetic.model import Transformer
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.manual_seed(c['seed']);torch.cuda.manual_seed_all(c['seed'])
    out=a.output;out.mkdir(parents=True,exist_ok=True)
    if any(f.name != f"execution-{os.environ['SLURM_JOB_ID']}.json" for f in out.iterdir()):
        raise FileExistsError('Choose a new output directory')
    dump(out/'config.json',c)
    dump(out/'provenance.json',dict(job=os.environ['SLURM_JOB_ID'],torch=torch.__version__,gpu=torch.cuda.get_device_name(),
        sources={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in Path(__file__).parent.glob('*.py')}))
    ns,ne,nr,na=(c[k] for k in ('n_subjects','n_entities','n_relations','answers'))
    g=torch.Generator().manual_seed(c['seed'])
    first=torch.randint(ne,(ns,nr),generator=g)
    second=torch.randint(na,(ne,nr),generator=g)
    flat=torch.randint(na,(ns,nr,nr),generator=g)
    torch.save(dict(first=first,second=second,flat=flat),out/'facts.pt')
    tc=resolve(dict(n_subjects=ns+ne,n_relations=nr,answers=na,n_noise=c['n_noise'],length=c['length'],
                    d_model=c['d_model'],layers=c['layers'],heads=c['heads']))
    model=Transformer(tc).cuda()
    model.probes.requires_grad_(False)  # Only intermediate-entity probes are trained here.
    probes=nn.ModuleList([nn.Linear(c['d_model'],ne) for _ in range(c['layers'])]).cuda()
    opt=torch.optim.AdamW(list(model.parameters())+list(probes.parameters()),lr=c['lr'],weight_decay=c['weight_decay'])

    def target(s,r1,r2):
        e=first[s,r1]
        return e, flat[s,r1,r2] if c['task']=='flat_control' else second[e,r2]

    def sample(n,rng):
        s=torch.randint(ns,(n,),generator=rng);r1=torch.randint(nr,(n,),generator=rng);r2=torch.randint(nr,(n,),generator=rng)
        e,y=target(s,r1,r2)
        x=torch.randint(ns+ne+nr,ns+ne+nr+c['n_noise'],(n,c['length']),generator=rng)
        pos=torch.randint(c['length']-2,(n,),generator=rng);rows=torch.arange(n)
        x[rows,pos]=s if c['task']!='direct_control' else ns+e
        if c['task']!='direct_control':x[:,-2]=ns+ne+r1
        x[:,-1]=ns+ne+r2
        return x,y,e,s,r1,r2,pos

    def pair_batch(kind,n):
        rng=torch.Generator().manual_seed(c['seed']+700+{'hop1':0,'hop2':1,'both':2}[kind])
        saved=[]
        for _ in range(100):
            x,y,e,s,r1,r2,pos=sample(max(n,64),rng)
            sp,rp,tp=s.clone(),r1.clone(),r2.clone()
            if kind in ('hop1','both'):
                sp=torch.randint(ns,s.shape,generator=rng);rp=torch.randint(nr,r1.shape,generator=rng)
            if kind in ('hop2','both'):tp=torch.randint(nr,r2.shape,generator=rng)
            ep,yp=target(sp,rp,tp)
            valid=(yp!=y)&((ep!=e) if kind in ('hop1','both') else (ep==e))
            if kind=='both':valid &= tp!=r2
            for i in torch.where(valid)[0].tolist():
                bx=x[i].clone()
                if kind in ('hop1','both'):
                    bx[pos[i]]=sp[i] if c['task']!='direct_control' else ns+ep[i]
                    if c['task']!='direct_control':bx[-2]=ns+ne+rp[i]
                if kind in ('hop2','both'):bx[-1]=ns+ne+tp[i]
                saved.append(dict(clean=x[i],corrupt=bx,target=y[i],corrupt_target=yp[i],entity=e[i],
                    corrupt_entity=ep[i],s=s[i],r1=r1[i],r2=r2[i],corrupt_s=sp[i],corrupt_r1=rp[i],corrupt_r2=tp[i],position=pos[i]))
                if len(saved)==n:
                    result={k:torch.stack([r[k] for r in saved]) for k in saved[0]}
                    assert (result['target']!=result['corrupt_target']).all()
                    if kind=='hop1':assert (result['r2']==result['corrupt_r2']).all()
                    elif kind=='hop2':
                        assert (result['entity']==result['corrupt_entity']).all()
                        assert ((result['clean']!=result['corrupt']).sum(1)==1).all()
                    return result
        raise RuntimeError('Insufficient hop-separated pairs')

    rng=torch.Generator().manual_seed(c['seed']*7+11)
    start = 0
    if c.get('checkpoint_path'):
        checkpoint = Path(c['checkpoint_path'])
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if saved['config'] != {k:v for k,v in c.items() if k!='checkpoint_path'}:
            raise ValueError('Checkpoint training protocol mismatch')
        model.load_state_dict(saved['model']);probes.load_state_dict(saved['probes'])
        opt.load_state_dict(saved['optimizer']);rng.set_state(saved['sampler_rng'])
        torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng'])
        start = saved['step']
        dump(out/'parent_checkpoint.json',dict(path=str(checkpoint),sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),step=start))
    for step in range(start+1,c['steps']+1):
        x,y,e,*_=sample(c['batch_size'],rng)
        logits,cache=model(x.cuda())
        task_loss=F.cross_entropy(logits,y.cuda())
        loss=task_loss+sum(F.cross_entropy(p(cache['residuals'][2*i+1].detach()),e.cuda()) for i,p in enumerate(probes))
        opt.zero_grad(set_to_none=True);loss.backward();opt.step()
        if step%100==0 or step==1:
            row=dict(step=step,task_loss=task_loss.item(),loss=loss.item())
            with (out/'training.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            print(json.dumps(row),flush=True)
        if step%c['checkpoint_every']==0 or step==c['steps']:
            torch.save(dict(step=step,config=c,model=model.state_dict(),probes=probes.state_dict(),optimizer=opt.state_dict(),
                sampler_rng=rng.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()),out/'checkpoint.tmp')
            (out/'checkpoint.tmp').replace(out/'checkpoint.pt')
    model.eval();probes.eval()
    eval_rng=torch.Generator().manual_seed(c['seed']+321)
    raw=[]
    with torch.no_grad():
        for start in range(0,c['eval_samples'],c['batch_size']):
            x,y,e,*_=sample(min(c['batch_size'],c['eval_samples']-start),eval_rng)
            logits,cache=model(x.cuda())
            z=torch.stack([p(cache['residuals'][2*i+1]).argmax(-1).cpu() for i,p in enumerate(probes)],1)
            raw.append(dict(x=x,y=y,e=e,pred=logits.argmax(-1).cpu(),entity_pred=z))
        raw={k:torch.cat([row[k] for row in raw]) for k in raw[0]}
        torch.save(raw,out/'evaluation.pt')
        def macro(pred,y):return sum((pred[y==k]==k).float().mean().item() for k in y.unique())/len(y.unique())
        metrics=dict(micro_accuracy=(raw['pred']==raw['y']).float().mean().item(),macro_accuracy=macro(raw['pred'],raw['y']),
            entity_probe_accuracy=[(raw['entity_pred'][:,i]==raw['e']).float().mean().item() for i in range(c['layers'])],
            entity_probe_macro=[macro(raw['entity_pred'][:,i],raw['e']) for i in range(c['layers'])],entity_random_baseline=1/ne,patching={})
        for kind in (('hop1','hop2','both') if c['task']=='direct_control' else ('hop1','hop2')):
            pairs=pair_batch(kind,c['pairs']);torch.save(pairs,out/f'{kind}_pairs.pt')
            rows=[]
            for start in range(0,c['pairs'],c['batch_size']):
                cx=pairs['clean'][start:start+c['batch_size']].cuda();bx=pairs['corrupt'][start:start+c['batch_size']].cuda()
                yy=pairs['target'][start:start+c['batch_size']].cuda()
                cl,cc=model(cx);bl,bc=model(bx)
                # Check identity and exact final-state restoration, not just successful execution.
                torch.testing.assert_close(model(bx,residual_replacements={1:bc['residuals'][1]})[0],bl)
                torch.testing.assert_close(model(bx,residual_replacements={2*c['layers']-1:cc['residuals'][-1]})[0],cl)
                scores={'clean':cl,'corrupt':bl}
                for i in range(c['layers']):scores[f'L{i+1}']=model(bx,residual_replacements={2*i+1:cc['residuals'][2*i+1]})[0]
                probs={k:v.softmax(-1).gather(1,yy[:,None]).squeeze(1).cpu().tolist() for k,v in scores.items()}
                for j in range(len(yy)):
                    rows.append(dict(pair=start+j,target=yy[j].item(),probability={k:v[j] for k,v in probs.items()}))
            with (out/f'{kind}_patching.jsonl').open('w') as f:
                for row in rows:f.write(json.dumps(row)+'\n')
            baseline=sum(r['probability']['corrupt'] for r in rows)/len(rows)
            metrics['patching'][kind]={k:dict(probability=sum(r['probability'][k] for r in rows)/len(rows),
                effect_pp=100*(sum(r['probability'][k] for r in rows)/len(rows)-baseline)) for k in rows[0]['probability']}
        dump(out/'metrics.json',metrics)
    print('TWOHOP_OK',flush=True)


if __name__=='__main__':main()
