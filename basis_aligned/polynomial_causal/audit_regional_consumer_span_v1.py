"""Registered two-consumer manipulation stress; no fitting.

A old ten-arm FP64 suffix versus saved native contrasts maxabs<=1e-4.
B worst relative write error over each family's complete 2D branch span<=.1.
C opposing and bounded per-row worst-write directions regional effect<=.1
in each family. Selected directions are adversarial diagnostics, not fresh OOD.
"""
import json
from pathlib import Path
import torch
import torch.nn.functional as F
P=Path(__file__).resolve().parent
EPS=torch.finfo(torch.float32).eps


def worst(k,e):
    g=k@k.T;v,b=torch.linalg.eigh(g)
    assert v[0]>1e-12*v[-1], 'Singular reference span needs explicit null-space treatment'
    white=b@torch.diag(v.rsqrt())
    ev,vec=torch.linalg.eigh(white.T@(e@e.T)@white)
    c=white@vec[:,-1];c=c/c.abs().max()
    return float(ev[-1].clamp_min(0).sqrt()),c,float(v[0]/v[-1])


@torch.no_grad()
def main():
    torch.set_num_threads(2)
    a=torch.load(P/'REGIONAL_ROUTING_CONSUMER_REUSE_V1_ARTIFACT.pt',weights_only=True,map_location='cpu')
    rows=json.loads((P/'REGIONAL_SOURCE_BLOCK_OOD_V1_ROWS.json').read_text())['rows']
    w=a['write_vertices'].double();pre=a['pre'].double();k=w[:,1:3]-w[:,0,None];approx=w[:,4:6]-w[:,0,None];err=approx-k
    perrow=[worst(k[i],err[i]) for i in range(len(rows))]
    direction=torch.stack([r[1] for r in perrow]);families=[]
    for family in range(4):
        ids=[i for i,r in enumerate(rows) if r['family']==family]
        bound,c,cond=worst(k[ids].transpose(0,1).flatten(1),err[ids].transpose(0,1).flatten(1))
        families.append(dict(family=family,span_bound=bound,worst_coefficients=c.tolist(),gram_eigen_ratio=cond,
                             max_individual_bound=max(perrow[i][0] for i in ids)))
    binding=json.loads((P/'REGIONAL_ROUTING_CONSUMER_REUSE_V1_BINDING.json').read_text())['files']
    ck=next(n for n in binding if n.endswith('pytorch_model.bin'))
    sd=torch.load(ck,weights_only=True,mmap=True,map_location='cpu')
    l,r,d=[sd['transformer.h.17.mlp.'+n+'.weight'].double() for n in ('Left','Right','Down')]
    bias=sd['transformer.h.17.mlp.Down_bias'].double()
    ids=torch.tensor([[row['uk_id'],row['us_id'],*row['control_ids']] for row in rows]);assert ids.shape==(48,4)
    u=sd['lm_head.weight'][ids].double()
    def score(delta):
        y=pre+delta;x=F.rms_norm(y,(1152,),eps=EPS)
        h=y+((x@l.T)*(x@r.T))@d.T+bias
        logits=30*torch.tanh(torch.einsum('nd,nod->no',F.rms_norm(h,(1152,),eps=EPS),u)/30)
        return torch.stack([logits[:,0]-logits[:,1],logits[:,2]-logits[:,3]],-1)
    baseline=score(torch.zeros_like(pre))
    replay=max(float((score(w[:,j]-w[:,0])-a['arm_margins'][j]).abs().max()) for j in range(10))
    cells=[]
    for name,c in [('opposing',torch.tensor([1.,-1.],dtype=torch.float64).expand(48,2)),('worst_write',direction)]:
        refwrite=torch.einsum('ni,nid->nd',c,k);predwrite=torch.einsum('ni,nid->nd',c,approx)
        ref=score(refwrite)-baseline;pred=score(predwrite)-baseline
        for family in range(4):
            ii=[i for i,row in enumerate(rows) if row['family']==family]
            target=ref[ii,0];delta=pred[ii,0]-target
            cells.append(dict(edit=name,family=family,effect_relative_error=float(delta.norm()/target.norm().clamp_min(1e-30)),
                effect_norm=float(target.norm()),effect_absolute_error=float(delta.norm()),
                write_relative_error=float((predwrite[ii]-refwrite[ii]).norm()/refwrite[ii].norm()),
                unrelated_native_norm=float(ref[ii,1].norm()),unrelated_prediction_error=float((pred[ii,1]-ref[ii,1]).norm())))
    result={'pred_a':replay<=1e-4,'pred_b':all(f['span_bound']<=.1 for f in families),
            'pred_c':replay<=1e-4 and all(c['effect_relative_error']<=.1 for c in cells),
            'suffix_replay_maxabs':replay,'families':families,'cells':cells,
            'perrow_worst_bound':[r[0] for r in perrow],'perrow_worst_coefficients':direction.tolist(),
            'scope':'Complete two-consumer write-span certificate on saved contexts; bounded adversarial port edits with native FP64 suffix. Not fresh text, token-realizability, all-input guarantee, or closed producer generation.'}
    (P/'REGIONAL_CONSUMER_SPAN_V1_RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({key:value for key,value in result.items() if not key.startswith('perrow')},indent=2))


if __name__=='__main__':main()
