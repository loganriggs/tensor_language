"""Fresh-context removal, frozen shared component, CPU native FP64 suffix.

A zero-arm native contrast replay maxabs<=1e-4 and control Gram error<=1e-10.
B eachfamily nativegap>.1,>=4/6positive; removal>=.1nativegap,>=4/6positive.
C unrelated meanabs<=.5regional and true mean reduction>all8control means.
Eight orthogonal signed permutations, seed1424; no fit. All upstream native
states retained. Null: old-context selective removal fails this fresh panel.
"""
import json
from pathlib import Path
import torch
import torch.nn.functional as F
P=Path(__file__).resolve().parent


@torch.no_grad()
def main():
    torch.set_num_threads(2);rng=torch.Generator().manual_seed(1424);eps=torch.finfo(torch.float32).eps
    a=torch.load(P/'REGIONAL_VALUE_FRESH_GEO_V1_ARTIFACT.pt',weights_only=True,map_location='cpu')
    rows=json.loads((P/'REGIONAL_VALUE_FRESH_GEO_V1_ROWS.json').read_text())['rows'];pre=a['pre'].double();component=a['write_vertices'][:,0].double()
    binding=json.loads((P/'REGIONAL_VALUE_FRESH_GEO_V1_BINDING.json').read_text())['files']
    ck=next(path for path in binding if path.endswith('pytorch_model.bin'));sd=torch.load(ck,weights_only=True,mmap=True,map_location='cpu')
    l,r,d=[sd['transformer.h.17.mlp.'+n+'.weight'].double() for n in ('Left','Right','Down')];bias=sd['transformer.h.17.mlp.Down_bias'].double()
    ids=torch.tensor([[row['uk_id'],row['us_id'],*row['control_ids']] for row in rows]);u=sd['lm_head.weight'][ids].double()
    def score(write):
        y=pre-write;x=F.rms_norm(y,(1152,),eps=eps);h=y+((x@l.T)*(x@r.T))@d.T+bias
        logits=30*torch.tanh(torch.einsum('nd,nod->no',F.rms_norm(h,(1152,),eps=eps),u)/30)
        return torch.stack([logits[:,0]-logits[:,1],logits[:,2]-logits[:,3]],-1)
    base=score(torch.zeros_like(component));removed=score(component);controls=[];gram_errors=[];g=component@component.T
    for seed in range(8):
        perm=torch.randperm(1152,generator=rng);sign=2*torch.randint(2,(1152,),generator=rng)-1
        control=component[:,perm]*sign;gram_errors.append(float((control@control.T-g).norm()/g.norm()));controls.append(score(control))
    controls=torch.stack(controls);cells=[]
    for family in range(4):
        uk=[i for i,row in enumerate(rows) if row['family']==family and row['cue']=='British'];us=[i+1 for i in uk]
        native=base[uk]-base[us];reduction=native-(removed[uk]-removed[us])
        null=native[None]-(controls[:,uk]-controls[:,us])
        cells.append(dict(family=family,native_mean=float(native[:,0].mean()),native_positive=int((native[:,0]>0).sum()),
            removal_mean=float(reduction[:,0].mean()),removal_fraction=float(reduction[:,0].mean()/native[:,0].mean()),
            removal_positive=int((reduction[:,0]>0).sum()),regional_meanabs=float(reduction[:,0].abs().mean()),
            unrelated_meanabs=float(reduction[:,1].abs().mean()),control_mean_reductions=null[:,:,0].mean(1).tolist()))
    replay=float((base-a['arm_margins'][0]).abs().max());aa=replay<=1e-4 and max(gram_errors)<=1e-10
    bb=aa and all(c['native_mean']>.1 and c['native_positive']>=4 and c['removal_fraction']>=.1 and c['removal_positive']>=4 for c in cells)
    cc=bb and all(c['unrelated_meanabs']<=.5*c['regional_meanabs'] and c['removal_mean']>max(c['control_mean_reductions']) for c in cells)
    result={'pred_a':aa,'pred_b':bb,'pred_c':cc,'cells':cells,'native_replay_maxabs':replay,'max_control_gram_error':max(gram_errors),
        'scope':'Entire frozen shared component removal on fresh geography/templates. Native final MLP/RMS/readout evaluated in FP64 from saved native pre-state; all upstream dependencies remain. Eight matched controls and narrow unrelated contrasts, not broad selectivity or independent extraction.'}
    (P/'REGIONAL_FRESH_REMOVAL_V1_RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
