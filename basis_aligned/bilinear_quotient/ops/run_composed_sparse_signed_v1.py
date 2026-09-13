#!/usr/bin/env python3
# BQGATE:48prefixes;624suffixes;180seconds.
"""pred_a full parent field replay <=1e-10 and positive paired cue capability.
pred_b each arm/family target relative effect error <=10%.
pred_c each arm/family control relative effect error <=10%.
pred_d no material effect sign flips at1e-5.
48prefixes;624suffixes;180seconds. One frozen composed sparse64-node reader, two branches,
three signed strengths, frozen lexical rows. Exact child retained; native suffix.
"""
import sys,os,json,time,signal
from pathlib import Path
from hashlib import sha256
import torch
import torch.nn.functional as F
P=Path(__file__).resolve().parents[2]/'polynomial_causal';sys.path.insert(0,str(P))
from fastload import load_model_fast
from live_crossfirst_prefix_v1 import prepare as live_prefix,assembled
from regional_even_routing_v1 import routing
from regional_cue_row_check_v1 import validate

@torch.no_grad()
def main():
    binding=json.loads((P/'COMPOSED_SPARSE_SIGNED_V1_BINDING.json').read_text())
    assert all(sha256(Path(k).read_bytes()).hexdigest()==v for k,v in binding['files'].items())
    rows=json.loads((P/'SHARED_TAIL_LEXICAL_V1_ROWS.json').read_text())['rows'];validate(rows)
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print('48prefixes;624suffixes;180seconds');return
    out=P/'COMPOSED_SPARSE_SIGNED_V1_RESULT.json';assert not out.exists()
    signal.alarm(180);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;start=time.perf_counter()
    model=load_model_fast().cuda().eval();weights=assembled.load_weights(model.state_dict(),'cuda')
    native=weights['routing'];candidates={}
    from sparse_parent_reader_v1 import unpack
    item=torch.load(P/'SPARSE_COMPLETE_EVEN_FIT_V1_PROGRAM.pt',weights_only=True)['0.25']
    _,_,q=unpack(item)
    candidates['composed_sparse25']=dict(native,key_basis=[native['key_basis'][0],q.cuda()])
    w=torch.load(P/'MLP9_CROSSFIRST_RESPONSE_V1_PROGRAM.pt',weights_only=True)['direction'].cuda()
    cells=[];replays=[]
    for index,row in enumerate(rows):
        ids=torch.tensor([row['ids']],device='cuda');live=live_prefix(model,ids,weights)
        norm=F.rms_norm(live['raw9'],(1152,));values=(norm.double()@native['current_value_reader'])[...,None]
        parent=routing(norm,native,1)@values
        replays.append(float((parent-live['child']-live['remainder']).norm()/parent.norm().clamp_min(1e-30)))
        parents={'native':parent,**{name:routing(norm,p,1)@values for name,p in candidates.items()}}
        def score(h9):
            state=h9
            for block in model.transformer.h[10:]:state,_=block(state,live['v1'],live['x0'])
            logits=(30*torch.tanh(model.lm_head(F.rms_norm(state[:,-1],(1152,)))/30))[0]
            return [float(logits[row['uk_id']]-logits[row['us_id']]),float(logits[row['control_ids'][0]]-logits[row['control_ids'][1]])]
        baseline=score(live['h9']);arms=[]
        for branch in ('remainder','parent'):
            for strength in (-1.,1.,2.):
                for name,parent_i in parents.items():
                    amplitude=parent_i-live['child'] if branch=='remainder' else parent_i
                    z=live['raw9']+(live['att9']-(strength*amplitude*w).to(live['att9'].dtype))
                    h=z+model.transformer.h[9].mlp(F.rms_norm(z,(1152,)))
                    scores=score(h);arms.append(dict(branch=branch,strength=strength,method=name,scores=scores,effect=[a-b for a,b in zip(baseline,scores)]))
        cells.append(dict(row=index,family=row['family'],baseline=baseline,arms=arms))
    groups=[];capabilities=[]
    for family in range(4):
        sub=cells[family*12:(family+1)*12]
        baseline=torch.tensor([r['baseline'] for r in sub]);capabilities.append(float((baseline[::2,0]-baseline[1::2,0]).mean()))
        def effects(branch,strength,method):
            return torch.tensor([next(a['effect'] for a in r['arms'] if a['branch']==branch and a['strength']==strength and a['method']==method) for r in sub],dtype=torch.float64)
        for branch in ('remainder','parent'):
            for strength in (-1.,1.,2.):
                ref=effects(branch,strength,'native')
                for method in candidates:
                    pred=effects(branch,strength,method)
                    groups.append(dict(family=family,branch=branch,strength=strength,method=method,
                        relative_errors=((pred-ref).norm(dim=0)/ref.norm(dim=0)).tolist(),
                        reference_norms=ref.norm(dim=0).tolist(),material_sign_flips=(((ref*pred)<0)&(ref.abs()>=1e-5)).sum(0).tolist()))
    result={'pred_a':max(replays)<=1e-10 and min(capabilities)>0,
        'pred_b':all(g['relative_errors'][0]<=.1 for g in groups),
        'pred_c':all(g['relative_errors'][1]<=.1 for g in groups),
        'pred_d':all(sum(g['material_sign_flips'])==0 for g in groups),
        'groups':groups,'cells':cells,'parent_field_replay_max':max(replays),'capabilities':capabilities,
        'seconds':time.perf_counter()-start,'scope':__doc__+' Group results preserve each candidate verdict; no fitting or new semantic-selectivity claim.'}
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('cells','groups')},indent=2))
if __name__=='__main__':main()
