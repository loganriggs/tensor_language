#!/usr/bin/env python3
# BQGATE:48prefixes;720suffixes;240seconds.
"""pred_a parent replay<=1e-10 relative and child-only output exact.
pred_b centered full-vocabulary effect error<=10% each family/coefficient pair.
pred_c probability-weighted effect error<=10% each family/coefficient pair.
pred_d candidate/native-intervention KL ratio<=.01 each family/coefficient pair.
48prefixes;720suffixes;240seconds. Frozen sparse parent, exact child, native
combined suffix for seven independent coefficient pairs. No new text fitting.
"""
import os,sys,json,time,signal
from pathlib import Path
from hashlib import sha256
import torch
import torch.nn.functional as F
P=Path(__file__).resolve().parents[2]/'polynomial_causal';sys.path.insert(0,str(P))
from fastload import load_model_fast
from live_crossfirst_prefix_v1 import prepare as live_prefix,assembled
from regional_even_routing_v1 import routing
from regional_cue_row_check_v1 import validate
from sparse_parent_reader_v1 import unpack
PAIRS=((1,0),(0,1),(1,1),(1,-1),(-1,1),(2,1),(1,2))


@torch.no_grad()
def main():
    binding=json.loads((P/'SPARSE_HIERARCHY_COMPOSITION_V1_BINDING.json').read_text())
    assert all(sha256(Path(k).read_bytes()).hexdigest()==v for k,v in binding['files'].items())
    rows=json.loads((P/'SHARED_TAIL_LEXICAL_V1_ROWS.json').read_text())['rows'];validate(rows)
    assert len(rows)==48
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print('48prefixes;720suffixes;240seconds');return
    out=P/'SPARSE_HIERARCHY_COMPOSITION_V1_RESULT.json';assert not out.exists()
    signal.alarm(240);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;start=time.perf_counter()
    model=load_model_fast().cuda().eval();weights=assembled.load_weights(model.state_dict(),'cuda');native=weights['routing']
    item=torch.load(P/'SPARSE_COMPLETE_EVEN_FIT_V1_PROGRAM.pt',weights_only=True)['0.25'];_,_,q=unpack(item)
    candidate=dict(native,key_basis=[native['key_basis'][0],q.cuda()])
    w=torch.load(P/'MLP9_CROSSFIRST_RESPONSE_V1_PROGRAM.pt',weights_only=True)['direction'].cuda()
    cells=[];replay=0.;child_only_error=0.
    for index,row in enumerate(rows):
        live=live_prefix(model,torch.tensor([row['ids']],device='cuda'),weights)
        norm=F.rms_norm(live['raw9'],(1152,));values=(norm.double()@native['current_value_reader'])[...,None]
        parent=routing(norm,native,1)@values;cp=routing(norm,candidate,1)@values;child=live['child']
        replay=max(replay,float((parent-child-live['remainder']).norm()/parent.norm().clamp_min(1e-30)))
        def score(h9):
            state=h9
            for block in model.transformer.h[10:]:state,_=block(state,live['v1'],live['x0'])
            return (30*torch.tanh(model.lm_head(F.rms_norm(state[:,-1],(1152,)))/30))[0].double()
        baseline=score(live['h9']);logbase=baseline.log_softmax(-1);prob=logbase.exp()
        for a,b in PAIRS:
            outputs=[]
            for p in (parent,cp):
                amplitude=a*child+b*(p-child)
                z=live['raw9']+(live['att9']-(amplitude*w).to(live['att9'].dtype))
                h=z+model.transformer.h[9].mlp(F.rms_norm(z,(1152,)));outputs.append(score(h))
            ref=baseline-outputs[0];error=outputs[0]-outputs[1]
            if b==0:child_only_error=max(child_only_error,float(error.abs().max()))
            centered=ref-ref.mean();weighted=ref-(prob*ref).sum()
            ce=error-error.mean();we=error-(prob*error).sum()
            logref=outputs[0].log_softmax(-1);pref=logref.exp()
            cells.append(dict(row=index,family=row['family'],a=a,b=b,
                centered_error2=float(ce.square().sum()),centered_reference2=float(centered.square().sum()),
                weighted_error2=float((prob*we.square()).sum()),weighted_reference2=float((prob*weighted.square()).sum()),
                candidate_kl=float((pref*(logref-outputs[1].log_softmax(-1))).sum()),
                native_intervention_kl=float((pref*(logref-logbase)).sum())))
    groups=[]
    for family in range(4):
        for a,b in PAIRS:
            sub=[c for c in cells if c['family']==family and c['a']==a and c['b']==b]
            total=lambda key:sum(c[key] for c in sub)
            assert total('centered_reference2')>0 and total('weighted_reference2')>0 and total('native_intervention_kl')>0
            groups.append(dict(family=family,a=a,b=b,
                centered_relative_error=(total('centered_error2')/total('centered_reference2'))**.5,
                weighted_relative_error=(total('weighted_error2')/total('weighted_reference2'))**.5,
                relative_kl=total('candidate_kl')/total('native_intervention_kl'),
                weighted_reference2=total('weighted_reference2')))
    result={'pred_a':replay<=1e-10 and child_only_error==0,
        'pred_b':all(g['centered_relative_error']<=.1 for g in groups),
        'pred_c':all(g['weighted_relative_error']<=.1 for g in groups),
        'pred_d':all(g['relative_kl']<=.01 for g in groups),
        'groups':groups,'cells':cells,'parent_replay':replay,'child_only_max_abs_error':child_only_error,
        'seconds':time.perf_counter()-start,'scope':__doc__+' Same-writer hierarchy, existing lexical styles, not multiple independent semantic behaviors or global circuit composition.'}
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('cells','groups')},indent=2))


if __name__=='__main__':main()
