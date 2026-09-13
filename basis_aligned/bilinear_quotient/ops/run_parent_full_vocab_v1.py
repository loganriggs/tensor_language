#!/usr/bin/env python3
# BQGATE:48prefixes;192suffixes;180seconds.
"""pred_a prior selected readouts replay <=1e-5 absolute.
pred_b centered vocabulary effect relative L2 <=10% each family/method.
pred_c baseline-probability weighted effect relative L2 <=10% each family/method.
pred_d candidate KL/native-removal KL <=.01 each family/method.
48prefixes;192suffixes;180seconds. Frozen lexical panel, unit parent removal,
both existing rank48 candidates. No data fitting; all50304modeled outputs.
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
    binding=json.loads((P/'PARENT_FULL_VOCAB_V1_BINDING.json').read_text())
    assert all(sha256(Path(k).read_bytes()).hexdigest()==v for k,v in binding['files'].items())
    rows=json.loads((P/'SHARED_TAIL_LEXICAL_V1_ROWS.json').read_text())['rows'];validate(rows)
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print('48prefixes;192suffixes;180seconds');return
    out=P/'PARENT_FULL_VOCAB_V1_RESULT.json';assert not out.exists()
    signal.alarm(180);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;start=time.perf_counter()
    model=load_model_fast().cuda().eval();weights=assembled.load_weights(model.state_dict(),'cuda')
    native=weights['routing'];candidates={}
    for name,file in [('query_product','SHARED_QUERY_PRODUCT_FIT_V1_PROGRAM.pt'),('complete_even','COMPLETE_EVEN_KEY_FIT_V1_PROGRAM.pt')]:
        rot=torch.load(P/file,weights_only=True)['basis_rotation'].cuda().double()
        candidates[name]=dict(native,key_basis=[native['key_basis'][0],native['key_basis'][1]@rot])
    w=torch.load(P/'MLP9_CROSSFIRST_RESPONSE_V1_PROGRAM.pt',weights_only=True)['direction'].cuda()
    prior=json.loads((P/'PARENT_KEY_MAIN_EFFECT_V1_RESULT.json').read_text())['cells']
    cells=[];replay=0.
    for index,row in enumerate(rows):
        ids=torch.tensor([row['ids']],device='cuda');live=live_prefix(model,ids,weights)
        norm=F.rms_norm(live['raw9'],(1152,));values=(norm.double()@native['current_value_reader'])[...,None]
        parents={'native':routing(norm,native,1)@values,**{name:routing(norm,p,1)@values for name,p in candidates.items()}}
        def score(h9):
            state=h9
            for block in model.transformer.h[10:]:state,_=block(state,live['v1'],live['x0'])
            return (30*torch.tanh(model.lm_head(F.rms_norm(state[:,-1],(1152,)))/30))[0].double()
        baseline=score(live['h9']);logbase=baseline.log_softmax(-1);prob=logbase.exp();outputs={}
        for name,amplitude in parents.items():
            z=live['raw9']+(live['att9']-(amplitude*w).to(live['att9'].dtype))
            h=z+model.transformer.h[9].mlp(F.rms_norm(z,(1152,)));outputs[name]=score(h)
            expected=next(a['scores'] for a in prior[index]['arms'] if a['method']==name and a['branch']=='parent' and a['strength']==1.)
            actual=[float(outputs[name][row['uk_id']]-outputs[name][row['us_id']]),float(outputs[name][row['control_ids'][0]]-outputs[name][row['control_ids'][1]])]
            replay=max(replay,max(abs(x-y) for x,y in zip(actual,expected)))
        reference=baseline-outputs['native'];centered=reference-reference.mean();weighted=reference-(prob*reference).sum()
        logref=outputs['native'].log_softmax(-1);pref=logref.exp();klref=(pref*(logref-logbase)).sum()
        for method in candidates:
            error=(baseline-outputs[method])-reference;ce=error-error.mean();we=error-(prob*error).sum()
            kl=(pref*(logref-outputs[method].log_softmax(-1))).sum()
            cells.append(dict(row=index,family=row['family'],method=method,
                centered_error2=float(ce.square().sum()),centered_reference2=float(centered.square().sum()),
                weighted_error2=float((prob*we.square()).sum()),weighted_reference2=float((prob*weighted.square()).sum()),
                candidate_kl=float(kl),native_removal_kl=float(klref)))
    groups=[]
    for family in range(4):
        for method in candidates:
            sub=[c for c in cells if c['family']==family and c['method']==method]
            total=lambda key:sum(c[key] for c in sub)
            groups.append(dict(family=family,method=method,
                centered_relative_error=(total('centered_error2')/total('centered_reference2'))**.5,
                weighted_relative_error=(total('weighted_error2')/total('weighted_reference2'))**.5,
                relative_kl=total('candidate_kl')/total('native_removal_kl'),
                mean_candidate_kl=total('candidate_kl')/len(sub),mean_native_removal_kl=total('native_removal_kl')/len(sub)))
    result={'pred_a':replay<=1e-5,'pred_b':all(g['centered_relative_error']<=.1 for g in groups),
        'pred_c':all(g['weighted_relative_error']<=.1 for g in groups),'pred_d':all(g['relative_kl']<=.01 for g in groups),
        'prior_readout_replay_maxabs':replay,'groups':groups,'cells':cells,'seconds':time.perf_counter()-start,
        'scope':__doc__+' Aggregate metrics do not imply preservation of every token or erase prior control sign failures.'}
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='cells'},indent=2))
if __name__=='__main__':main()
