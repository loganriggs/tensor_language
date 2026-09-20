"""Independent signed diagonal gauge audit of the complete conditional DAG."""
import copy
import json
from pathlib import Path
import torch
from conditional_attention_mlp_runtime import execute
from projected_bilinear_response import readout_prepared

ROOT=Path(__file__).resolve().parents[2]
artifact=ROOT/'basis_aligned/bilinear_quotient/circuits/followups/subject_conditional_dag_v670.pt'
package=torch.load(artifact,map_location='cpu',weights_only=True)
torch.set_num_threads(2)
scales=[torch.tensor([.5,-2.,.8,1.3,-.9,1.5,2.2,.7],dtype=torch.float64).roll(i) for i in range(7)]

def geometry(matrix,s):return matrix*s[:,None]*s[None,:]
def readout(f,c,s):
    f=copy.deepcopy(f);c=copy.deepcopy(c)
    f['reader']*=s;f['geometry']=geometry(f['geometry'],s);c['overlap']*=s
    return f,c

def attention_gauge(ap,s):
    ap=copy.deepcopy(ap)
    for name in ['score1','score2']:
        score=ap[name];score['query']*=s;score['key']*=s
        score['joint']*=s[:,None]*s[None,:]
        for name in ['qnorm','knorm']:
            norm=score[name];norm['linear']*=s;norm['gram']*=s[:,None]*s[None,:]
    g=ap['global_norm'];g['linear']*=s;g['gram']=geometry(g['gram'],s)
    ap['value_base']/=s;ap['cached_value']/=s
    ap['value_basis']*=s[None,:]/s[:,None]
    return ap

changed_runtime=copy.deepcopy(package['runtime'])
for i,b in enumerate(changed_runtime['blocks']):
    sin,sout=scales[i],scales[i+1]
    b['coefficients']*=sin[b['pair_i']]*sin[b['pair_j']]
    b['coefficients']/=sout[:,None]
    b['carry']*=sin[None,:]/sout[:,None]
    b['geometry']=geometry(b['geometry'],sin)
errors=[];control_errors=[]
for case in package['cases']:
    changed=copy.deepcopy(case)
    changed['initial']/=scales[0]
    changed['attention']=[attention_gauge(ap,scales[i]) for i,ap in enumerate(case['attention'])]
    for i,c in enumerate(changed['contexts']):
        c['linear']*=scales[i][None,:]/scales[i+1][:,None]
        c['overlap']*=scales[i];c['baseline_write']/=scales[i+1]
    changed_runtime['readout'],changed['final_context']=readout(package['runtime']['readout'],case['final_context'],scales[-1])
    cf,cc=readout(case['control_fixed'],case['control_context'],scales[-1])
    logits,z=execute(changed_runtime,changed['attention'],changed['initial'],changed['contexts'],changed['final_context'],changed['positions'])
    controls=readout_prepared(cf,z,cc)
    errors.append(float((logits-case['reference_logits']).abs().max()))
    control_errors.append(float((controls-case['reference_controls']).abs().max()))
assert max(errors+control_errors)<=1e-8
assert not torch.cuda.is_initialized()
result=dict(max_logit_error=max(errors),max_control_logit_error=max(control_errors),
    transformed_boundaries=7,rows=8,passed=True,
    scope='Signed diagonal coordinate-gauge invariance only; does not identify unique semantic features')
(ROOT/'basis_aligned/polynomial_causal/CONDITIONAL_DAG_GAUGE_AUDIT_2026-09-20.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
