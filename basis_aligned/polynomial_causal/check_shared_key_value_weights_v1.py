"""Weight-only diagnostic, independent Gaussian queries/sources, no normalization.

Equal query/key rotary frame (zero relative displacement). This is an algebraic
control, not a sample from causal attention or the retained interaction ports.
"""
import json
from pathlib import Path
import torch
from shared_key_value_moment_v1 import moment
from head17_source_interface_v1 import CHECKPOINT

torch.set_num_threads(2)
torch.manual_seed(7131206)
sd=torch.load(CHECKPOINT,map_location='cpu',weights_only=True,mmap=True)
def head(name):
    return sd[f'transformer.h.17.attn.{name}.weight'].reshape(9,128,1152)[2].double()
q=torch.randn(256,1152,dtype=torch.float64)
a=(q@head('c_q').T)@head('c_k')
b=(q@head('c_q2').T)@head('c_k2')
f=(1-float(sd['transformer.h.17.attn.lamb']))*head('c_v')
c=a.square().sum(-1)*b.square().sum(-1)+2*(a*b).sum(-1).square()
independent=c.mean()*(f@f.T)
shared=moment(f,a,b).mean(0)
def spectrum(h):
    e=torch.linalg.eigvalsh(h).flip(0).clamp_min(0)
    return {'rank':int(torch.linalg.matrix_rank(h)),
            'rank64_tail_fraction':float(e[64:].sum()/e.sum())}
result={
    'queries':256,'seed':7131206,
    'relative_covariance_change':float((shared-independent).norm()/shared.norm()),
    'correction_trace_fraction':float(torch.trace(shared-independent)/torch.trace(shared)),
    'independent_key_value':spectrum(independent),'shared_key_value':spectrum(shared),
    'scope':__doc__,
    'interpretation':'Conditional rank-two correction need not remain rank two after averaging queries. No compression fit or behavior claim.'}
out=Path(__file__).with_name('SHARED_KEY_VALUE_WEIGHTS_V1_RESULT.json')
with out.open('x') as h:json.dump(result,h,indent=2);h.write('\n')
print(json.dumps(result,indent=2))
