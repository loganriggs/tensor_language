"""Compare analytic contraction with independent residual-state autodiff."""
import json
from pathlib import Path
import torch
from final_readout_field_program import pack_fields
from readout_field_differential import contract

torch.manual_seed(613);torch.set_num_threads(2)
b,d,o,k=4,11,9,5
state=torch.randn(b,d,dtype=torch.float64);rows=torch.randn(b,o,2,d,dtype=torch.float64);jac=torch.randn(b,d,k,dtype=torch.float64)
fields=pack_fields(state,rows)
dn=torch.einsum('bovd,bdk->bovk',rows,jac).flatten(1,2);dq=2*torch.einsum('bd,bdk->bk',state,jac)/d
differentials=torch.cat([dn,dq[:,None]],dim=1)
actual=contract(fields,differentials)
x=torch.zeros(b,k,dtype=torch.float64,requires_grad=True)
h=state+torch.einsum('bdk,bk->bd',jac,x)
z=torch.nn.functional.rms_norm(h,(d,),eps=torch.finfo(torch.float32).eps)
logits=30*torch.tanh(torch.einsum('bovd,bd->bov',rows,z)/30);y=logits[:,:,0]-logits[:,:,1]
expected=torch.stack([torch.autograd.grad(y[:,i].sum(),x,retain_graph=i<o-1)[0] for i in range(o)],dim=1)
error=float((actual-expected).abs().max());assert error<1e-12
# Remove radial correction to ensure the normalizer term is live.
wrong=differentials.clone();wrong[:,-1]=0
tripwire=float((contract(fields,wrong)-expected).abs().max());assert tripwire>1e-3
out=dict(exact_replay=error,omitted_radial_error=tripwire,field_count=2*o+1,scope='Planted analytic final-readout differential; native context-transplant experiment pending. q differential includes state-dependent radial geometry.')
(Path(__file__).parent/'READOUT_FIELD_DIFFERENTIAL_CONTROL.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
