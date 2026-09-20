"""Model-free CPU replay and local degree-removal screen; no behavioral claim."""
import json
from pathlib import Path
import torch
from prepared_bilinear_response_tensor import evaluate
p=Path(__file__).resolve().parent
a=p.parent/'bilinear_quotient/circuits/followups'
torch.set_num_threads(2)
b=torch.load(a/'v4_mlp_response_fold_v1_program.pt',map_location='cpu',weights_only=False)
g=b['program'];z=b['beta'];r=z.shape[-1]
# Independent packed monomial construction, using Python loops and matrix products.
phi=torch.cat([z,torch.stack([z[:,i]*z[:,j] for i in range(r) for j in range(i,r)],-1)],-1)
s=g['s0']+(g['norm']*phi).sum(-1,keepdim=True)
carry=(g['carry'][0]@z[0])[None]
linear=(g['coefficients'][0,:,:r]@z[0])[None]/s
quad=(g['coefficients'][0,:,r:]@phi[0,r:])[None]/s
y=carry+linear+quad
reference=b['reference_response']
err=float((y-reference).abs().max());rel=float((y-reference).norm()/reference.norm())
zero=float(evaluate(g,torch.zeros_like(z)).abs().max())
assert rel<1e-8 and zero==0
assert float((evaluate(g,z)-y).abs().max())<1e-9
result=dict(independent_max_error=err,independent_relative_error=rel,zero_response=zero,
 stored_values=sum(v.numel() for v in g.values()),
 dropped_quadratic_relative_response_error=float(quad.norm()/y.norm()),
 dropped_linear_relative_response_error=float(linear.norm()/y.norm()),
 linear_quadratic_cosine=float((linear*quad).sum()/(linear.norm()*quad.norm())),
 scope='One exported native context, frozen beta. Degree-removal response-space screen only; no native output or OOD test.',
 prices=dict(canonical_per_context=498070,implicit_per_context=290710,implicit_shared_down=5308416,
 canonical_48=498070*48,implicit_48=290710*48+5308416))
f=p/'MLP_RESPONSE_EXPORT_CPU_V1.json';assert not f.exists();f.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
