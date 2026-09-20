"""Independent polynomial and normalized controls for the interaction census."""
import json
from pathlib import Path
import torch
from finite_mixed_residual_split import split

torch.manual_seed(1729);torch.set_num_threads(2)
x,a,b,i=[torch.randn(7,9,dtype=torch.float64) for _ in range(4)]
states=[x,x+a,x+b,x+a+b+i]
r=split(lambda z:z.square(),states)
quadratic_generated=float((r['generated']-2*a*b).abs().max())
quadratic_transport=float((r['transported']-(2*(x+a+b)*i+i.square())).abs().max())
linear=split(lambda z:3*z,states)
no_incoming=split(lambda z:z.square(),[x,x+a,x+b,x+a+b])
norm=lambda z:z/(z.square().mean(-1,keepdim=True)+1e-7).sqrt()
normalized=split(lambda z:norm(z).square(),states)
closure=max(float(v['closure'].abs().max()) for v in [r,linear,no_incoming,normalized])
out=dict(closure=closure,quadratic_generated=quadratic_generated,quadratic_transport=quadratic_transport,linear_generated=float(linear['generated'].abs().max()),no_incoming_transport=float(no_incoming['transported'].abs().max()),nonlinear_generated_norm=float(normalized['generated'].norm()),scope='Exact residual quartet identity; native layer census and causal attribution not yet tested.')
assert max(out[k] for k in ['closure','quadratic_generated','quadratic_transport','linear_generated','no_incoming_transport'])<1e-12
(Path(__file__).parent/'FINITE_MIXED_RESIDUAL_SPLIT_CONTROL.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out))
