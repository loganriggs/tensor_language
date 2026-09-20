"""Known rank-one quartic: exact coefficient loss gradient versus sampled-entry gradients."""
import json,math
from pathlib import Path
import torch
P=Path(__file__).resolve().parent;torch.set_num_threads(1);torch.set_default_dtype(torch.float64);records=[]
for d in [6,32,128,1152]:
 torch.manual_seed(1639);u=torch.randn(d);u/=u.norm();raw=torch.randn(d,requires_grad=True);a=raw/raw.norm();exact=(a.square().sum())**4+1-2*(a@u)**4;g=torch.autograd.grad(exact,raw)[0].detach();cos=float((a@u).detach());gn=float(g.norm())
 for n in [256,4096]:
  for mode in ['full_sample','exact_self_sampled_cross']:
   grads=[];losses=[]
   for seed in range(20):
    torch.manual_seed(seed);ids=torch.randint(d,(n,4));a=raw/raw.norm();s=a[ids].prod(-1);t=u[ids].prod(-1)
    loss=d**4*(s-t).square().mean() if mode=='full_sample' else (a.square().sum())**4+1-2*d**4*(s*t).mean()
    grads.append(torch.autograd.grad(loss,raw)[0].detach());losses.append(float(loss.detach()))
   G=torch.stack(grads);noise=(G-g).square().sum(-1).mean().sqrt();alignment=(G@g)/(G.norm(dim=-1)*g.norm()).clamp_min(1e-30)
   records.append(dict(dimension=d,samples=n,mode=mode,initial_direction_cosine=cos,exact_loss=float(exact.detach()),exact_gradient_norm=gn,gradient_relative_rmse=float(noise/g.norm()),mean_gradient_cosine=float(alignment.mean()),fraction_negative_alignment=float((alignment<0).double().mean()),loss_relative_rmse=float(((torch.tensor(losses)-exact.detach()).square().mean()).sqrt()/exact.detach()),repetitions=20))
  print(d,n,[(r['mode'],r['gradient_relative_rmse'],r['mean_gradient_cosine']) for r in records[-2:]],flush=True)
(P/'QUARTIC_GRADIENT_NOISE_V1.json').write_text(json.dumps(dict(records=records,scope='Planted exactly rank-one quartic and representable student, random orientation. Squared Frobenius loss/gradient available analytically;20 stochastic replicates. Dimension-specific one initialization, not a theorem or native gradient measurement.'),indent=2)+'\n')
