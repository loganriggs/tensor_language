"""Paired output scaling: initial equality, gradient chain rule, and finite-query rank bound."""
import itertools,json,math
from pathlib import Path
import torch
from implicit_quartic import entries
P=Path(__file__).resolve().parent;torch.set_num_threads(1);torch.set_default_dtype(torch.float64);d,o,k,r=6,5,4,3;scale=.1/math.sqrt(o*r);idx=torch.tensor(list(itertools.product(range(d),repeat=4)))
torch.manual_seed(1622);teacher=[torch.randn(*s) for s in [(o,5),(5,4),(5,4),(4,7),(7,d),(7,d)]];target=entries(*teacher,idx);target/=target.square().sum(-1).mean().sqrt()
U,S,V=torch.linalg.svd(target,full_matrices=False);bound=float(S[r:].norm()/S.norm());best=(U[:,:r]*S[:r])@V[:r];assert abs(float((target-best).norm()/target.norm())-bound)<1e-12
rows=[]
for optimizer in ['adam','muon']:
 bundles=[]
 for mode in ['physical','scaled']:
  torch.manual_seed(10);raw=torch.randn(o,r);C=torch.nn.Parameter(raw*scale if mode=='physical' else raw);L=torch.nn.Parameter(torch.randn(r,k));R=torch.nn.Parameter(torch.randn(r,k));A=torch.nn.Parameter(torch.randn(k,d));B=torch.nn.Parameter(torch.randn(k,d));params=[C,L,R,A,B];D=torch.eye(k)
  def coeff():return entries(C if mode=='physical' else C*scale,L/L.norm(dim=1,keepdim=True),R/R.norm(dim=1,keepdim=True),D,A/A.norm(dim=1,keepdim=True)*math.sqrt(d),B/B.norm(dim=1,keepdim=True)*math.sqrt(d),idx)
  before=coeff();loss=(before-target).square().mean();loss.backward();grads=[p.grad.detach().clone() for p in params];opt=torch.optim.Adam(params,lr=.05) if optimizer=='adam' else torch.optim.Muon(params,lr=.05,weight_decay=0.,adjust_lr_fn='match_rms_adamw');opt.step();after=coeff();error=float(((after-target).norm()/target.norm()).detach());assert error+1e-12>=bound
  bundles.append(dict(before=before.detach(),grads=grads));rows.append(dict(optimizer=optimizer,parameterization=mode,initial_relative_error=float(((before-target).norm()/target.norm()).detach()),after_one_step_error=error,relative_function_movement=float(((after-before).norm()/before.norm()).detach())))
 a,b=bundles;initial=float((a['before']-b['before']).norm()/a['before'].norm());chain=float((b['grads'][0]-scale*a['grads'][0]).norm()/b['grads'][0].norm());other=max(float((u-v).norm()/u.norm()) for u,v in zip(a['grads'][1:],b['grads'][1:]));assert max(initial,chain,other)<1e-12
out=dict(initial_relative_discrepancy=initial,output_gradient_chain_rule_error=chain,other_parameter_gradient_error=other,finite_query_output_rank_lower_bound=bound,records=rows,scope='Exact paired initial function and gradients; one-step optimizer geometry differs. SVD bound is for the explicitly enumerated query matrix only. No native scaling winner predicted.')
(P/'QUARTIC_PARAMETERIZATION_CHECK_V1.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
