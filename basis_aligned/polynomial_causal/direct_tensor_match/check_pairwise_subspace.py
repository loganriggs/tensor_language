from pathlib import Path
import json,torch
from shared_subspace_loss import SharedSubspaceLoss
from pairwise_subspace_loss import PairwiseSubspaceLoss,from_common
P=Path(__file__).parent;torch.set_num_threads(2);rows=[]
for case in range(5):
 rng=torch.Generator().manual_seed(31000+case);r=2*(1+case%3);p=2+case%2;d=r+p+3
 rand=lambda *s:torch.randn(*s,dtype=torch.float64,generator=rng)
 initial=[rand(d,r)]+[rand(d,p) for j in range(3)]
 old=SharedSubspaceLoss(torch.stack([torch.eye(d,dtype=torch.float64)]*6))
 targets=[]
 for U in old.spans(initial):
  K=rand(2,r+p,r+p);K=(K+K.transpose(-1,-2))/2;targets.append(U@K@U.T)
 T=torch.cat(targets);metric=PairwiseSubspaceLoss(T);converted=from_common(initial)
 error=float(metric.loss(converted,True));assert error<1e-20
 params=[(a+.2*rand(*a.shape)).requires_grad_() for a in converted]
 implicit=metric.loss(params);explicit=metric.loss(params,True);gi=torch.autograd.grad(implicit,params);ge=torch.autograd.grad(explicit,params)
 replay=max(float((a-b).norm()/(1+a.norm())) for a,b in zip(gi,ge));assert replay<1e-8 and abs(float((implicit-explicit).detach()))<1e-10
 rows.append(dict(case=case,exact_inherited_span_squared_error=error,dense_gradient_replay=replay,private_width=p,shared_pair_width=r//2))
# Prospective graph: each pair reads two64dictionaries through128x160 map,
# then224private projections. Retain384products and two readouts per pair.
projection=1152*(3*64+3*224)+3*128*160;products=3*384;readout=2*products
price=dict(projection_multiplications=projection,products=products,readout_multiplications=readout,source_multiplications=projection+products+readout,stored_floats=projection+readout+11532,baseline_multiplications=1330560,required_maximum=.8*1330560,saving_fraction=1-(projection+products+readout)/1330560,scope='Prospective sparse product topology only; NOT the dense-core relaxation. Literal two dictionary reads per pair, with no stored structural-zero blocks.')
assert price['source_multiplications']<=price['required_maximum']
(P/'PAIRWISE_SUBSPACE_PREFLIGHT_V1.json').write_text(json.dumps(dict(controls=rows,prospective_graph_price=price,scope='Five exact inherited-span and dense-gradient controls. Pairwise dictionaries can move independently after an exact inherited initialization. No native fit or adoption.'),indent=2)+'\n');print(json.dumps(price))
