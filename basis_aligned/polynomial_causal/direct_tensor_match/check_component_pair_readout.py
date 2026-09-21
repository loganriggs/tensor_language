from pathlib import Path
import json,torch
from component_pair_readout import ComponentPairReadout
P=Path(__file__).parent;torch.set_num_threads(2);rows=[]
for seed,lam in enumerate([0.,.01,.1,1.,10.]):
 g=torch.Generator().manual_seed(9900+seed);r=lambda *s:torch.randn(*s,dtype=torch.float64,generator=g)
 Fa,Fb=r(20,4),r(20,5);ta,tb=r(20),r(20);Da,Db=r(64,4),r(64,5);A0,B0=r(64),r(64);y=r(64);energy=ta.square().sum()+tb.square().sum()
 m=ComponentPairReadout(Fa.T@Fa,Fb.T@Fb,Fa.T@ta,Fb.T@tb,energy,Da,Db,A0,B0,y,lam)
 wa,wb=r(4),r(5);dense=((Fa@wa-ta).square().sum()+(Fb@wb-tb).square().sum()+m.ridge*(wa.square().sum()+wb.square().sum()))/energy+lam*((A0+Da@wa)*(B0+Db@wb)-y).square().mean()/m.variance
 replay=abs(float(dense-m.value(wa,wb)));assert replay<1e-12
 gradients=[]
 for side in ('a','b'):
  w=m.update(wa,wb,side).requires_grad_();loss=m.value(w,wb) if side=='a' else m.value(wa,w)
  grad=float(torch.autograd.grad(loss,w)[0].abs().max());assert grad<1e-10;gradients.append(grad)
 outa,outb,history=m.fit(wa,wb,20)
 rows.append(dict(seed=seed,lam=lam,dense_loss_replay=replay,block_gradient_checks=gradients,initial_loss=history[0],final_loss=history[-1],monotonic=True))
(P/'COMPONENT_PAIR_READOUT_PREFLIGHT_V1.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(rows,indent=2))
