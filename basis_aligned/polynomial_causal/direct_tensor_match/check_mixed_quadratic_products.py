"""Five planted mixed-product structures, independent dense gradients and fits."""
from pathlib import Path
import torch,json,time,math
from shared_quadratic_products import mixed_coefficient_loss,materialize_mixed
P=Path(__file__).resolve().parent;torch.set_num_threads(2);start=time.perf_counter();records=[]
for index,name in enumerate(['indefinite_shared','private_outputs','shared_left','signed_outputs','paired_squares']):
 g=torch.Generator().manual_seed(370+index);d=8;r=4;o=6
 L=torch.randn(d,r,generator=g,dtype=torch.float64)/d**.5;R=torch.randn(d,r,generator=g,dtype=torch.float64)/d**.5;W=torch.randn(o,r,generator=g,dtype=torch.float64)
 if name=='private_outputs':W.zero_();W[:r]=torch.eye(r,dtype=torch.float64)
 if name=='shared_left':L[:,1:]=L[:,0,None]
 if name=='signed_outputs':W[1]=-W[0];W[3]=W[2]+W[0]
 if name=='paired_squares':
  a=L.clone();b=R.clone();L=a+b;R=a-b
 T=materialize_mixed(L,R,W)
 l=(torch.randn(d,r,generator=g,dtype=torch.float64)/d**.5).requires_grad_();rr=(torch.randn(d,r,generator=g,dtype=torch.float64)/d**.5).requires_grad_();w=torch.randn(o,r,generator=g,dtype=torch.float64,requires_grad=True)
 implicit=mixed_coefficient_loss(T,l,rr,w);explicit=(materialize_mixed(l,rr,w)-T).square().sum()/T.square().sum()
 a=torch.autograd.grad(implicit,[l,rr,w],retain_graph=True);b=torch.autograd.grad(explicit,[l,rr,w]);grad=max(float((x-y).norm()/x.norm()) for x,y in zip(a,b));value=float(abs(implicit-explicit).detach());assert grad<1e-10 and value<1e-10
 opt=torch.optim.Adam([l,rr,w],lr=.05)
 for step in range(4000):
  opt.param_groups[0]['lr']=.05*.5*(1+math.cos(math.pi*step/4000));opt.zero_grad();loss=mixed_coefficient_loss(T,l,rr,w);loss.backward();opt.step()
 error=float(((materialize_mixed(l,rr,w)-T).norm()/T.norm()).detach());record=dict(structure=name,value_replay=value,gradient_replay=grad,random_start_error=error,recovered=error<1e-4);records.append(record);print(record,flush=True)
out=dict(records=records,all_instrument_checks_pass=True,all_fits_recovered=all(r['recovered'] for r in records),steps=4000,schedule='cosine',optimizer='Adam',learning_rate=.05,seconds=time.perf_counter()-start)
(P/'MIXED_QUADRATIC_PRODUCTS_TOYS_V1.json').write_text(json.dumps(out,indent=2)+'\n')
