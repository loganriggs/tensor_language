"""Independent dense Hessian / finite-difference controls plus native HVP price."""
from pathlib import Path
import time,json
import numpy as np
import torch
from cubic_secant_coordinates_v1 import encode,components,features
from cubic_projection_curvature_v1 import Curvature
from cubic_secant_block_v1 import capture as secant_capture
from shared_cubic_source_projection_v1 import capture as raw_capture
P=Path(__file__).resolve().parent

def main():
 torch.set_num_threads(2);torch.set_default_dtype(torch.float64);torch.manual_seed(9122051)
 out=P/'CUBIC_PROJECTION_CURVATURE_V1_CONTROL.json';assert not out.exists()
 atoms=torch.randn(2,3,4);z,t=encode(atoms,(0,1));target=torch.randn(64,2)
 def objective(x):
  bank=features(x,t).T;q,_=torch.linalg.qr(bank,mode='reduced');r=q@(q.T@target)-target;return r.square().sum()/target.square().sum()
 adapter=Curvature(objective,z.shape);x=z.numpy().ravel();d=np.random.default_rng(812).normal(size=x.shape);d/=np.linalg.norm(d)
 hv=adapter.hessp(x,d);dense=torch.autograd.functional.hessian(lambda y:objective(y.reshape(z.shape)),z.flatten()).numpy()@d
 eps=1e-5;fd=(adapter.fun(x+eps*d)[1]-adapter.fun(x-eps*d)[1])/(2*eps)
 errors={'dense_hvp':float(np.linalg.norm(hv-dense)/np.linalg.norm(dense)),'gradient_fd_hvp':float(np.linalg.norm(hv-fd)/np.linalg.norm(fd))}
 # Reuse old native diagnostic construction, not a new checkpoint / dataset.
 from folded_producer_cubic_weights_v1 import weights as load_weights
 from folded_normalized_router_v1 import rotary
 bind=json.loads((P/'FOLDED_PRODUCER_CUBIC_NATIVE_V1_BINDING.json').read_text())['files']
 sd=torch.load(next(k for k in bind if k.endswith('pytorch_model.bin')),weights_only=True,mmap=True,map_location='cpu')
 readers=torch.load(P/'MIXED_TOKEN_HEAD2_NATIVE_V1_PROGRAM.pt',weights_only=True)['current_readers'];(q1,k1,q2,k2,v,o),_,_=load_weights(sd,readers,'cpu')
 r=rotary(8,128).T@rotary(7,128);w=(q1,torch.cat([torch.einsum('ab,hbd->had',r,k1),torch.zeros_like(k1)],-1),q2,torch.cat([torch.einsum('ab,hbd->had',r,k2),torch.zeros_like(k2)],-1),v,o)
 saved=torch.load(P/'FOLDED_PRODUCER_CUBIC_NATIVE_V1_ARTIFACT.pt',weights_only=True);rows=[]
 for arm,pair in [(0,(5,15)),(1,(4,5))]:
  a=saved['atoms'][arm];theta,sep=encode(a,pair)
  with torch.no_grad():reference=float(raw_capture(a,*w))
  def obj(x):
   cc,mm=components(x,sep);return -secant_capture(cc,mm,w)/reference
  ad=Curvature(obj,theta.shape);xx=theta.numpy().ravel();start=time.perf_counter();value,grad=ad.fun(xx);gradient_seconds=time.perf_counter()-start
  direction=grad/np.linalg.norm(grad);start=time.perf_counter();hv=ad.hessp(xx,direction);hvp_seconds=time.perf_counter()-start
  # directional derivative of gradient provides independent native curvature check.
  ep=1e-5;numeric=(ad.fun(xx+ep*direction)[1]-ad.fun(xx-ep*direction)[1])/(2*ep)
  error=float(np.linalg.norm(hv-numeric)/max(np.linalg.norm(numeric),1e-30))
  rows.append(dict(arm=arm,objective=value,initial_equivalence_error=abs(value+1),gradient_norm=float(np.linalg.norm(grad)),gradient_seconds=gradient_seconds,hvp_seconds=hvp_seconds,hvp_fd_error=error,rayleigh=float(direction@hv),nonlinear_parameters=xx.size))
  print(json.dumps(rows[-1]),flush=True)
 result=dict(pred_a=max(errors.values())<=1e-6 and all(r['hvp_fd_error']<=1e-4 and r['initial_equivalence_error']<=1e-7 for r in rows),dense_controls=errors,native=rows,scope='Exact Hessian-vector product of reduced coefficient objective in fixed-separation secant coordinates, CPU price on frozen failed fits. No native optimization, recovery, or causal improvement claimed. Negative curvature is a local diagnostic only.')
 out.write_text(json.dumps(result,indent=2)+'\n');assert result['pred_a']
if __name__=='__main__':main()
