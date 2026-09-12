from pathlib import Path
import time,json,numpy as np,torch
from cubic_cluster_coordinates_v1 import encode,components,reconstruct,normalized_reader_gradient
from cubic_secant_coordinates_v1 import features
from cubic_projection_curvature_v1 import Curvature
from cubic_secant_block_v1 import capture
from shared_cubic_source_projection_v1 import capture as raw_capture
from folded_producer_cubic_weights_v1 import weights as producer_weights
from folded_normalized_router_v1 import rotary
P=Path(__file__).resolve().parent

def main():
 torch.set_num_threads(2);torch.set_default_dtype(torch.float64);torch.manual_seed(9122038);out=P/'CUBIC_CLUSTER_CURVATURE_V1_CONTROL.json';assert not out.exists()
 a=torch.randn(4,3,5);theta,c=encode(a,[0,2,3]);target=torch.randn(125,2)
 def dense_obj(x,chart=None):
  if chart is None:bank=features(x).T
  else:
   aa,mm=components(x,chart);bank=(mm@features(aa)).T
  q,_=torch.linalg.qr(bank,mode='reduced');return (q@(q.T@target)-target).square().sum()
 z=theta.requires_grad_(True);g=torch.autograd.grad(dense_obj(z,c),z)[0];raw=reconstruct(theta.detach(),c).requires_grad_(True);gr=torch.autograd.grad(dense_obj(raw),raw)[0];n=raw.norm(dim=-1,keepdim=True);u=raw/n;ref=n*(gr-(gr*u).sum(-1,keepdim=True)*u);pullback_error=float((normalized_reader_gradient(z,g,c)-ref).norm()/ref.norm())
 old=json.loads((P/'FOLDED_PRODUCER_CUBIC_NATIVE_V1_BINDING.json').read_text())['files'];sd=torch.load(next(k for k in old if k.endswith('pytorch_model.bin')),weights_only=True,mmap=True,map_location='cpu');C=torch.load(P/'MIXED_TOKEN_HEAD2_NATIVE_V1_PROGRAM.pt',weights_only=True)['current_readers'];(q1,k1,q2,k2,v,o),_,_=producer_weights(sd,C,'cpu');r=rotary(8,128).T@rotary(7,128);w=(q1,torch.cat([torch.einsum('ab,hbd->had',r,k1),torch.zeros_like(k1)],-1),q2,torch.cat([torch.einsum('ab,hbd->had',r,k2),torch.zeros_like(k2)],-1),v,o)
 rows=[]
 for saved in torch.load(P/'CUBIC_CLUSTER_NATIVE_V1_CHARTS.pt',weights_only=True):
  theta,chart=saved['theta'],saved['chart'];cc,mm=components(theta,chart)
  with torch.no_grad():scale=float(capture(cc,mm,w))
  def objective(x):
   cc,mm=components(x,chart);return -capture(cc,mm,w)/scale
  adapter=Curvature(objective,theta.shape);x=theta.numpy().ravel();start=time.perf_counter();value,grad=adapter.fun(x);gs=time.perf_counter()-start;d=grad/np.linalg.norm(grad);start=time.perf_counter();h=adapter.hessp(x,d);hs=time.perf_counter()-start;eps=1e-5;fd=(adapter.fun(x+eps*d)[1]-adapter.fun(x-eps*d)[1])/(2*eps);err=float(np.linalg.norm(h-fd)/np.linalg.norm(fd));row=dict(arm=saved['arm'],hvp_fd_error=err,gradient_seconds=gs,hvp_seconds=hs,unit_reader_gradient=float(normalized_reader_gradient(theta,torch.from_numpy(grad).reshape(theta.shape),chart).norm()));rows.append(row);print(json.dumps(row),flush=True)
 result=dict(pred_a=pullback_error<=1e-10 and all(r['hvp_fd_error']<=1e-4 for r in rows),unit_reader_pullback_error=pullback_error,rows=rows,scope='Independent unit-reader gradient and native matrix-free Hessian-vector finite-difference checks. No continuation or circuit result.')
 out.write_text(json.dumps(result,indent=2)+'\n');assert result['pred_a']
if __name__=='__main__':main()
