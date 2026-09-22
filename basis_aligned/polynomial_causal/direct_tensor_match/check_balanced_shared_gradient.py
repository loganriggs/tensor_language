"""Output-metric transformation against direct weighted autograd, five structures."""
import json
from pathlib import Path
import torch
from shared_training_gradient import backward
from shared_quadratic_bank import normalize_bank
from sparse_quartic_bank import support,gram as cg,native_cross as cx
from shared_gaussian_moments import gram as gg,native_cross as gx
from noncentral_gaussian_cp import project_shifted
P=Path(__file__).resolve().parent

def main():
 torch.set_num_threads(2);rows=[];dtype=torch.float64
 for seed,family in enumerate(['independent','shared_input','shared_output','squares','cancellation']):
  torch.manual_seed(13000+seed);u=torch.randn(4,2,3,dtype=dtype);v=torch.randn_like(u)
  if family=='shared_input':u[1]=u[0]
  if family=='squares':v=u.clone()
  if family=='cancellation':u[1]=u[0];v[1]=-v[0]
  params=[u.flatten(0,1).requires_grad_(),v.flatten(0,1).requires_grad_()];U,V=normalize_bank(*[p.reshape(4,2,3) for p in params]);pairs=support(4,7,12)
  teacher=[torch.randn(*shape,dtype=dtype) for shape in [(2,3),(3,3),(3,3),(3,4),(4,3),(4,3)]]
  if family=='shared_output':teacher[0][1]=teacher[0][0]
  S=torch.eye(3,dtype=dtype);mu=torch.randn(3,dtype=dtype);proj=project_shifted(teacher,mu);lam=17.;w=torch.tensor([.02,1.98],dtype=dtype);sw=w.sqrt();ridge=.01
  G=(gg(U,V,pairs,U@mu,V@mu)+lam*cg(U,V,pairs))/(1+lam);X=(gx(teacher,mu,proj,U,V,pairs,U@mu,V@mu)+lam*cx(teacher,U,V,pairs))/(1+lam)
  C=torch.linalg.solve(G.detach()+ridge*torch.eye(7,dtype=dtype),X.detach().T).T
  loss=(G*(C.T@(w[:,None]*C))).sum()-2*(w[:,None]*C*X).sum();ref=torch.autograd.grad(loss,params)
  tw=[teacher[0]*sw[:,None]]+teacher[1:];pw=tuple(t*sw.reshape((-1,)+(1,)*(t.ndim-1)) for t in proj)
  independent=project_shifted(tw,mu);perr=max(float((a-b).norm()/b.norm()) for a,b in zip(pw,independent));assert perr<1e-12
  value=backward(params,4,2,pairs,C*sw[:,None],tw,tw,mu,pw,S,mu,lam,chunk=2)
  a=torch.cat([p.grad.flatten() for p in params]);b=torch.cat([g.flatten() for g in ref]);err=float((a-b).norm()/b.norm());assert err<1e-10
  vreplay=abs(value-float(loss.detach()))/(1+abs(float(loss.detach())));assert vreplay<1e-10
  rows.append(dict(family=family,gradient_error=err,projection_error=perr,value_error=vreplay))
 (P/'BALANCED_SHARED_GRADIENT_CONTROLS_V1.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(rows))
if __name__=='__main__':main()
