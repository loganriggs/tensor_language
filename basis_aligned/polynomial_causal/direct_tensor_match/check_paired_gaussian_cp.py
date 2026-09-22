"""Independent paired quadrature and derivative checks, including gradients."""
import itertools,json
from pathlib import Path
import numpy as np
import torch
from paired_gaussian_cp import gram_degrees,cross_degrees,combine
from noncentral_gaussian_cp import project_shifted

def evaluate(t,x):
 C,l,r,D,L,R=t;h=((x@L.T)*(x@R.T))@D.T;return ((h@l.T)*(h@r.T))@C.T

def controls():
 torch.set_num_threads(2);dt=torch.float64;nodes,weights=np.polynomial.hermite.hermgauss(5);ix=torch.tensor(list(itertools.product(range(5),repeat=4)));pts=torch.tensor(nodes*2**.5,dtype=dt)[ix];w=torch.tensor(weights/np.sqrt(np.pi),dtype=dt)[ix].prod(1);z=pts[:,:2];eps=pts[:,2:];rows=[]
 for seed in range(5):
  torch.manual_seed(33000+seed);teacher=[torch.randn(*s,dtype=dt) for s in [(2,3),(3,3),(3,3),(3,4),(4,2),(4,2)]];mu=torch.zeros(2,dtype=dt) if seed==0 else torch.randn(2,dtype=dt);S=torch.tensor([[1.,.2],[.1,.7]],dtype=dt)
  if seed==2:teacher[5]=teacher[4].clone()
  if seed==3:teacher[0][1]=teacher[0][0]
  if seed==4:teacher[4][1]=teacher[4][0];teacher[5][1]=-teacher[5][0]
  f=[torch.randn(3,2,dtype=dt,requires_grad=True) for _ in range(4)];fw=[a@S for a in f];bias=[a@mu for a in f];trans=teacher[:4]+[teacher[4]@S,teacher[5]@S];loc=torch.linalg.solve(S,mu);proj=project_shifted(trans,loc);G=gram_degrees(fw,bias);X=cross_degrees(trans,loc,proj,fw,bias);C=torch.randn(2,3,dtype=dt)
  phi=lambda x:torch.stack([x@a.T for a in f]).prod(0)
  for rho in [0.,.5,.9,1.]:
   g=combine(G,rho);cross=combine(X,rho)
   if rho<1:
    x=z@S.T+mu;y=(rho*z+(1-rho*rho)**.5*eps)@S.T+mu;dp=phi(x)-phi(y);dy=evaluate(teacher,x)-evaluate(teacher,y);gref=dp.T@(w[:,None]*dp)/(2*(1-rho));xref=dy.T@(w[:,None]*dp)/(2*(1-rho))
   else:
    zz=z.detach().clone().requires_grad_();x=zz@S.T+mu;p=phi(x);yy=evaluate(teacher,x);jp=torch.stack([torch.autograd.grad(p[:,i].sum(),zz,create_graph=True,retain_graph=True)[0] for i in range(3)],dim=1);jy=torch.stack([torch.autograd.grad(yy[:,i].sum(),zz,create_graph=True,retain_graph=True)[0] for i in range(2)],dim=1);gref=torch.einsum('n,nid,njd->ij',w,jp,jp);xref=torch.einsum('n,nid,njd->ij',w,jy,jp)
   loss=(C@g*C).sum()-2*(C*cross).sum();direct=(C@gref*C).sum()-2*(C*xref).sum();ga=torch.autograd.grad(loss,f,retain_graph=True);gb=torch.autograd.grad(direct,f,retain_graph=True);rel=lambda a,b:float(((a-b).norm()/b.norm().clamp_min(1e-30)).detach());row=dict(seed=seed,rho=rho,gram=rel(g,gref),cross=rel(cross,xref),gradient=rel(torch.cat([a.flatten() for a in ga]),torch.cat([a.flatten() for a in gb])));assert max(row[k] for k in ['gram','cross','gradient'])<1e-9,row;rows.append(row)
 zero=[torch.zeros_like(a) for a in fw];one=[torch.ones_like(a) for a in bias];constant=gram_degrees(zero,one);assert torch.count_nonzero(constant[1:])==0
 return rows
if __name__=='__main__':
 rows=controls();Path(__file__).with_name('PAIRED_GAUSSIAN_CP_CONTROLS_V1.json').write_text(json.dumps(dict(rows=rows,constant_invisibility=True),indent=2)+'\n');print('20 paired/derivative quadrature and gradient checks passed',max(max(r[k] for k in ['gram','cross','gradient']) for r in rows))
