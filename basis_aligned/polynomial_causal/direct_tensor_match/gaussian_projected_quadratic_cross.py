"""Matrix-free coefficient cross with the Gaussian-projected native quadratic.

A small number of student products is allowed; never expand the native quartic.
Covariance M is both the reference Gaussian law and the quadratic coefficient metric.
"""
import torch
from centered_quadratic_factors import fold
from quadratic_student_fit import cross

class ProjectedQuadratic:
 def __init__(self,teacher,mu,M):
  self.teacher=teacher;self.M=M;self.L=torch.linalg.cholesky(M)
  C,l,r,D,A,B=teacher
  _,_,ct,at,bt=fold(teacher,mu)
  hm=D@((A@M)*B).sum(1);extra=(C@((r@hm)[:,None]*l+(l@hm)[:,None]*r))@D
  self.C=torch.cat([ct,extra],1);self.A=torch.cat([at,A],0)@self.L;self.B=torch.cat([bt,B],0)@self.L
 def bilinear_covariance_term(self,u,v):
  C,l,r,D,A,B=self.teacher
  def J(z):return D@((A@z)[:,None]*B+(B@z)[:,None]*A)
  ju,jv=J(u),J(v)
  return .5*C@(((l@ju)@self.M*(r@jv)).sum(1)+((l@jv)@self.M*(r@ju)).sum(1))
 def cross(self,a,b):
  base=cross(self.C,self.A,self.B,a@self.L,b@self.L)
  return base+torch.stack([self.bilinear_covariance_term(u,v) for u,v in zip(a@self.M,b@self.M)],1)

def check():
 import json
 from pathlib import Path
 from gaussian_low_degree_projection import project
 from implicit_quartic import entries
 torch.set_num_threads(1);torch.set_default_dtype(torch.float64);torch.manual_seed(2012);teacher=[torch.randn(*s) for s in [(2,3),(3,5),(3,5),(5,6),(6,4),(6,4)]];mu=torch.randn(4);raw=torch.randn(4,4);M=raw@raw.T+.3*torch.eye(4);idx=torch.cartesian_prod(*[torch.arange(4)]*4);H=entries(*teacher,idx).T.reshape(2,4,4,4,4);_,_,Q,_=project(H,mu,M);operator=ProjectedQuadratic(teacher,mu,M);a=torch.randn(3,4,requires_grad=True);b=torch.randn(3,4,requires_grad=True);got=operator.cross(a,b);reference=torch.einsum('ki,vij,kj->vk',a@M,Q,b@M);value=float(((got-reference).norm()/reference.norm()).detach());g1=torch.autograd.grad(got.square().sum(),[a,b],retain_graph=True);g2=torch.autograd.grad(reference.square().sum(),[a,b]);gradient=max(float((x-y).norm()/y.norm()) for x,y in zip(g1,g2));assert max(value,gradient)<1e-12
 result=dict(cross_relative_error=value,gradient_relative_error=gradient,scope='Implicit cross with exact degree2 Gaussian projection of a native-format quartic; dense toy coefficient and gradient oracle. Native1152dim execution notyetperformed.');(Path(__file__).resolve().parent/'GAUSSIAN_PROJECTED_QUADRATIC_CROSS_CHECK_V1.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
if __name__=='__main__':check()
