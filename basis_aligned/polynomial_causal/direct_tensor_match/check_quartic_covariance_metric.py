"""Independent dense/gradient/quadrature checks for covariance quartic objectives."""
import itertools,json
from pathlib import Path
import numpy as np
import torch
from implicit_quartic import entries
from shared_quadratic_bank import bank_entries,bank_gram,native_bank_cross
P=Path(__file__).resolve().parent

def transform(H,L):return torch.einsum('vijkl,ia,jb,kc,le->vabce',H,L,L,L,L)
def wick_norm(H):
 trace=torch.einsum('vijkk->vij',H);double=torch.einsum('vii->v',trace)
 return 24*H.square().sum()+72*trace.square().sum()+9*double.square().sum()

def main():
 torch.set_num_threads(1);torch.set_default_dtype(torch.float64);torch.manual_seed(1839)
 d=3;idx=torch.cartesian_prod(*[torch.arange(d)]*4);teacher=[torch.randn(*shape) for shape in [(2,3),(3,2),(3,2),(2,4),(4,d),(4,d)]];U=torch.randn(2,2,d,requires_grad=True);V=torch.randn(2,2,d,requires_grad=True);C=torch.randn(2,3);H=entries(*teacher,idx).T.reshape(2,d,d,d,d);student=(bank_entries(U,V,idx)@C.T).T.reshape_as(H);delta=H-student;A=torch.randn(d,d);M=A@A.T+.1*torch.eye(d);L=torch.linalg.cholesky(M)
 dense=transform(delta,L).square().sum();t=[*teacher[:-2],teacher[-2]@L,teacher[-1]@L];u=U@L;v=V@L;selfnorm=((C.T@C)*bank_gram(u,v)).sum();cross=(native_bank_cross(t,u,v)*C).sum();implicit=transform(H,L).square().sum()+selfnorm-2*cross;error=float(((implicit-dense).abs()/dense).detach());g1=torch.autograd.grad(implicit,[U,V],retain_graph=True);g2=torch.autograd.grad(dense,[U,V],retain_graph=True);grad=max(float((a-b).norm()/b.norm()) for a,b in zip(g1,g2));assert max(error,grad)<1e-11
 nodes,weights=np.polynomial.hermite.hermgauss(5);grid=torch.tensor(list(itertools.product(range(5),repeat=d)));z=torch.tensor(nodes)[grid]*2**.5;w=torch.tensor(weights/np.sqrt(np.pi))[grid].prod(1);x=z@L.T;values=torch.einsum('vijkl,ni,nj,nk,nl->nv',delta,x,x,x,x);quadrature=(w[:,None]*values.square()).sum();wick=wick_norm(transform(delta,L));we=float(((wick-quadrature).abs()/quadrature).detach());assert we<1e-11
 # Same covariance, radically different eighth moments: f=x0^4-x1^4.
 radial=torch.zeros(1,2,2,2,2);radial[0,0,0,0,0]=1;radial[0,1,1,1,1]=-1;rademacher=torch.tensor(list(itertools.product([-1.,1.],repeat=2)));empirical=torch.einsum('vijkl,ni,nj,nk,nl->nv',radial,rademacher,rademacher,rademacher,rademacher).square().mean();gaussian=wick_norm(radial);assert float(empirical)==0 and float(gaussian)==192
 hidden=torch.zeros(1,2,2,2,2);hidden[0,1,1,1,1]=1;floors=[]
 for epsilon in [0.,.001,.01,.1,1.]:
  root=torch.diag(torch.tensor([1.,epsilon**.5]));value=float(transform(hidden,root).square().sum());assert abs(value-epsilon**4)<1e-12;floors.append(dict(eigenvalue_floor=epsilon,weighted_coefficient_energy=value,unweighted_energy=1.))
 out=dict(coefficient_value_relative_error=error,coefficient_gradient_relative_error=grad,wick_quadrature_relative_error=we,same_covariance_example=dict(coefficient_norm_squared=2.,gaussian_function_norm_squared=float(gaussian),rademacher_function_norm_squared=float(empirical)),floor_blindspot=floors,scope='Weighted four-slot coefficient metric differs from Gaussian eighth-moment function metric and empirical loss. Exact dense/gradient/quadrature checks; singular covariance is blind to hidden directions.')
 (P/'QUARTIC_COVARIANCE_METRIC_CHECK_V1.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
if __name__=='__main__':main()
