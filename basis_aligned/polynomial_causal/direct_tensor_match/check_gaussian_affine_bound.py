"""Check the cubic orthogonal-sector identity and a tight rank bound."""
import json
from pathlib import Path
import torch
torch.set_num_threads(2);torch.manual_seed(545);rows=[]
def tensor(A,u):
 return (torch.einsum('ij,k->ijk',A,u)+torch.einsum('ik,j->ijk',A,u)+torch.einsum('jk,i->ijk',A,u))/3
for n in [5,7]:
 u=torch.randn(n,dtype=torch.float64);v=torch.randn(n,dtype=torch.float64)
 Q=torch.linalg.qr(torch.stack([u,v],1),mode='reduced').Q
 S=Q@Q.T;P=torch.eye(n,dtype=u.dtype)-S
 A=torch.randn(n,n,dtype=u.dtype);A=(A+A.T)/2
 B=torch.randn(n,n,dtype=u.dtype);B=(B+B.T)/2
 T=tensor(A,v)+tensor(B,u)
 sector=torch.einsum('ia,jb,kc,abc->ijk',P,P,S,T)
 three_sector_norm=3*sector.square().sum()
 Ap=P@A@P;Bp=P@B@P
 gram=(v@v)*Ap.square().sum()+(u@u)*Bp.square().sum()+2*(u@v)*(Ap*Bp).sum()
 err=float((three_sector_norm-gram/3).abs()/three_sector_norm)
 assert err<1e-12 and three_sector_norm<=T.square().sum()+1e-10
 rows.append(dict(dimension=n,sector_identity_relative_error=err))
# With u and v in separate context directions, B supported on source coords,
# and A=0, the B Schur tail bound is attained by spectral truncation.
n=6;r=2;u=torch.zeros(n,dtype=torch.float64);u[-1]=2
v=torch.zeros_like(u);v[-2]=1
B=torch.diag(torch.tensor([4.,3.,2.,1.,0.,0.],dtype=u.dtype))
student=torch.diag(torch.tensor([4.,3.,0.,0.,0.,0.],dtype=u.dtype))
actual=6*tensor(B-student,u).square().sum()
bound=2*(u@u)*torch.tensor([2.,1.],dtype=u.dtype).square().sum()
assert abs(float(actual-bound))<1e-10
out=dict(passed=True,records=rows,tight_example_squared_error=float(actual),tight_example_squared_bound=float(bound))
Path(__file__).with_name('GAUSSIAN_AFFINE_BOUND_CHECK_V1.json').write_text(json.dumps(out,indent=2)+'\n')
print(out)
