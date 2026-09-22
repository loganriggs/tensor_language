"""Independent Gauss-Hermite check of mean and varying quadratic error."""
import itertools,json
from pathlib import Path
import numpy as np
import torch

def main():
 torch.set_num_threads(2)
 gen=torch.Generator().manual_seed(28001)
 nodes,weights=np.polynomial.hermite.hermgauss(3)
 ids=list(itertools.product(range(3),repeat=3))
 x=torch.tensor([[nodes[i]*2**.5 for i in ix] for ix in ids],dtype=torch.float64)
 w=torch.tensor([np.prod([weights[i]/np.sqrt(np.pi) for i in ix]) for ix in ids],dtype=torch.float64)
 rows=[]
 for kind in ['identity','traceless','dense','rank_one','signed_outputs']:
  a=torch.randn(4,3,3,generator=gen,dtype=torch.float64);a=(a+a.transpose(-1,-2))/2
  if kind=='identity':a=torch.eye(3,dtype=torch.float64).expand(4,-1,-1).clone()
  if kind=='traceless':a-=torch.diag_embed(a.diagonal(dim1=-2,dim2=-1).sum(1)[:,None].expand(-1,3)/3)
  if kind=='rank_one':
   v=torch.randn(4,3,generator=gen,dtype=torch.float64);a=v[:,:,None]*v[:,None,:]
  if kind=='signed_outputs':a[1]=-a[0];a[2]=2*a[0];a[3]=-2*a[0]
  mean=a.diagonal(dim1=-2,dim2=-1).sum(1)
  y=torch.einsum('ni,vij,nj->nv',x,a,x)
  actual=(w[:,None]*y.square()).sum(0)
  formula=2*a.square().sum((1,2))+mean.square()
  varying=(w[:,None]*(y-mean).square()).sum(0)
  variance=2*a.square().sum((1,2))
  err=float((actual-formula).abs().max()/formula.max())
  verr=float((varying-variance).abs().max()/variance.max())
  assert max(err,verr)<1e-12
  rows.append(dict(kind=kind,total_energy_relative_discrepancy=err,centered_energy_relative_discrepancy=verr))
 p=Path(__file__).parent/'QUADRATIC_METRIC_CONTROLS_V1.json';p.write_text(json.dumps(dict(rows=rows,quadrature='3 points per dimension; exact through coordinate degree 5; quadratic squared has degree 4'),indent=2)+'\n')
 print(rows)
if __name__=='__main__':main()
