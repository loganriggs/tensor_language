import itertools,json
from pathlib import Path
import torch,numpy as np
from core import covariance_metric,metric,evaluate,terms

def main():
 torch.manual_seed(274);torch.set_default_dtype(torch.float64);torch.set_num_threads(1);reports=[]
 for n in [2,4]:
  d=3;a=torch.randn(d,d);cov=a@a.T+.1*torch.eye(d);M=covariance_metric(d,n,cov);c=torch.randn(2,len(terms(d,n)))
  nodes,w=np.polynomial.hermite.hermgauss(n+1);nodes*=2**.5;w/=np.pi**.5
  grid=list(itertools.product(range(n+1),repeat=d));z=torch.tensor([[nodes[i] for i in t] for t in grid]);weight=torch.tensor([np.prod([w[i] for i in t]) for t in grid]);x=z@torch.linalg.cholesky(cov).T
  direct=(evaluate(c,x,n).square().sum(-1)*weight).sum();closed=((c@M)*c).sum();err=float((direct-closed).abs()/closed)
  isotropic=float((covariance_metric(d,n,torch.eye(d))-metric(d,n)).abs().max());assert err<1e-12 and isotropic==0
  reports.append(dict(degree=n,quadrature_relative_error=err,isotropic_replay=isotropic,minimum_metric_eigenvalue=float(torch.linalg.eigvalsh(M)[0])))
 # Nullspace redteam: covariance with zero variance cannot constrain that coordinate.
 S=torch.diag(torch.tensor([1.,0.]));M=covariance_metric(2,2,S);c=torch.tensor([[0.,0.,1.]])
 assert ((c@M)*c).sum()==0 and ((c@metric(2,2))*c).sum()>0
 return dict(records=reports,singular_covariance_blindspot_verified=True)
if __name__=='__main__':
 r=main();Path(__file__).with_name('COVARIANCE_CHECK_V1.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
