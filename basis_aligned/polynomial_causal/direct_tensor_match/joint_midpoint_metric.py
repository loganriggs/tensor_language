"""Tiny exact control: marginal covariances do not determine bilinear loss."""
import torch,json
from pathlib import Path

def check():
 x=torch.tensor([-3**.5,0.,3**.5],dtype=torch.float64);w=torch.tensor([1/6,2/3,1/6],dtype=torch.float64)
 # Both joint laws have the same zero mean and unit marginal variance.
 paired=(w*x.pow(4)).sum();independent=(w*x.square()).sum().square()
 assert abs(float(paired)-3)<1e-12 and abs(float(independent)-1)<1e-12
 gen=torch.Generator().manual_seed(261043)
 n=torch.randn(13,3,generator=gen,dtype=torch.float64);m=torch.randn(13,4,generator=gen,dtype=torch.float64);n[:,0]=m[:,0]
 delta=torch.randn(2,3,4,generator=gen,dtype=torch.float64);lift=torch.einsum('bi,bj->bij',n,m).flatten(1);moment=lift.T@lift/len(n)
 direct=torch.einsum('vij,bi,bj->bv',delta,n,m).square().sum()/len(n);flat=delta.flatten(1);contracted=(flat*(flat@moment)).sum()
 error=float((direct-contracted).abs()/direct);assert error<1e-12
 return dict(same_marginal_variance=float((w*x.square()).sum()),paired_product_energy=float(paired),independent_product_energy=float(independent),joint_lifted_metric_replay=error,scope='Exact counterexample to replacing a joint lifted moment with a Kronecker product of marginal second moments. Not a native model result.')
if __name__=='__main__':
 r=check();Path(__file__).with_name('JOINT_MIDPOINT_METRIC_ORACLE_V1.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
