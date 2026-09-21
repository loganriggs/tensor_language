"""Directional features for an affine-repaired quadratic product dictionary."""
import torch

def features(x,v,a,b):
 return (x@a.T)*(v@b.T)+(v@a.T)*(x@b.T)

def controls():
 gen=torch.Generator().manual_seed(943);rows=[]
 for d,k,n in [(3,5,13),(4,8,21),(6,9,24),(5,11,29),(7,13,33)]:
  rand=lambda *s:torch.randn(*s,generator=gen,dtype=torch.float64)
  x,v,a,b,W=rand(n,d),rand(n,d),rand(k,d),rand(k,d),rand(3,k)
  f=features(x,v,a,b);_,ref=torch.autograd.functional.jvp(lambda z:((z@a.T)*(z@b.T))@W.T,x,v);error=float((f@W.T-ref).norm()/ref.norm());ids=torch.arange(k-2);K=f.T@f/n;cross=W@K[:,ids];student=torch.linalg.solve(K[ids][:,ids],cross.T).T;explicit=torch.linalg.lstsq(f[:,ids],ref).solution.T
  solve=float((f[:,ids]@(student-explicit).T).norm()/ref.norm());rows.append(dict(derivative_error=error,normal_vs_direct_lstsq_error=solve))
 assert max(max(r.values()) for r in rows)<1e-10
 return rows
if __name__=='__main__':print(controls())
