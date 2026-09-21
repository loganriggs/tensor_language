"""Signed low-rank quartic Gram search by alternating affine/rank projections.
Heuristic only: no convergence/global-minimality guarantee. Uses repeated inputs.
"""
import itertools,math,json
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent

def operator(d):
 pairs=list(itertools.combinations_with_replacement(range(d),2));tuples=list(itertools.combinations_with_replacement(range(d),4));index={t:i for i,t in enumerate(tuples)};m=len(pairs);A=np.zeros((len(tuples),m*m))
 for i,a in enumerate(pairs):
  for j,b in enumerate(pairs):
   t=tuple(sorted(a+b));mult=math.factorial(4)
   for k in set(t):mult//=math.factorial(t.count(k))
   A[index[t],i*m+j]=math.sqrt(1 if a[0]==a[1] else 2)*math.sqrt(1 if b[0]==b[1] else 2)/math.sqrt(mult)
 return A,pairs

def project(K,A,target):
 return (K.ravel()+A.T@((target-A@K.ravel())/(A*A).sum(1))).reshape(K.shape)

def truncate(K,r):
 w,V=np.linalg.eigh((K+K.T)/2);idx=np.argsort(abs(w))[-r:];return (V[:,idx]*w[idx])@V[:,idx].T

def search(A,target,r,seed,steps=1000):
 m=round(math.sqrt(A.shape[1]));rng=np.random.default_rng(seed);canonical=project(np.zeros((m,m)),A,target)
 K=canonical.copy()
 if seed:
  noise=rng.normal(size=(m,m));noise=(noise+noise.T)/2;K+=project(noise,A,np.zeros_like(target))*np.linalg.norm(canonical)/max(np.linalg.norm(noise),1e-30)
 best=(float('inf'),K)
 for step in range(steps):
  low=truncate(K,r);err=np.linalg.norm(A@low.ravel()-target)/np.linalg.norm(target)
  if err<best[0]:best=(err,low.copy())
  if err<1e-10:break
  K=project(low,A,target)
 return best[0],best[1],step+1

def main():
 rng=np.random.default_rng(926);d=5;A,pairs=operator(d);m=len(pairs)
 q=rng.normal(size=m);q/=np.linalg.norm(q);r=rng.normal(size=m);r/=np.linalg.norm(r);t=rng.normal(size=m);t/=np.linalg.norm(t)
 radial=np.array([float(i==j) for i,j in pairs]);outer=lambda x,y:(np.outer(x,y)+np.outer(y,x))/2
 cases={'radial_square':(np.outer(radial,radial),1),'rotated_dense_square':(np.outer(q,q),1),'signed_two_squares':(np.outer(q,q)-np.outer(r,r),2),'quadratic_product':(outer(q,r),2),'shared_factor_sum':(outer(q,r)+outer(q,t),2)}
 rows=[]
 for name,(K,rank) in cases.items():
  target=A@K.ravel();canonical=project(np.zeros_like(K),A,target)
  # Function replay independently uses polynomial evaluation, not A.
  x=rng.normal(size=(100,d));z=np.stack([x[:,i]*x[:,j]*math.sqrt(1 if i==j else 2) for i,j in pairs],1)
  replay=np.linalg.norm(np.einsum('ni,ij,nj->n',z,K-canonical,z))/np.linalg.norm(np.einsum('ni,ij,nj->n',z,K,z));assert replay<1e-12
  fits=[]
  for seed in range(4):
   err,fit,steps=search(A,target,rank,seed)
   fits.append(dict(seed=seed,coefficient_error=err,steps=steps,function_error=float(np.linalg.norm(np.einsum('ni,ij,nj->n',z,fit-K,z))/np.linalg.norm(np.einsum('ni,ij,nj->n',z,K,z)))))
  rows.append(dict(case=name,planted_rank=rank,canonical_rank=int(np.linalg.matrix_rank(canonical)),canonical_replay=float(replay),fits=fits))
 out=dict(rows=rows,predictions=dict(exact_replay=all(r['canonical_replay']<1e-12 for r in rows),all_families_recovered=all(min(f['coefficient_error'] for f in r['fits'])<1e-6 for r in rows)),scope='Five known scalar quartic structures, d5, four starts, at most1000iterations. Planted root ranks supplied. Leaf quadratic forms may be dense; low Gram rank does not establish cheap leaf computation or feature semantics.')
 (P/'QUARTIC_GRAM_SEARCH_TOYS_V1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
