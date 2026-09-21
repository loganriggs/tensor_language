"""Joint signed Gram representatives with one shared quadratic feature subspace."""
import json,time
from pathlib import Path
import numpy as np
from quartic_gram_search import operator,project
P=Path(__file__).resolve().parent

def core_fit(A,target,U):
 r=U.shape[1];pairs=[(i,j) for i in range(r) for j in range(i,r)]
 atoms=np.stack([(np.outer(U[:,i],U[:,j])+np.outer(U[:,j],U[:,i]))/(2 if i==j else 2**.5) for i,j in pairs])
 design=A@atoms.reshape(len(atoms),-1).T
 coeff=np.linalg.lstsq(design,target.T,rcond=1e-12)[0].T
 K=np.einsum('vk,kij->vij',coeff,atoms)
 return K,coeff

def search_shared(A,target,width,seed,steps=1000):
 m=round(A.shape[1]**.5);rng=np.random.default_rng(seed)
 K=np.stack([project(np.zeros((m,m)),A,t) for t in target]);norm=np.linalg.norm(target)
 if seed:
  noise=rng.normal(size=K.shape);noise=(noise+noise.transpose(0,2,1))/2
  K+=np.stack([project(n,A,np.zeros_like(target[0])) for n in noise])*np.linalg.norm(K)/np.linalg.norm(noise)
 best=(float('inf'),None,None,None)
 for step in range(steps):
  _,V=np.linalg.eigh(np.einsum('vij,vkj->ik',K,K));U=V[:,-width:]
  low,coeff=core_fit(A,target,U);error=np.linalg.norm(low.reshape(len(target),-1)@A.T-target)/norm
  if error<best[0]:best=(float(error),U.copy(),coeff.copy(),low.copy())
  if error<1e-9:break
  K=np.stack([project(k,A,t) for k,t in zip(low,target)])
 return best,step+1

def main():
 A,_=operator(5);rng=np.random.default_rng(927);rows=[];start=time.monotonic()
 for case,width in [('shared_squares',2),('signed_mixed',2),('dense_core',3),('common_factor',3),('four_features',4)]:
  U=np.linalg.qr(rng.normal(size=(15,width)))[0];core=rng.normal(size=(4,width,width));core=(core+core.transpose(0,2,1))/2
  if case=='shared_squares':core=np.array([np.diag(rng.normal(size=width)) for _ in range(4)])
  if case=='common_factor':core[:,1:,1:]=0
  K=np.einsum('ip,vpq,jq->vij',U,core,U);target=K.reshape(4,-1)@A.T
  oracle,_=core_fit(A,target,U);oracle_error=np.linalg.norm(oracle-K)/np.linalg.norm(K);assert oracle_error<1e-10
  fits=[]
  for seed in range(3):
   result,n=search_shared(A,target,width,seed,1000);fits.append(dict(seed=seed,error=result[0],iterations=n))
  rows.append(dict(case=case,width=width,oracle_replay=float(oracle_error),fits=fits))
 out=dict(rows=rows,seconds=time.monotonic()-start,predictions=dict(all_recovered=all(min(f['error'] for f in row['fits'])<1e-6 for row in rows)),scope='Shared multi-output quadratic dictionary. Rank/width supplied. No feature identity guarantee; conditional core solves exact, subspace update heuristic.')
 (P/'SHARED_QUARTIC_GRAM_TOYS_V1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
