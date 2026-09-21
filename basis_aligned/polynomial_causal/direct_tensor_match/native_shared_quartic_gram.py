"""Joint four-output shared Gram search on saved native amplitude restrictions."""
import itertools,json,math,time
from pathlib import Path
import numpy as np,torch
from quartic_gram_search import operator,project
from shared_quartic_gram import search_shared,core_fit
P=Path(__file__).resolve().parent

def main():
 torch.set_num_threads(2);start=time.monotonic();cases=torch.load(P.parent.parent/'bilinear_quotient/circuits/followups/native_two_mlp_quartic_ht_v1r1.pt',weights_only=True,map_location='cpu');A,pairs=operator(5);tuples=list(itertools.combinations_with_replacement(range(5),4));rows=[]
 for ci,case in enumerate(cases):
  tensors=case['hessian_not_applicable_quartic'][0].double().numpy();target=np.array([[sum(H[t] for t in set(itertools.permutations(key)))/math.sqrt(len(set(itertools.permutations(key)))) for key in tuples] for H in tensors]);scale=np.linalg.norm(target);target/=scale
  canonical=np.stack([project(np.zeros((15,15)),A,t) for t in target]);_,V=np.linalg.eigh(np.einsum('vij,vkj->ik',canonical,canonical))
  for width in (4,6,8):
   initial,_=core_fit(A,target,V[:,-width:]);baseline=float(np.linalg.norm(initial.reshape(4,-1)@A.T-target));fits=[]
   for seed in (0,1):
    (error,U,coeff,K),n=search_shared(A,target,width,seed,1000);per=np.linalg.norm(K.reshape(4,-1)@A.T-target,axis=1)/np.linalg.norm(target,axis=1)
    # Actual two-stage graph vs its coefficient representation, random polynomial inputs.
    rng=np.random.default_rng(931);x=rng.normal(size=(128,5));z=np.stack([x[:,i]*x[:,j]*math.sqrt(1 if i==j else 2) for i,j in pairs],1);q=z@U;root=np.stack([q[:,i]*q[:,j]*(1 if i==j else math.sqrt(2)) for i in range(width) for j in range(i,width)],1);y=root@coeff.T;direct=np.einsum('ni,vij,nj->nv',z,K,z);replay=float(np.linalg.norm(y-direct)/np.linalg.norm(direct));assert replay<1e-10
    fits.append(dict(seed=seed,error=error,per_output_error=per.tolist(),iterations=n,graph_replay=replay))
   rows.append(dict(case=ci,width=width,initial_subspace_error=baseline,fits=fits,stored_coefficients=15*width+4*width*(width+1)//2,distinct_products=15+width*(width+1)//2))
 out=dict(records=rows,seconds=time.monotonic()-start,predictions=dict(width4_all_below_5pct=all(min(f['error'] for f in r['fits'])<.05 for r in rows if r['width']==4),width8_all_below_1pct=all(min(f['error'] for f in r['fits'])<.01 for r in rows if r['width']==8)),scope='All four outputs jointly in each of16fixed5inputcontexts. Native output geometry preserved. Not a single cross-context program or full1152input decomposition. Polynomial graph replay is not normalized native execution.')
 (P/'NATIVE_SHARED_QUARTIC_GRAM_V1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(seconds=out['seconds'],predictions=out['predictions'])))
if __name__=='__main__':main()
