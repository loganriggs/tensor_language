"""Signed Gram search on saved five-input native restrictions, no model forwards."""
import json,itertools,math,time
import numpy as np,torch
from pathlib import Path
from quartic_gram_search import operator,project,search,truncate
P=Path(__file__).resolve().parent

def main():
 torch.set_num_threads(2);start=time.monotonic();cases=torch.load(P.parent.parent/'bilinear_quotient/circuits/followups/native_two_mlp_quartic_ht_v1r1.pt',weights_only=True,map_location='cpu');A,pairs=operator(5);tuples=list(itertools.combinations_with_replacement(range(5),4));rows=[]
 for ci,case in enumerate(cases):
  for output,H in enumerate(case['hessian_not_applicable_quartic'][0].double().numpy()):
   target=np.array([sum(H[t] for t in set(itertools.permutations(key)))/math.sqrt(len(set(itertools.permutations(key)))) for key in tuples]);target/=np.linalg.norm(target);canonical=project(np.zeros((15,15)),A,target)
   for rank in (2,4,8):
    fixed=truncate(canonical,rank);baseline=float(np.linalg.norm(A@fixed.ravel()-target));fits=[]
    for seed in (0,1):
     error,K,steps=search(A,target,rank,seed,1000);fits.append(dict(seed=seed,error=error,steps=steps))
    rows.append(dict(case=ci,output=output,rank=rank,fixed_representative_error=baseline,fits=fits))
 out=dict(records=rows,seconds=time.monotonic()-start,predictions=dict(rank4_all_below_5pct=all(min(x['error'] for x in r['fits'])<.05 for r in rows if r['rank']==4),rank8_all_below_1e6=all(min(x['error'] for x in r['fits'])<1e-6 for r in rows if r['rank']==8)),scope='16 saved contexts, first row,4 scalar outputs each,5 amplitude inputs. Independent scalar fits; no output sharing, no feature identity, no full1152input or normalizedmodel adoption. Leaf costs remain dense. Fixed-representative truncation is an initialization baseline, not an optimized Tucker competitor.')
 (P/'NATIVE_QUARTIC_GRAM_SEARCH_V1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(predictions=out['predictions'],seconds=out['seconds'])))
if __name__=='__main__':main()
