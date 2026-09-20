import json,time
from pathlib import Path
import torch
from core import Model,metric,normrows,packed_quadratic
from sweep import target_cases,fit
from symmetric_quadratic_als import sweep
P=Path(__file__).resolve().parent

def main():
 torch.set_num_threads(1);out=P/'QUADRATIC_ALS_TOYS_V1.json';assert not out.exists();start=time.perf_counter();records=[]
 for case in [x for x in target_cases() if x['degree']==2]:
  width={'coordinate_sparse':4,'bilinear_cp':3,'sparse_tucker':4}[case['name']];target=case['target'];chol=torch.linalg.cholesky(metric(6,2,'frobenius'));den=(target@chol).square().sum()
  for seed in range(3):
   torch.manual_seed(seed);m=Model(6,3,'cp',width);C=m.weight.detach();A=normrows(m.left.detach());B=normrows(m.right.detach());history=[];last=float('inf');violation=0.;t0=time.perf_counter()
   for iteration in range(500):
    C,A,B=sweep(C,A,B,target,chol);loss=float((((C@packed_quadratic(A,B)-target)@chol).square().sum()/den));violation=max(violation,loss-last);last=loss
    if iteration%25==0:history.append(dict(iteration=iteration,relative_error=loss**.5))
    if loss<1e-12:break
   assert violation<1e-8
   records.append(dict(case=case['name'],width=width,seed=seed,optimizer='als_svd',relative_error=last**.5,iterations=iteration+1,monotonicity_violation=violation,seconds=time.perf_counter()-t0,history=history,factors=dict(C=C.tolist(),A=A.tolist(),B=B.tolist())))
   for optimizer in ['adam','muon']:
    r,_=fit(target,6,2,'cp',width,optimizer,.05,seed,1200,metric_kind='frobenius');r.update(case=case['name']);records.append(r)
   print(case['name'],seed,[(r['optimizer'],r['relative_error']) for r in records[-3:]],flush=True)
   out.write_text(json.dumps(dict(records=records,seconds=time.perf_counter()-start,scope='27 paired initialization symmetric quadratic CP fits; exact SVD block solves versus gradient methods. Dense toy coefficient designs only, not scalable native implementation.'),indent=2)+'\n')
if __name__=='__main__':main()
