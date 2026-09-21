"""Five planted structures under the exact Gaussian numerator moment loss.
Two random starts per case, Adam .05,400steps. Same known rank/sign capacity.
"""
from pathlib import Path
import torch,json,time,math
from quartic_pair_metric import numerator_forms
from gaussian_quartic_lowrank import prepare_teacher,squared_error_by_degree
P=Path(__file__).resolve().parent;torch.set_num_threads(2);start=time.perf_counter()
cases=torch.load(P/'QUARTIC_NUMERATOR_TOY_FITS_V1.pt',weights_only=True);records=[]
for name,case in cases.items():
 Qa,Qb=case['Qa'],case['Qb'];d=len(Qa);A,B=numerator_forms(Qa,Qb,.7,-.4);teacher=prepare_teacher(A,B)
 zero=torch.zeros_like(Qa);fixed=numerator_forms(zero,zero,.7,-.4);parts=[]
 for M in fixed:
  ev,Q=torch.linalg.eigh(M);ids=ev.abs().argsort(descending=True)[:2];parts.append((Q[:,ids],ev[ids]))
 for seed in range(2):
  gen=torch.Generator().manual_seed(22500+seed);U=torch.nn.Parameter(torch.randn(d,2,dtype=torch.float64,generator=gen)/d**.5);V=torch.nn.Parameter(torch.randn(d,2,dtype=torch.float64,generator=gen)/d**.5)
  opt=torch.optim.Adam([U,V],lr=.05);best=float('inf')
  for step in range(401):
   factors=[]
   for Z,sign,factor,(X,w) in zip([U,V],[case['signA'],case['signB']],[-.5,1.],parts):
    padded=torch.cat([Z,Z.new_zeros(3,2)],0);factors.extend([torch.cat([padded,X],1),torch.cat([factor*sign,w])])
   loss=squared_error_by_degree(teacher,*factors).sum()/teacher['variance'];value=float(loss.detach());best=min(best,value)
   if step==400:break
   opt.zero_grad();loss.backward();opt.step()
   for group in opt.param_groups:group['lr']=.05*(.05+.95*.5*(1+math.cos(math.pi*(step+1)/400)))
  records.append(dict(structure=name,seed=seed,relative_gaussian_error=max(best,0.)**.5))
 print(name,[r['relative_gaussian_error'] for r in records if r['structure']==name],flush=True)
recovered=sum(min(r['relative_gaussian_error'] for r in records if r['structure']==name)<1e-3 for name in cases)
out=dict(predictions=dict(pred_a_recovery=recovered>=4),records=records,recovered_structures=recovered,seconds=time.perf_counter()-start,scope='Five known rank2 source families, ten random-start Adam.05 fits,400steps, exact Gaussian numerator objective. Numerical nearzero loss subtraction not an exact-recovery certificate.')
(P/'GAUSSIAN_QUARTIC_TOY_FITS_V1.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k!='records'},indent=2))
