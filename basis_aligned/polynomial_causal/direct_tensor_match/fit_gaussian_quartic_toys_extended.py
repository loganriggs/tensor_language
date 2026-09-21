"""Longer-budget diagnostic for the two failed Gaussian planted cases.
Original seed0, Adam .05,2000steps; direct small Hermite-tensor recovery audit.
"""
from pathlib import Path
import torch,json,time,math
from quartic_pair_metric import numerator_forms
from gaussian_quartic_lowrank import prepare_teacher,squared_error_by_degree
from gaussian_quartic_moment import features
import itertools
P=Path(__file__).resolve().parent;torch.set_num_threads(2);start=time.perf_counter()
cases=torch.load(P/'QUARTIC_NUMERATOR_TOY_FITS_V1.pt',weights_only=True);records=[]
for name,case in cases.items():
 if name not in ["opposite_forms","separated_scale"]:continue
 Qa,Qb=case['Qa'],case['Qb'];d=len(Qa);A,B=numerator_forms(Qa,Qb,.7,-.4);teacher=prepare_teacher(A,B)
 zero=torch.zeros_like(Qa);fixed=numerator_forms(zero,zero,.7,-.4);parts=[]
 for M in fixed:
  ev,Q=torch.linalg.eigh(M);ids=ev.abs().argsort(descending=True)[:2];parts.append((Q[:,ids],ev[ids]))
 for seed in range(1):
  gen=torch.Generator().manual_seed(22500+seed);U=torch.nn.Parameter(torch.randn(d,2,dtype=torch.float64,generator=gen)/d**.5);V=torch.nn.Parameter(torch.randn(d,2,dtype=torch.float64,generator=gen)/d**.5)
  opt=torch.optim.Adam([U,V],lr=.05);best=float('inf');best_factors=None
  for step in range(2001):
   factors=[]
   for Z,sign,factor,(X,w) in zip([U,V],[case['signA'],case['signB']],[-.5,1.],parts):
    padded=torch.cat([Z,Z.new_zeros(3,2)],0);factors.extend([torch.cat([padded,X],1),torch.cat([factor*sign,w])])
   loss=squared_error_by_degree(teacher,*factors).sum()/teacher['variance'];value=float(loss.detach());
   if value<best:best=value;best_factors=(U.detach().clone(),V.detach().clone())
   if step==2000:break
   opt.zero_grad();loss.backward();opt.step()
   for group in opt.param_groups:group['lr']=.05*(.05+.95*.5*(1+math.cos(math.pi*(step+1)/2000)))
  bu,bv=best_factors
  ca=(bu*case['signA'])@bu.T;cb=(bv*case['signB'])@bv.T
  C,D=numerator_forms(ca,cb,.7,-.4)
  ft=features(A,B);fs=features(C,D)
  def explicit4(M,N):
   raw=torch.einsum('ij,kl->ijkl',M,N)
   return sum(raw.permute(order) for order in itertools.permutations(range(4)))/24
  def explicit3(terms):
   return sum((torch.einsum('ij,k->ijk',M,v)+torch.einsum('ik,j->ijk',M,v)+torch.einsum('jk,i->ijk',M,v))/3 for M,v in terms)
  direct=24*(explicit4(*ft['quartic'])-explicit4(*fs['quartic'])).square().sum()+6*(explicit3(ft['cubic'])-explicit3(fs['cubic'])).square().sum()+2*(ft['quadratic']-fs['quadratic']).square().sum()+(ft['linear']-fs['linear']).square().sum()+(ft['constant']-fs['constant']).square()
  records.append(dict(structure=name,seed=seed,relative_gaussian_error=float((direct/teacher['variance']).sqrt()),subtracted_objective_report=max(best,0.)**.5))
 print(name,[r['relative_gaussian_error'] for r in records if r['structure']==name],flush=True)
recovered=sum(min(r['relative_gaussian_error'] for r in records if r['structure']==name)<1e-3 for name in ["opposite_forms","separated_scale"])
out=dict(predictions=dict(pred_a_recovery=recovered==2),records=records,recovered_structures=recovered,seconds=time.perf_counter()-start,scope='Only the two failed planted structures; original seed0 and Adam.05 restarted at2000steps. Extended-budget diagnostic, not independent optimizer selection, exact Gaussian numerator objective. Numerical nearzero loss subtraction not an exact-recovery certificate.')
(P/'GAUSSIAN_QUARTIC_TOY_EXTENDED_V1.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k!='records'},indent=2))
