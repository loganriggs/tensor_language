import json,time
from pathlib import Path
import torch
from exact_cp_fit import fit
from quartic_cp import cp_inner,cp_gram
P=Path(__file__).resolve().parent

def perturb(factors,seed):
 torch.manual_seed(seed);return [torch.nn.functional.normalize(a+.1*torch.randn_like(a)/a.shape[1]**.5,dim=1) for a in factors]

def main():
 torch.set_num_threads(1);torch.set_default_dtype(torch.float64);torch.manual_seed(1658);pairs=[torch.linalg.qr(torch.randn(1152,2))[0].T for _ in range(4)];out=P/'RESIDUAL_OBSTRUCTION_V1.json';assert not out.exists();rows=[];start=time.perf_counter()
 for rho in [0.,.5,.95]:
  teacher=[torch.stack([q[0],rho*q[0]+(1-rho*rho)**.5*q[1]]) for q in pairs];tc=torch.tensor([[1.,0.],[0.,.5],[0.,0.]]);tc/=cp_inner(tc,teacher,tc,teacher).sqrt()
  first,c1,f1=fit(tc,teacher,1,'adam',0,600);rc=torch.cat([tc,-c1],1);rf=[torch.cat([a,b],0) for a,b in zip(teacher,f1)];gram=rc@cp_gram(rf,rf)@rc.T;ev=torch.linalg.eigvalsh(gram);bound=float(((ev.sum()-ev[-1])/ev.sum()).clamp_min(0).sqrt());second,c2,f2=fit(rc,rf,1,'adam',100,600);initial=[torch.cat([a,b],0) for a,b in zip(f1,f2)];runs=[]
  for name,init,steps in [('joint300',initial,300),('joint1500',initial,1500),('perturbed_joint1500',perturb(initial,1702),1500),('oracle_perturbed300',perturb(teacher,1702),300)]:
   row,c,f=fit(tc,teacher,2,'adam',0,steps,initial=init);row.update(method=name,steps=steps);runs.append(row)
  record=dict(linear_overlap=rho,amplitude_ratio=.5,first=first,second=second,residual_output_eigenvalues=ev.tolist(),residual_rank_one_output_lower_bound=bound,runs=runs);rows.append(record);print(rho,'residual lower bound',bound,'second error',second['relative_error'],'refinements',[(r['method'],r['relative_error']) for r in runs],flush=True);out.write_text(json.dumps(dict(records=rows,seconds=time.perf_counter()-start,scope='Rank-one output relaxation is a lower bound only. Longer/oracle controls not equal-budget native algorithms.'),indent=2)+'\n')
if __name__=='__main__':main()
