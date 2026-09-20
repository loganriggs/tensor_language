import json,time,math
from pathlib import Path
import torch
from exact_cp_fit import fit
from quartic_cp import cp_inner,cp_gram
P=Path(__file__).resolve().parent

def refine(tc,teacher,initial,lr):
 den=cp_inner(tc,teacher,tc,teacher);raw=[torch.nn.Parameter(a.clone()) for a in initial];opt=torch.optim.Adam(raw,lr=lr);base=None;best=float('inf');history=[]
 for step in range(301):
  opt.zero_grad();f=[a/a.norm(dim=1,keepdim=True) for a in raw];K=cp_gram(f,f);cross=tc@cp_gram(teacher,f);C=torch.linalg.solve(K+1e-10*K.diag().mean()*torch.eye(2),cross.T).T;loss=((C@K)*C).sum()-2*(cross*C).sum();error=float(((den+loss)/den).detach());best=min(best,error)
  if step==0:base=max(float(-loss.detach()),1e-30);initial_error=error
  if step%25==0:history.append(dict(step=step,signed_squared_relative_error=error))
  if step==300:break
  (loss/base).backward();opt.step()
  for g in opt.param_groups:g['lr']=lr*(.01+.99*.5*(1+math.cos(math.pi*(step+1)/300)))
 return dict(lr=lr,initial_error=math.sqrt(max(0,initial_error)),final_error=math.sqrt(max(0,error)),best_error=math.sqrt(max(0,best)),best_signed_squared_error=best,history=history)

def main():
 torch.set_num_threads(1);torch.set_default_dtype(torch.float64);torch.manual_seed(1658);pairs=[torch.linalg.qr(torch.randn(1152,2))[0].T for _ in range(4)];out=P/'REFINEMENT_RATE_V1.json';assert not out.exists();rows=[];start=time.perf_counter()
 for rho in [0.,.5,.95]:
  teacher=[torch.stack([q[0],rho*q[0]+(1-rho*rho)**.5*q[1]]) for q in pairs];tc=torch.tensor([[1.,0.],[0.,.5],[0.,0.]]);tc/=cp_inner(tc,teacher,tc,teacher).sqrt();first,c1,f1=fit(tc,teacher,1,'adam',0,600);rc=torch.cat([tc,-c1],1);rf=[torch.cat([a,b],0) for a,b in zip(teacher,f1)];second,c2,f2=fit(rc,rf,1,'adam',100,600);initial=[torch.cat([a,b],0) for a,b in zip(f1,f2)]
  for lr in [.05,.005,.0005]:
   row=refine(tc,teacher,initial,lr);row['linear_overlap']=rho;rows.append(row);print(rho,lr,row['initial_error'],row['final_error'],row['best_error'],flush=True)
  out.write_text(json.dumps(dict(records=rows,seconds=time.perf_counter()-start,scope='Fixed600+600 residual initialization;9 refinements. Initial/best/final training errors; no heldout selection, one teacher orientation and initialization per condition.'),indent=2)+'\n')
if __name__=='__main__':main()
