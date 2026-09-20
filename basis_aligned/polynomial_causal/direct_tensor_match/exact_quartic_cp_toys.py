"""Exact quartic CP variable projection, planted rank-two teachers at varying dimensions."""
import json,itertools,time,math
from pathlib import Path
import torch
from quartic_cp import cp_gram,cp_inner
P=Path(__file__).resolve().parent

def main():
 torch.set_num_threads(1);torch.set_default_dtype(torch.float64);out=P/'EXACT_QUARTIC_CP_TOYS_V1.json';assert not out.exists();start=time.perf_counter();records=[]
 for d in [6,32,128,1152]:
  torch.manual_seed(1644);teacher=[torch.randn(2,d) for _ in range(4)];teacher=[v/v.norm(dim=1,keepdim=True) for v in teacher];tc=torch.randn(3,2);energy=cp_inner(tc,teacher,tc,teacher);tc/=energy.sqrt()
  for optname,seed in itertools.product(['adam','muon'],[0,1]):
   torch.manual_seed(seed);raw=[torch.nn.Parameter(torch.randn(2,d)) for _ in range(4)];opt=torch.optim.Adam(raw,lr=.05) if optname=='adam' else torch.optim.Muon(raw,lr=.05,weight_decay=0.,adjust_lr_fn='match_rms_adamw');history=[];initial_score=None;t0=time.perf_counter()
   for step in range(600):
    opt.zero_grad();f=[v/v.norm(dim=1,keepdim=True) for v in raw];K=cp_gram(f,f);cross=tc@cp_gram(teacher,f);ridge=1e-10*K.diag().mean();C=torch.linalg.solve(K+ridge*torch.eye(2),cross.T).T
    # Minimize the actual residual up to its constant teacher norm. Scale by initial explained energy.
    residual_without_constant=((C@K)*C).sum()-2*(cross*C).sum()
    if initial_score is None:initial_score=max(float((-residual_without_constant).detach()),1e-30)
    loss=residual_without_constant/initial_score;assert bool(torch.isfinite(loss))
    if step%100==0:history.append(dict(step=step,explained_fraction=float((-residual_without_constant).detach())))
    loss.backward();opt.step()
    for group in opt.param_groups:group['lr']=.05*(.01+.99*.5*(1+math.cos(math.pi*(step+1)/600)))
   with torch.no_grad():
    f=[v/v.norm(dim=1,keepdim=True) for v in raw];K=cp_gram(f,f);cross=tc@cp_gram(teacher,f);C=torch.linalg.solve(K+1e-10*K.diag().mean()*torch.eye(2),cross.T).T;error2=float(1+cp_inner(C,f,C,f)-2*cp_inner(tc,teacher,C,f));error=math.sqrt(max(0,error2))
   row=dict(dimension=d,optimizer=optname,seed=seed,relative_frobenius_error=error,initial_explained_fraction=initial_score,steps=600,seconds=time.perf_counter()-t0,history=history);records.append(row);print(d,optname,seed,error,flush=True)
   out.write_text(json.dumps(dict(records=records,seconds=time.perf_counter()-start,scope='16 exact-gradient rank-two quartic CP fits with output elimination, dimensions6/32/128/1152, two restarts. Exact teacher norm; objective scaled by initial explained energy; no coefficient sampling.'),indent=2)+'\n')
if __name__=='__main__':main()
