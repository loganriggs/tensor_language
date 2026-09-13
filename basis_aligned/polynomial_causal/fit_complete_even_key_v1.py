"""Ten-start complete numerator fit, rank48 and fixed coefficient objective."""
from pathlib import Path
import json,time,torch
from complete_even_key_objective_v1 import loss
P=Path(__file__).resolve().parent

def fit(initial,g,normalization,tolerance=1e-7,max_steps=3000):
 u=torch.linalg.qr(initial,mode='reduced').Q;history=[]
 def value_grad(u):
  p=(u@u.T).detach().requires_grad_(True);value=loss(p,*g)/normalization
  grad=torch.autograd.grad(value,p)[0];gu=(grad+grad.T)@u
  return value.detach(),gu-u@(u.T@gu)
 for iteration in range(max_steps):
  value,tangent=value_grad(u);stationarity=float(tangent.norm());history.append(float(value))
  if stationarity<=tolerance:break
  step=10.
  for backtrack in range(40):
   candidate=torch.linalg.qr(u-step*tangent,mode='reduced').Q
   nextvalue=loss(candidate@candidate.T,*g)/normalization
   if nextvalue<=value-1e-4*step*tangent.square().sum():break
   step/=2
  else:break
  u=candidate
 value,tangent=value_grad(u)
 return u,dict(normalized_loss=float(value),iterations=iteration+1,stationarity=float(tangent.norm()),converged=float(tangent.norm())<=tolerance,monotone=all(b<=a+1e-14 for a,b in zip(history,history[1:])))

def main():
 torch.set_num_threads(2);torch.manual_seed(7131136);tic=time.perf_counter();saved=torch.load(P/'COMPLETE_EVEN_KEY_V1_GRAMS.pt',weights_only=True);g=saved['grams'];normalization=saved['normalization'];prior=saved['queryfold_basis'];rows=[];best=None
 baseline=float(loss(prior@prior.T,*g)/normalization)
 for start in range(10):
  initial=[saved['original_basis'],prior,saved['outside_basis']][start] if start<3 else prior+.2*torch.randn_like(prior) if start<5 else torch.randn_like(prior)
  u,r=fit(initial,g,normalization);r['start']=start;rows.append(r)
  if best is None or r['normalized_loss']<best[0]:best=(r['normalized_loss'],u)
  (P/'COMPLETE_EVEN_KEY_FIT_V1_PROGRESS.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(r),flush=True)
 improvement=1-best[0]/baseline
 result={'pred_a':all(r['monotone'] for r in rows),'pred_b':all(r['converged'] for r in rows),'pred_c':improvement>=.01,'prior_queryfold_normalized_loss':baseline,'best_normalized_loss':best[0],'relative_improvement_over_queryfold':improvement,'rows':rows,'seconds':time.perf_counter()-tic,'scope':'Complete unnormalized even-key replacement coefficient objective atfourpositions, rank48. Local multistart; not denominator-weighted behavior or global certificate.'}
 torch.save({'basis_rotation':best[1]},P/'COMPLETE_EVEN_KEY_FIT_V1_PROGRAM.pt');(P/'COMPLETE_EVEN_KEY_FIT_V1_RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
