import json,time,math,itertools
from pathlib import Path
import torch
from quartic_cp import cp_gram
from gaussian_cp import gaussian_cp_gram
P=Path(__file__).resolve().parent

def main():
 torch.set_num_threads(1);torch.set_default_dtype(torch.float64);d=16;t=torch.eye(d)[:2];teacher=[t.clone() for _ in range(4)];tc=torch.diag(torch.tensor([1.,.5]));M=torch.eye(d);M[0,0]=.2;M[1,1]=2.;metrics={'frobenius':cp_gram,'isotropic':gaussian_cp_gram,'covariance':lambda a,b:gaussian_cp_gram(a,b,M)};norms={name:((tc.T@tc)*gram(teacher,teacher)).sum() for name,gram in metrics.items()};rows=[];baselines=[];out=P/'QUARTIC_METRIC_TRADEOFF_V1.json';assert not out.exists();start=time.perf_counter()
 def evaluate(c,f):
  result={}
  for name,g in metrics.items():
   e=float((norms[name]+((c.T@c)*g(f,f)).sum()-2*((tc.T@c)*g(teacher,f)).sum())/norms[name]);result[name]=math.sqrt(max(0,e))
  return result
 for metric,g in metrics.items():
  for coordinate in [0,1]:
   f=[a[coordinate:coordinate+1] for a in teacher];K=g(f,f);cross=tc@g(teacher,f);c=cross/K[0,0];baselines.append(dict(metric=metric,coordinate=coordinate,errors=evaluate(c,f)))
 for metric,lr,seed in itertools.product(metrics,[.01,.05],[0,1]):
  g=metrics[metric];torch.manual_seed(seed);raw=[torch.nn.Parameter(torch.randn(1,d)) for _ in range(4)];opt=torch.optim.Adam(raw,lr=lr);best=float('inf');history=[]
  for step in range(601):
   opt.zero_grad();f=[a/a.norm(dim=1,keepdim=True) for a in raw];K=g(f,f);cross=tc@g(teacher,f);c=cross/K[0,0];loss=((c@K)*c).sum()-2*(cross*c).sum();e=float(((norms[metric]+loss)/norms[metric]).detach())
   if e<best:best=e;best_step=step;best_c=c.detach().clone();best_f=[a.detach().clone() for a in f]
   if step%100==0:history.append(dict(step=step,signed_squared_error=e))
   if step==600:break
   (loss/norms[metric]).backward();opt.step()
   for group in opt.param_groups:group['lr']=lr*(.01+.99*.5*(1+math.cos(math.pi*(step+1)/600)))
  row=dict(metric=metric,lr=lr,seed=seed,selected_step=best_step,final_training_error=math.sqrt(max(0,e)),errors=evaluate(best_c,best_f),factors=[a.tolist() for a in best_f],C=best_c.tolist(),history=history);rows.append(row);print(metric,lr,seed,row['errors'],flush=True);out.write_text(json.dumps(dict(records=rows,baselines=baselines,seconds=time.perf_counter()-start,scope='Known rank2 polynomial underfit by rank1; exact Frobenius and zero-mean Gaussian metrics, covariance specified as variances. Not empirical eighth moments.'),indent=2)+'\n')
if __name__=='__main__':main()
