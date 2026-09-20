import json,itertools,math,time
from pathlib import Path
import torch
from core import Model,inner,metric
from sweep import target_cases
from sparse_basis import gauge,coefficients,sparse_refits

def main():
 torch.set_num_threads(1);p=Path(__file__).resolve().parent;out=p/'SPARSE_BASIS_SWEEP_V1.json';assert not out.exists()
 r=json.loads((p/'TOY_SWEEP_V1.json').read_text());i=min((i for i,x in enumerate(r['records']) if x['case']=='sparse_tucker'),key=lambda i:r['records'][i]['relative_error']);state=torch.load(p/'TOY_SWEEP_V1.pt',map_location='cpu',weights_only=True)[i]['state'];target=next(x['target'] for x in target_cases() if x['name']=='sparse_tucker')
 model=Model(6,3,'tucker',3,2);model.load_state_dict(state);reference=model().detach();M=metric(6,2)
 zero=torch.zeros(3,3,dtype=torch.float64);small=torch.zeros(2,2,dtype=torch.float64);P,W,G=gauge(state,zero,small);assert float((coefficients(P,W,G)-reference).norm())<1e-12
 baseline=sparse_refits(P,W,G,target);rows=[];start=time.perf_counter()
 for optname,lr,seed in itertools.product(['adam','muon'],[.005,.05],[0,1,2]):
  torch.manual_seed(seed);K=torch.nn.Parameter(torch.randn(3,3,dtype=torch.float64)*.05);J=torch.nn.Parameter(torch.randn(2,2,dtype=torch.float64)*.05)
  opt=torch.optim.Adam([K,J],lr=lr) if optname=='adam' else torch.optim.Muon([K,J],lr=lr,weight_decay=0.,adjust_lr_fn='match_rms_adamw')
  best=float('inf');saved=None;history=[]
  for step in range(3000):
   opt.zero_grad();P,W,G=gauge(state,K,J);loss=G.abs().sum()
   if float(loss.detach())<best:best=float(loss.detach());saved=(K.detach().clone(),J.detach().clone())
   if step%300==0:history.append(dict(step=step,core_l1=float(loss.detach())))
   loss.backward();opt.step()
   for group in opt.param_groups:group['lr']=lr*(.01+.99*.5*(1+math.cos(math.pi*(step+1)/3000)))
  with torch.no_grad():
   P,W,G=gauge(state,*saved);c=coefficients(P,W,G);delta=c-reference;error=float((inner(delta,delta,M)/inner(reference,reference,M)).sqrt());assert error<1e-8
   row=dict(optimizer=optname,lr=lr,seed=seed,core_l1=best,function_preservation_error=error,condition_input=float(torch.linalg.cond(torch.matrix_exp(saved[0]))),condition_output=float(torch.linalg.cond(torch.matrix_exp(saved[1]))),refits=sparse_refits(P,W,G,target),history=history);rows.append(row)
  out.write_text(json.dumps(dict(baseline=baseline,records=rows,seconds=time.perf_counter()-start,scope='Gauge-only sparse-basis search in the recovered Tucker function, no sampled inputs. Hard support selected by transformed core magnitude then exact weight-metric least-squares refit.'),indent=2)+'\n');print(optname,lr,seed,[(x['budget'],x['relative_gaussian_error']) for x in row['refits']],flush=True)
if __name__=='__main__':main()
