"""Compact joint bilinear product fitting and planted optimizer comparison."""
import torch,time,json
from pathlib import Path
P=Path(__file__).resolve().parent

def tensor(factors):
 W,A,B=factors
 return torch.einsum('gr,ir,jr->gij',W,A,B)

def fit(target,rank,optimizer,lr,seed,steps):
 gen=torch.Generator().manual_seed(seed);shape=target.shape;factors=[torch.randn(d,rank,generator=gen,dtype=target.dtype) for d in shape]
 for i in range(3):factors[i]=factors[i]/factors[i].norm(dim=0)
 factors[0]/=rank**.5;factors=[x.requires_grad_() for x in factors]
 cls=torch.optim.Adam if optimizer=='adam' else torch.optim.Muon
 opt=cls(factors,lr=lr,weight_decay=0);best=float('inf');checkpoint=None
 for step in range(steps):
  opt.zero_grad();loss=(tensor(factors)-target).square().sum();value=float(loss.detach())
  if value<best:best=value;checkpoint=[x.detach().clone() for x in factors]
  loss.backward();opt.step()
 scaleprod=torch.stack([x.norm(dim=0) for x in checkpoint]).prod(0)
 return dict(error=best**.5,cancellation_ratio=float(scaleprod.sum()/target.norm()),factors=checkpoint)

def toys():
 torch.set_num_threads(2);torch.set_default_dtype(torch.float64);gen=torch.Generator().manual_seed(261114);rand=lambda *s:torch.randn(*s,generator=gen);cases={}
 cases['orthogonal']=(torch.eye(4)[:,:2],torch.eye(6)[:,:2],torch.eye(6)[:,:2])
 cases['dense_rank3']=(rand(4,3),rand(6,3),rand(6,3))
 cases['output_shared']=(rand(4,1).repeat(1,3),rand(6,3),rand(6,3))
 cases['signed_cancellation']=(rand(4,2),rand(6,2),rand(6,2));w,a,b=cases['signed_cancellation'];w[:,1]=-w[:,0];a[:,1]=a[:,0]+.2*a[:,1];b[:,1]=b[:,0]+.2*b[:,1]
 cases['dense_rank1']=(rand(4,1),rand(6,1),rand(6,1))
 records=[];start=time.perf_counter()
 for name,factors in cases.items():
  target=tensor(factors);target/=target.norm();rank=factors[0].shape[1]
  for opt in ['adam','muon']:
   for lr in [.003,.01]:
    for seed in [0,1]:
     r=fit(target,rank,opt,lr,261114+seed,600);records.append(dict(case=name,optimizer=opt,lr=lr,seed=seed,steps=600,error=r['error'],cancellation_ratio=r['cancellation_ratio']))
  print(name,'done',flush=True)
 scores={opt:sum(min(r['error'] for r in records if r['case']==case and r['optimizer']==opt) for case in cases)/len(cases) for opt in ['adam','muon']};winner=min(scores,key=scores.get);result=dict(records=records,mean_best_case_error=scores,selected_optimizer=winner,seconds=time.perf_counter()-start,scope='Five planted compact tensor structures; equal2rates*2seeds*600steps. Optimizer choice basedonlyontoys. Output-sharing example is deliberately non-identifiable; no component-recoveryclaim.')
 out=P/'JOINT_PRODUCT_TOYS_V1.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2));return result
if __name__=='__main__':toys()
