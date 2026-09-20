import json,math,time,itertools
from pathlib import Path
import torch
from core import Model,metric,inner
from sweep import target_cases
from quartic_sparse_basis import gauge,atoms,refits

def main():
 torch.set_num_threads(1);p=Path(__file__).resolve().parent;out=p/'QUARTIC_BASIS_SWEEP_V1.json';assert not out.exists();targets=[]
 toy=json.loads((p/'TOY_SWEEP_V1.json').read_text());i=min((i for i,x in enumerate(toy['records']) if x['case']=='shared_quartic_dag' and x['width']==3),key=lambda i:toy['records'][i]['relative_error']);state=torch.load(p/'TOY_SWEEP_V1.pt',weights_only=True,map_location='cpu')[i]['state'];target=next(x['target'] for x in target_cases() if x['name']=='shared_quartic_dag');targets.append(('planted_shared',6,3,state,target,[2,3,4,6]))
 native=json.loads((p/'NATIVE_QUARTIC_PILOT_V1.json').read_text());saved=torch.load(p/'NATIVE_QUARTIC_PILOT_V1.pt',weights_only=True,map_location='cpu');i=min((i for i,x in enumerate(native['records']) if x['kind']=='dag' and x['width']==4 and x['metric']=='frobenius'),key=lambda i:native['records'][i]['relative_error']);targets.append(('native_shared',5,4,saved['students'][i],saved['target'],[2,4,6,8,10]))
 rows=[];baselines=[];start=time.perf_counter()
 for name,d,k,state,target,budgets in targets:
  bank,root=gauge(state,torch.zeros(k,k,dtype=torch.float64),d);reference=root@atoms(bank,d);M=metric(d,4);baselines.append(dict(target=name,refits=refits(bank,root,target,d,budgets)))
  for optname,lr,seed in itertools.product(['adam','muon'],[.005,.05],[0,1,2]):
   torch.manual_seed(seed);K=torch.nn.Parameter(torch.randn(k,k,dtype=torch.float64)*.05);opt=torch.optim.Adam([K],lr=lr) if optname=='adam' else torch.optim.Muon([K],lr=lr,weight_decay=0.,adjust_lr_fn='match_rms_adamw');best=float('inf');savedK=None;history=[]
   for step in range(3000):
    opt.zero_grad();bank,root=gauge(state,K,d);loss=root.norm(dim=0).sum()
    if float(loss.detach())<best:best=float(loss.detach());savedK=K.detach().clone()
    if step%300==0:history.append(dict(step=step,group_norm=float(loss.detach())))
    loss.backward();opt.step()
    for group in opt.param_groups:group['lr']=lr*(.01+.99*.5*(1+math.cos(math.pi*(step+1)/3000)))
   with torch.no_grad():
    bank,root=gauge(state,savedK,d);delta=root@atoms(bank,d)-reference;preserve=float((inner(delta,delta,M)/inner(reference,reference,M)).sqrt());assert preserve<1e-8
    row=dict(target=name,optimizer=optname,lr=lr,seed=seed,preservation_error=preserve,condition=float(torch.linalg.cond(torch.matrix_exp(savedK))),group_norm=best,bank=bank.tolist(),root=root.tolist(),refits=refits(bank,root,target,d,budgets),history=history);rows.append(row)
   out.write_text(json.dumps(dict(records=rows,baselines=baselines,seconds=time.perf_counter()-start,scope='24gauge-only searches; quadratic-bank span fixed, shared root-pair group sparsity, exact coefficient refits. Planted and firstnativequartic context only.'),indent=2)+'\n');print(name,optname,lr,seed,[(x['products'],x['error']) for x in row['refits']],flush=True)
if __name__=='__main__':main()
