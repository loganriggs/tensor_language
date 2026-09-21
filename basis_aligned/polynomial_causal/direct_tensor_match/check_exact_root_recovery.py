"""Exact-objective optimizer controls on five planted three-feature quartics."""
import json,math
from pathlib import Path
import torch
from exact_root_tensor_objective import objective
from shared_quadratic_bank import normalize_bank,bank_entries
from quartic_cp import directional
P=Path(__file__).resolve().parent

def main():
 torch.set_num_threads(2);dtype=torch.float64;rows=[]
 for seed,family in enumerate(['independent','shared_inputs','shared_outputs','squares','cancellation']):
  torch.manual_seed(760+seed);d=4;left=torch.randn(3,d,dtype=dtype);right=torch.randn_like(left);c=torch.randn(2,4,dtype=dtype);a=torch.randn(4,3,dtype=dtype);b=torch.randn_like(a)
  if family=='shared_inputs':left[1]=left[0]
  if family=='shared_outputs':c=torch.randn(2,1,dtype=dtype)@torch.randn(1,4,dtype=dtype)
  if family=='squares':right=left.clone()
  if family=='cancellation':left[2]=left[0];right[2]=-right[0]
  teacher=[c,a,b,torch.eye(3,dtype=dtype),left,right];idx=torch.cartesian_prod(*[torch.arange(d) for _ in range(4)]);eye=torch.eye(d,dtype=dtype);h=directional(*teacher,[eye[idx[:,i]] for i in range(4)]);energy=h.square().sum()
  def error(u,v,coeff):return float(((bank_entries(u,v,idx)@coeff.T-h).norm()/h.norm()).detach())
  with torch.no_grad():
   u,v=normalize_bank(left[:,None,:],right[:,None,:]);_,coeff,_,_=objective(teacher,u,v,ridge=1e-8);warm=error(u,v,coeff)
  assert warm<1e-4
  for optimizer in ['adam','muon']:
   for restart in [0,1]:
    torch.manual_seed(8000+restart);params=[torch.nn.Parameter(torch.randn(3,d,dtype=dtype)/2) for _ in range(2)]
    opt=torch.optim.Adam(params,lr=.01) if optimizer=='adam' else torch.optim.Muon(params,lr=.01,weight_decay=0.,adjust_lr_fn='match_rms_adamw')
    best=None
    for step in range(101):
     u,v=normalize_bank(params[0][:,None,:],params[1][:,None,:]);loss,coeff,_,_=objective(teacher,u,v,ridge=1e-8);score=float(loss.detach())
     if best is None or score<best[0]:best=(score,step,error(u,v,coeff))
     if step==100:break
     opt.zero_grad();(loss/energy).backward();opt.step()
    rows.append(dict(family=family,optimizer=optimizer,restart=restart,steps=100,rate=.01,planted_error=warm,selected_step=best[1],exact_coefficient_error=best[2]))
 result=dict(rows=rows,summary={o:dict(below1percent=sum(r['exact_coefficient_error']<.01 for r in rows if r['optimizer']==o),median_error=float(torch.tensor([r['exact_coefficient_error'] for r in rows if r['optimizer']==o]).median())) for o in ['adam','muon']},scope='Bounded100step2restart optimizer control, exactobjective. Native25stepAdam already queued; this audit is not retrospective native optimizerselection or proof of convergence.')
 (P/'EXACT_ROOT_RECOVERY_TOYS_V1.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
