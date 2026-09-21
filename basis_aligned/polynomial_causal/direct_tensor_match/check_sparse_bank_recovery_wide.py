"""Five planted selected-pair hierarchies, exact native coefficient fitting."""
import json,math,time
from pathlib import Path
import torch
from sparse_quartic_bank import support,gram,native_cross,entries
from shared_quadratic_bank import normalize_bank
from quartic_cp_profile import profile
from quartic_cp import directional
P=Path(__file__).resolve().parent

def wider_support(old):
 all_pairs=torch.triu_indices(8,8)
 used=set(map(tuple,old.T.tolist()))
 extra=torch.tensor([v for v in all_pairs.T.tolist() if tuple(v) not in used],dtype=torch.long).T
 order=torch.randperm(extra.shape[1],generator=torch.Generator().manual_seed(10501))
 return torch.cat([old,extra[:,order[:17]]],1)

def main():
 torch.set_num_threads(2);rows=[];dtype=torch.float64;d=4;pairs=support(4,7,seed=45);idx=torch.cartesian_prod(*[torch.arange(d) for _ in range(4)]);eye=torch.eye(d,dtype=dtype)
 for s,family in enumerate(['independent','shared_inputs','shared_outputs','squares','cancellation']):
  torch.manual_seed(10300+s);left=torch.randn(4,d,dtype=dtype);right=torch.randn_like(left);out=torch.randn(2,7,dtype=dtype)
  if family=='shared_inputs':left[1]=left[0]
  if family=='shared_outputs':out=torch.randn(2,1,dtype=dtype)@torch.randn(1,7,dtype=dtype)
  if family=='squares':right=left.clone()
  if family=='cancellation':left[1]=left[0];right[1]=-right[0]
  teacher=[out,torch.eye(4,dtype=dtype)[pairs[0]],torch.eye(4,dtype=dtype)[pairs[1]],torch.eye(4,dtype=dtype),left,right]
  target=directional(*teacher,[eye[idx[:,i]] for i in range(4)]);energy=target.square().sum(); fit_pairs=wider_support(pairs)
  def objective(u,v):
   active=pairs if len(u)==4 else fit_pairs
   return profile(gram(u,v,active),native_cross(teacher,u,v,active),ridge=1e-8)
  u,v=normalize_bank(left[:,None,:],right[:,None,:]);_,c=objective(u,v);planted=float(((entries(u,v,pairs,idx)@c.T-target).norm()/target.norm()).detach());assert planted<1e-4
  for optimizer in ['adam','muon']:
   for restart in [0,1]:
    torch.manual_seed(10400+restart);params=[torch.nn.Parameter(torch.randn(8,d,dtype=dtype)/2) for _ in range(2)]
    opt=torch.optim.Adam(params,lr=.1) if optimizer=='adam' else torch.optim.Muon(params,lr=.1,weight_decay=0.,adjust_lr_fn='match_rms_adamw');best=None;start=time.monotonic()
    for step in range(401):
     u,v=normalize_bank(params[0][:,None,:],params[1][:,None,:]);loss,c=objective(u,v);score=float(loss.detach())
     if best is None or score<best[0]:
      with torch.no_grad():error=float((entries(u,v,fit_pairs,idx)@c.T-target).norm()/target.norm())
      best=(score,step,error)
     if step==400:break
     opt.zero_grad();(loss/energy).backward();opt.step()
     for group in opt.param_groups:group['lr']=.1*(.01+.99*.5*(1+math.cos(math.pi*(step+1)/400)))
    row=dict(family=family,optimizer=optimizer,restart=restart,planted_error=planted,error=best[2],step=best[1],seconds=time.monotonic()-start);rows.append(row);print(json.dumps(row),flush=True)
 summary={o:dict(below1percent=sum(r['error']<.01 for r in rows if r['optimizer']==o),median_error=float(torch.tensor([r['error'] for r in rows if r['optimizer']==o]).median())) for o in ['adam','muon']}
 result=dict(rows=rows,summary=summary,recovery_gate=any(v['below1percent']>=8 for v in summary.values()),scope='Widened8feature24pair student, same representable4feature7pair targets,400steps/.1rate/cosine1pct,2starts; different teachers than CP suite, not cross-family optimizer league table.')
 (P/'SPARSE_QUARTIC_BANK_RECOVERY_WIDE_V1.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
