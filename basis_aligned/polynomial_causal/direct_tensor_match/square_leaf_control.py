"""Square-leaf structural hypothesis on teachers with certified capacity."""
import itertools,json,math,time
from pathlib import Path
import torch
from shared_bank_toys import teacher_for
from shared_quadratic_bank import bank_gram,native_bank_cross,bank_entries
from implicit_quartic import entries
P=Path(__file__).resolve().parent

def fit(teacher,optimizer,lr,seed,steps,initial=None):
 torch.manual_seed(seed);parameters=[torch.nn.Parameter(x.clone()) for x in (initial if initial is not None else torch.randn(6,1,6)/6**.5)]
 opt=torch.optim.Adam(parameters,lr=lr) if optimizer=='adam' else torch.optim.Muon(parameters,lr=lr,weight_decay=0.,adjust_lr_fn='match_rms_adamw')
 best=float('inf');base=None
 for step in range(steps+1):
  opt.zero_grad();raw=torch.stack(parameters);U=raw/raw.square().sum((1,2),keepdim=True).sqrt();G=bank_gram(U,U);cross=native_bank_cross(teacher,U,U);C=torch.linalg.solve(G+1e-8*G.diag().mean()*torch.eye(len(G)),cross.T).T;loss=((C@G)*C).sum()-2*(cross*C).sum();error=float((1+loss).detach())
  if error<best:best=error;bu=U.detach().clone();bc=C.detach().clone();beststep=step
  if base is None:base=max(float(-loss.detach()),1e-20);initial_error=math.sqrt(max(0,error))
  if step==steps:break
  (loss/base).backward();opt.step()
  for group in opt.param_groups:group['lr']=lr*(.01+.99*.5*(1+math.cos(math.pi*(step+1)/steps)))
 return dict(error=math.sqrt(max(0,best)),signed_squared_error=best,initial_error=initial_error,selected_step=beststep),bu,bc

def main():
 torch.set_num_threads(1);torch.set_default_dtype(torch.float64);out=P/'SQUARE_LEAF_CONTROL_V1.json';assert not out.exists();start=time.perf_counter();rows=[];witness=[];idx=torch.cartesian_prod(*[torch.arange(6)]*4)
 for family in ['coordinate','rotated_signed']:
  teacher,U,V=teacher_for(family);target=entries(*teacher,idx);result,u,c=fit(teacher,'adam',.005,0,0,U.reshape(6,1,6));witness.append(dict(family=family,error=float((bank_entries(u,u,idx)@c.T-target).norm()/target.norm())))
  for optimizer,lr,seed in itertools.product(['adam','muon'],[.005,.05],[0,1]):
   result,u,c=fit(teacher,optimizer,lr,seed,600);dense_error=float((bank_entries(u,u,idx)@c.T-target).norm()/target.norm());rows.append(dict(family=family,optimizer=optimizer,lr=lr,seed=seed,enumerated_error=dense_error,input_parameters=36,output_parameters=63,**result));print(family,optimizer,lr,seed,dense_error,flush=True)
 predictions=dict(pred_d_square_capacity=all(r['error']<1e-6 for r in witness),pred_e_coordinate_rescue=min(r['enumerated_error'] for r in rows if r['family']=='coordinate')<.01,pred_f_signed_preserved=min(r['enumerated_error'] for r in rows if r['family']=='rotated_signed')<.01)
 out.write_text(json.dumps(dict(records=rows,witnesses=witness,predictions=predictions,seconds=time.perf_counter()-start,scope='Known-capacity square-leaf hypothesis for two planted families; no generic positivity/native circuit claim.'),indent=2)+'\n');print(predictions,flush=True)
if __name__=='__main__':main()
