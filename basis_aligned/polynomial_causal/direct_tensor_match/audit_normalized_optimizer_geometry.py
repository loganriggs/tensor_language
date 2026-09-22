"""Direction-recovery control for raw parameters normalized row-wise in the model."""
import json,time,inspect,hashlib
import torch
import torch.optim._muon as implementation
from audit_conditional_residual_accounting import P

def main():
 torch.set_num_threads(2);start=time.monotonic();rows=[]
 for shape in [(4,4),(96,1152)]:
  for seed in [25001,25002]:
   generator=torch.Generator().manual_seed(seed);initial=torch.randn(*shape,dtype=torch.float64,generator=generator);target=torch.nn.functional.normalize(torch.randn(*shape,dtype=torch.float64,generator=generator),dim=1)
   for optimizer in ['adam','muon_original','muon_match_rms']:
    param=initial.clone().requires_grad_();rate=.1 if optimizer=='adam' else .01
    opt=torch.optim.Adam([param],lr=rate) if optimizer=='adam' else torch.optim.Muon([param],lr=rate,adjust_lr_fn=None if optimizer=='muon_original' else 'match_rms_adamw')
    history=[]
    for step in range(251):
     unit=torch.nn.functional.normalize(param,dim=1);loss=(unit-target).square().sum(1).mean()/2
     if step in [0,1,25,250]:
      cosine=(unit*target).sum(1).clamp(-1,1);initialunit=torch.nn.functional.normalize(initial,dim=1);movement=(unit*initialunit).sum(1).clamp(-1,1)
      history.append(dict(step=step,loss=float(loss.detach()),median_target_angle_degrees=float(torch.rad2deg(torch.acos(cosine.detach())).median()),median_rotation_degrees=float(torch.rad2deg(torch.acos(movement.detach())).median()),median_parameter_norm=float(param.detach().norm(dim=1).median())))
     if step==250:break
     opt.zero_grad();loss.backward();opt.step()
    row=dict(shape=shape,seed=seed,optimizer=optimizer,lr=rate,defaults={k:v for k,v in opt.defaults.items() if isinstance(v,(str,int,float,bool,type(None)))},history=history);rows.append(row);print(shape,seed,optimizer,history[-1],flush=True)
 result=dict(rows=rows,torch_version=torch.__version__,muon_source_sha256=hashlib.sha256(inspect.getsource(implementation).encode()).hexdigest(),seconds=time.monotonic()-start,scope='Synthetic normalized-row direction recovery, not quartic fitting or native performance. Same initial/target per shape/seed, native/toy shapes and frozen existing rates; match_rms is a diagnostic alternative, no queued protocol edits.')
 (P/'NORMALIZED_OPTIMIZER_GEOMETRY_V1.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
