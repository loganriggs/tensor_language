"""Random-initialization optimizer/structure sweeps using only target coefficients."""
import json,math,time,itertools
from pathlib import Path
import torch
from core import Model,metric,inner,evaluate,terms

def target_cases():
 torch.manual_seed(420);d,o=6,3;result=[]
 for name,kind,width,degree in [('coordinate_sparse','monomial',1,2),('bilinear_cp','cp',3,2),('sparse_tucker','tucker',3,2),('quartic_tree','tree',2,4),('shared_quartic_dag','dag',3,4)]:
  teacher=Model(d,o,kind,width,degree)
  with torch.no_grad():
   if name=='coordinate_sparse':
    teacher.weight.zero_();teacher.weight[0,0]=1.;teacher.weight[1,6]=-1.;teacher.weight[2,12]=.7;teacher.weight[0,20]=.3
   if name=='sparse_tucker':
    teacher.core.zero_();teacher.core[0,0]=1.;teacher.core[0,4]=-.6;teacher.core[1,1]=.8;teacher.core[1,5]=.5
   if name=='shared_quartic_dag':
    teacher.weight[:,3:]=0 # root pairs (0,0),(0,1),(0,2): q0 reused three ways
   target=teacher().detach();target/=inner(target,target,metric(d,degree)).sqrt()
  result.append(dict(name=name,kind=kind,width=width,degree=degree,d=d,o=o,target=target))
 return result

def fit(target,d,degree,kind,width,optimizer,lr,seed,steps=1200,penalty=0.,metric_kind='gaussian',device='cpu',M_override=None,initial_state=None):
 torch.manual_seed(seed);t0=time.perf_counter();target=target.to(device);model=Model(d,target.shape[0],kind,width,degree).to(device);M=metric(d,degree,metric_kind,device) if M_override is None else M_override.to(device);F=metric(d,degree,'frobenius',device);den=inner(target,target,M)
 if initial_state is not None:model.load_state_dict(initial_state)
 opt=(torch.optim.Adam(model.parameters(),lr=lr) if optimizer=='adam' else torch.optim.Muon(model.parameters(),lr=lr,weight_decay=0.,adjust_lr_fn='match_rms_adamw'))
 history=[];best=math.inf;beststate=None;nonfinite=False
 for step in range(steps):
  opt.zero_grad();delta=model()-target;error=inner(delta,delta,M)/den;loss=error+penalty*model.penalty()
  if not bool(torch.isfinite(loss)):nonfinite=True;break
  if float(loss.detach())<best:best=float(loss.detach());beststate={k:v.detach().clone() for k,v in model.state_dict().items()}
  if step%100==0 or step==steps-1:history.append(dict(step=step,error=float(error.detach()),objective=float(loss.detach())))
  if error<1e-12 and penalty==0:break
  loss.backward();opt.step()
  rate=lr*(.01+.99*.5*(1+math.cos(math.pi*(step+1)/steps)))
  for group in opt.param_groups:group['lr']=rate
 if beststate is not None:model.load_state_dict(beststate)
 with torch.no_grad():
  c=model();delta=c-target;relative=float((inner(delta,delta,M)/den).clamp_min(0).sqrt());frob=float((inner(delta,delta,F)/inner(target,target,F)).clamp_min(0).sqrt())
  isotropic=metric(d,degree,'gaussian',device);isoerror=float((inner(delta,delta,isotropic)/inner(target,target,isotropic)).clamp_min(0).sqrt())
  cosine=float(inner(c,target,M)/(inner(c,c,M)*den).sqrt());scale=float((inner(c,c,M)/den).sqrt())
  # Independent shifted/heavy-tail evaluations never enter optimization or selection.
  gen=torch.Generator(device=device).manual_seed(923);x=torch.randn(256,d,generator=gen,device=device,dtype=target.dtype)*2+1
  a,b=evaluate(c,x,degree),evaluate(target,x,degree);shift=float((a-b).norm()/b.norm())
  weight=model.weight;threshold=weight.abs().max()*1e-3;active=int((weight.abs()>threshold).sum())
 return dict(kind=kind,width=width,optimizer=optimizer,lr=lr,seed=seed,penalty=penalty,metric=metric_kind,relative_error=relative,isotropic_gaussian_error=isoerror,symmetric_frobenius_error=frob,cosine=cosine,norm_ratio=scale,shifted_input_error=shift,parameter_values=sum(v.numel() for v in model.parameters()),diagnostic_weight_entries_above_relative_1e3=active,steps=step+1,nonfinite=nonfinite,seconds=time.perf_counter()-t0,history=history),{k:v.cpu() for k,v in model.state_dict().items()}

def toys(out,steps=1200):
 torch.set_num_threads(1);records=[];saved=[];start=time.perf_counter()
 for case in target_cases():
  for variant in ['matched','wide_sparse']:
   width=case['width'] if variant=='matched' else case['width']*2;penalty=0. if variant=='matched' else 1e-4
   for opt,lr,seed in itertools.product(['adam','muon'],[.01,.05],[0,1,2]):
    row,state=fit(case['target'],case['d'],case['degree'],case['kind'],width,opt,lr,seed,steps,penalty)
    row.update(case=case['name'],variant=variant);records.append(row);saved.append(dict(case=case['name'],variant=variant,row=len(records)-1,state=state))
    print(json.dumps({k:row[k] for k in ['case','variant','optimizer','lr','seed','relative_error','seconds']}),flush=True)
  out.write_text(json.dumps(dict(records=records,seconds=time.perf_counter()-start),indent=2)+'\n')
 summary={name:{opt:dict(best=min(r['relative_error'] for r in records if r['case']==name and r['optimizer']==opt),recoveries=sum(r['relative_error']<1e-3 for r in records if r['case']==name and r['optimizer']==opt)) for opt in ['adam','muon']} for name in [c['name'] for c in target_cases()]}
 out.write_text(json.dumps(dict(records=records,summary=summary,seconds=time.perf_counter()-start,scope='Exact weight-only Gaussian polynomial matching; shifted inputs validate only. Thresholded weights are diagnostic, not deployed sparsity.'),indent=2)+'\n')
 torch.save(saved,out.with_suffix('.pt'));print(json.dumps(summary),flush=True)
if __name__=='__main__':
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--steps',type=int,default=1200);args=ap.parse_args();assert not args.out.exists();toys(args.out,args.steps)
