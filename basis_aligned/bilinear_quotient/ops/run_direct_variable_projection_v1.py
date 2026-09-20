#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_replay pred_b_improves pred_c_baseline
"""8 random input-factor fits, ridge-output solve each step.
pred_a_replay relative-squared metric discrepancy<2e-5;
pred_b_improves every fit improves initial objective;
pred_c_baseline width512 Frobenius error<.8981.
Null: variable projection cannot beat teacher-channel baseline at these settings.
Price3*1152*width plus frames. No native forward; no circuit claim.
"""
import os,sys,time,json,math,itertools
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'
PLAN=dict(widths=[128,512],optimizers=['adam','muon'],seeds=[0,1],steps=400,lr=.005,ridge_fraction=1e-6,objective='frobenius',native_forwards=0)
def main():
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print(json.dumps(PLAN));return
 import torch
 from disk_guard import guard_torch_save
 sys.path.insert(0,str(P));from implicit_quadratic import inner
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;start=time.perf_counter();out=P/'VARIABLE_PROJECTION_V1.json';assert not out.exists()
 state=torch.load('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin',weights_only=True)
 def w(k):return state[k].cuda().float()
 with torch.no_grad():
  _,ru=torch.linalg.qr(w('lm_head.weight'));E=torch.cat([torch.eye(1152,device='cuda'),w('transformer.h.16.mlp.Down.weight')*w('transformer.h.17.lambdas')[0],w('transformer.h.17.attn.c_proj.weight')],1);_,re=torch.linalg.qr(E.T)
  c=ru@w('transformer.h.17.mlp.Down.weight');a=w('transformer.h.17.mlp.Left.weight')@re.T;b=w('transformer.h.17.mlp.Right.weight')@re.T
  scale=inner(c,a,b,c,a,b,False).sqrt();c=c/scale;teacher=float(inner(c,a,b,c,a,b,False));tg=float(inner(c,a,b,c,a,b))
 def solve(ll,rr):
  K=.5*((ll@ll.T)*(rr@rr.T)+(ll@rr.T)*(rr@ll.T));cross=c@(.5*((a@ll.T)*(b@rr.T)+(a@rr.T)*(b@ll.T)))
  ridge=PLAN['ridge_fraction']*K.diag().mean();D=torch.linalg.solve(K+ridge*torch.eye(len(ll),device='cuda'),cross.T).T
  loss=(teacher+((D@K)*D).sum()-2*(cross*D).sum())/teacher
  return loss,D,K,cross
 records=[];best={}
 for width,optname,seed in itertools.product(PLAN['widths'],PLAN['optimizers'],PLAN['seeds']):
  torch.manual_seed(seed);l=torch.nn.Parameter(torch.randn(width,1152,device='cuda'));r=torch.nn.Parameter(torch.randn(width,1152,device='cuda'));opt=torch.optim.Adam([l,r],lr=PLAN['lr']) if optname=='adam' else torch.optim.Muon([l,r],lr=PLAN['lr'],weight_decay=0.,adjust_lr_fn='match_rms_adamw');saved=None;valuebest=float('inf');history=[];t0=time.perf_counter()
  for step in range(PLAN['steps']):
   opt.zero_grad();ll=l/l.norm(dim=1,keepdim=True);rr=r/r.norm(dim=1,keepdim=True);loss,D,K,cross=solve(ll,rr);value=float(loss.detach());assert bool(torch.isfinite(loss))
   if value<valuebest:valuebest=value;saved=[v.detach().clone() for v in (D,ll,rr)]
   if step%100==0 or step==PLAN['steps']-1:history.append(dict(step=step,loss=value))
   loss.backward();opt.step()
   for group in opt.param_groups:group['lr']=PLAN['lr']*(.01+.99*.5*(1+math.cos(math.pi*(step+1)/PLAN['steps'])))
  with torch.no_grad():
   D,ll,rr=saved;fi=float((teacher+inner(D,ll,rr,D,ll,rr,False)-2*inner(c,a,b,D,ll,rr,False))/teacher);gi=float((tg+inner(D,ll,rr,D,ll,rr)-2*inner(c,a,b,D,ll,rr))/tg);_,_,K,cross=solve(ll,rr);res=float((D@K-cross).norm()/cross.norm());eig=torch.linalg.eigvalsh(K)
  row=dict(width=width,optimizer=optname,seed=seed,frobenius_relative_error=math.sqrt(max(0,fi)),gaussian_relative_error=math.sqrt(max(0,gi)),metric_replay=abs(fi-valuebest),normal_equation_residual=res,gram_condition=float(eig[-1]/eig[0]),history=history,seconds=time.perf_counter()-t0,parameter_values=3*1152*width);records.append(row);print(json.dumps(row),flush=True)
  key=f'{width}_{optname}'
  if key not in best or fi<best[key]['error']:best[key]=dict(error=fi,seed=seed,factors=[v.cpu() for v in saved])
  out.write_text(json.dumps(dict(plan=PLAN,records=records,seconds=time.perf_counter()-start),indent=2)+'\n')
 result=dict(plan=PLAN,records=records,seconds=time.perf_counter()-start,predictions=dict(pred_a_replay=all(r['metric_replay']<2e-5 for r in records),pred_b_improves=all(r['history'][-1]['loss']<r['history'][0]['loss'] for r in records),pred_c_baseline=any(r['width']==512 and r['frobenius_relative_error']<.8981 for r in records)),scope='Random-input initialization, differentiable ridge-output variable projection. No native activation fitting or circuit identification.')
 guard_torch_save(dict(best=best,teacher_scale=scale.cpu()),str(P/'VARIABLE_PROJECTION_V1.pt'));out.write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
