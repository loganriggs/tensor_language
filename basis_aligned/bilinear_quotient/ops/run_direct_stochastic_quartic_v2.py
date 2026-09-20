#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_finite pred_b_improves pred_c_small_error
"""Eight native full-input quartic random-student fits; no native forward calls.
pred_a_finite all training/eval finite; pred_b_improves min error<.99;
pred_c_small_error min evaluation coefficient error<.9.
Null: low-width shared bilinear DAG cannot fit this global quartic at this budget.
Price2*k*1152+2*128*k+1152*128 plus fixed output frame.
"""
import os,sys,json,time,math,itertools
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'
PLAN=dict(output_parameterization='unit_rms_parameter_times_fixed_scale',bank_widths=[128,512],root_width=128,optimizers=['adam','muon'],seeds=[0,1],steps=300,batch_size=256,lr=.05,calibration_queries=2048,evaluation_queries=8192,function_queries=256,native_forwards=0)
def main():
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print(json.dumps(PLAN));return
 import torch
 from disk_guard import guard_torch_save
 sys.path.insert(0,str(P));from implicit_quartic import entries
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;out=P/'NATIVE_STOCHASTIC_QUARTIC_V2.json';assert not out.exists();start=time.perf_counter()
 state=torch.load('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin',weights_only=True)
 def w(k):return state[k].cuda().float()
 with torch.no_grad():
  _,ru=torch.linalg.qr(w('lm_head.weight'));teacher=[ru@w('transformer.h.17.mlp.Down.weight'),w('transformer.h.17.mlp.Left.weight'),w('transformer.h.17.mlp.Right.weight'),w('transformer.h.16.mlp.Down.weight')*w('transformer.h.17.lambdas')[0],w('transformer.h.16.mlp.Left.weight'),w('transformer.h.16.mlp.Right.weight')];d=1152;o=1152
  gen=torch.Generator(device='cuda');gen.manual_seed(1618);cal=torch.randint(d,(2048,4),device='cuda',generator=gen);scale=entries(*teacher,cal).square().sum(-1).mean().sqrt();teacher[0]=teacher[0]/scale
  evaluation=torch.randint(d,(8192,4),device='cuda',generator=gen);target=torch.cat([entries(*teacher,z) for z in evaluation.split(512)]);evalden=target.square().sum();s=torch.linalg.eigvalsh(target.T@target);rank_bound=float((s[:-128].clamp_min(0).sum()/s.clamp_min(0).sum()).sqrt())
  x=torch.randn(256,d,device='cuda',generator=gen)
  def execute(p,x):
   C,L,R,D,A,B=p;q=((x@A.T)*(x@B.T))@D.T;return ((q@L.T)*(q@R.T))@C.T
  function_target=execute(teacher,x)
 records=[];saved={}
 for k,optname,seed in itertools.product(PLAN['bank_widths'],PLAN['optimizers'],PLAN['seeds']):
  torch.manual_seed(seed);r=128;C=torch.nn.Parameter(torch.randn(o,r,device='cuda'));L=torch.nn.Parameter(torch.randn(r,k,device='cuda'));R=torch.nn.Parameter(torch.randn(r,k,device='cuda'));A=torch.nn.Parameter(torch.randn(k,d,device='cuda'));B=torch.nn.Parameter(torch.randn(k,d,device='cuda'));D=torch.eye(k,device='cuda');parameters=[C,L,R,A,B];opt=torch.optim.Adam(parameters,lr=.05) if optname=='adam' else torch.optim.Muon(parameters,lr=.05,weight_decay=0.,adjust_lr_fn='match_rms_adamw');train_gen=torch.Generator(device='cuda');train_gen.manual_seed(17000+seed);history=[];finite=True;t0=time.perf_counter()
  def student():return [C*(.1/math.sqrt(o*r)),L/L.norm(dim=1,keepdim=True),R/R.norm(dim=1,keepdim=True),D,A/A.norm(dim=1,keepdim=True)*math.sqrt(d),B/B.norm(dim=1,keepdim=True)*math.sqrt(d)]
  for step in range(300):
   idx=torch.randint(d,(256,4),device='cuda',generator=train_gen)
   with torch.no_grad():truth=entries(*teacher,idx)
   opt.zero_grad();pred=entries(*student(),idx);loss=(pred-truth).square().sum(-1).mean()
   if not bool(torch.isfinite(loss)):finite=False;break
   if step%50==0 or step==299:history.append(dict(step=step,sampled_scaled_squared_error=float(loss.detach())))
   loss.backward();opt.step()
   for group in opt.param_groups:group['lr']=.05*(.01+.99*.5*(1+math.cos(math.pi*(step+1)/300)))
  with torch.no_grad():
   p=student();prediction=torch.cat([entries(*p,z) for z in evaluation.split(512)]);error=float((prediction-target).norm()/evalden.sqrt());function_error=float((execute(p,x)-function_target).norm()/function_target.norm())
  row=dict(bank_width=k,root_width=r,optimizer=optname,seed=seed,evaluation_coefficient_error=error,gaussian_function_sample_error=function_error,finite=finite and math.isfinite(error),parameter_values=2*k*d+2*r*k+o*r,history=history,seconds=time.perf_counter()-t0);records.append(row);print(json.dumps(row),flush=True);saved[f'{k}_{optname}_{seed}']=[v.detach().cpu() for v in p if v is not D]
  out.write_text(json.dumps(dict(plan=PLAN,records=records,finite_query_output_rank128_lower_bound=rank_bound,teacher_scale=float(scale),seconds=time.perf_counter()-start),indent=2)+'\n')
 result=dict(plan=PLAN,records=records,finite_query_output_rank128_lower_bound=rank_bound,teacher_scale=float(scale),seconds=time.perf_counter()-start,predictions=dict(pred_a_finite=all(r['finite'] for r in records),pred_b_improves=min(r['evaluation_coefficient_error'] for r in records)<.99,pred_c_small_error=min(r['evaluation_coefficient_error'] for r in records)<.9),scope='Full1152input pure native quartic coefficient fit with dense shared bilinear cores, final checkpoints, finite query evaluation; no exact small-error certificate, behavioral OOD or circuit identity claim.')
 guard_torch_save(dict(students=saved,teacher_scale=scale.cpu(),factor_order_without_identity=['C','L2','R2','L1','R1']),str(P/'NATIVE_STOCHASTIC_QUARTIC_V2.pt'));out.write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
