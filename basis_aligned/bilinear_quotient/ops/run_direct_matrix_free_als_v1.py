#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_finite pred_b_cg pred_c_baseline
"""Four native symmetric quadratic matrix-free ALS fits.
pred_a_finite all loss finite; pred_b_cg >=95%block residuals<.1;
pred_c_baseline width512 Frobenius error<.8981.
30sweeps,30CGsteps/proximalblock, no native forwards. Reject increasing sweeps,
report rejections. Price3*1152*width plus exactQRframes; no circuit claim.
"""
import os,sys,json,time,math,itertools
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'
PLAN=dict(widths=[128,512],seeds=[0,1],sweeps=30,cg_steps=30,cg_tolerance=1e-5,proximal_fraction=1e-6,native_forwards=0)
def main():
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print(json.dumps(PLAN));return
 import torch
 from disk_guard import guard_torch_save
 sys.path.insert(0,str(P));from implicit_quadratic import inner
 from matrix_free_quadratic import input_operator,input_rhs,conjugate_gradient
 torch.set_num_threads(4);out=P/'MATRIX_FREE_NATIVE_ALS_V1.json';assert not out.exists();start=time.perf_counter()
 state=torch.load('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin',weights_only=True)
 def w(k):return state[k].cuda().double()
 with torch.no_grad():
  _,ru=torch.linalg.qr(w('lm_head.weight'));E=torch.cat([torch.eye(1152,device='cuda',dtype=torch.float64),w('transformer.h.16.mlp.Down.weight')*w('transformer.h.17.lambdas')[0],w('transformer.h.17.attn.c_proj.weight')],1);_,re=torch.linalg.qr(E.T);tc=ru@w('transformer.h.17.mlp.Down.weight');ta=w('transformer.h.17.mlp.Left.weight')@re.T;tb=w('transformer.h.17.mlp.Right.weight')@re.T;scale=inner(tc,ta,tb,tc,ta,tb,False).sqrt();tc=tc/scale;tf=float(inner(tc,ta,tb,tc,ta,tb,False));tg=float(inner(tc,ta,tb,tc,ta,tb))
  records=[];saved={}
  def error(C,A,B,gaussian=False):
   den=tg if gaussian else tf
   return float((den+inner(C,A,B,C,A,B,gaussian)-2*inner(tc,ta,tb,C,A,B,gaussian))/den)
  for width,seed in itertools.product(PLAN['widths'],PLAN['seeds']):
   torch.manual_seed(seed);A=torch.randn(width,1152,device='cuda',dtype=torch.float64);B=torch.randn_like(A);A/=A.norm(dim=1,keepdim=True);B/=B.norm(dim=1,keepdim=True);C=torch.randn(1152,width,device='cuda',dtype=torch.float64)*.1/math.sqrt(1152*width);history=[];cgrows=[];rejections=0;old=error(C,A,B);initial=old;t0=time.perf_counter()
   for iteration in range(30):
    previous=(C.clone(),A.clone(),B.clone());K=.5*((A@A.T)*(B@B.T)+(A@B.T)*(B@A.T));cross=tc@(.5*((ta@A.T)*(tb@B.T)+(ta@B.T)*(tb@A.T)));ridge=1e-6*K.diag().mean();C=torch.linalg.solve(K+ridge*torch.eye(width,device='cuda'),(cross+ridge*C).T).T
    for which in ['A','B']:
     active,other=(A,B) if which=='A' else (B,A);apply,diag=input_operator(C,other);rhs=input_rhs(C,other,tc,ta,tb);ridge=1e-6*diag.mean();new,receipt=conjugate_gradient(lambda x:apply(x)+ridge*x,rhs+ridge*active,active,diag+ridge,tolerance=1e-5,max_steps=30);receipt.update(iteration=iteration,block=which);cgrows.append(receipt);norm=new.norm(dim=1).clamp_min(1e-30);new/=norm[:,None];C=C*norm
     if which=='A':A=new
     else:B=new
    proposed=error(C,A,B);assert math.isfinite(proposed)
    if proposed>old+1e-9:C,A,B=previous;rejections+=1
    else:old=proposed
    history.append(dict(iteration=iteration,accepted_relative_squared_error=old,proposed_relative_squared_error=proposed))
   row=dict(width=width,seed=seed,frobenius_relative_error=math.sqrt(max(0,old)),gaussian_relative_error=math.sqrt(max(0,error(C,A,B,True))),initial_relative_error=math.sqrt(initial),rejected_sweeps=rejections,cg=cgrows,history=history,seconds=time.perf_counter()-t0,parameter_values=3*1152*width);records.append(row);saved[f'{width}_{seed}']=[x.cpu() for x in [C,A,B]];print(width,seed,row['frobenius_relative_error'],rejections,flush=True)
   out.write_text(json.dumps(dict(plan=PLAN,records=records,seconds=time.perf_counter()-start),indent=2)+'\n')
  cg=[v for r in records for v in r['cg']];fraction=sum(v['true_relative_residual']<.1 for v in cg)/len(cg);result=dict(plan=PLAN,records=records,cg_fraction_below_point1=fraction,seconds=time.perf_counter()-start,predictions=dict(pred_a_finite=all(math.isfinite(r['frobenius_relative_error']) for r in records),pred_b_cg=fraction>=.95,pred_c_baseline=any(r['width']==512 and r['frobenius_relative_error']<.8981 for r in records)),scope='Random-start matrix-free symmetric coefficient ALS, explicit proximal blocks and rejected-sweep ledger. No global guarantee or causal identification.')
  guard_torch_save(dict(students=saved,teacher_scale=scale.cpu()),str(P/'MATRIX_FREE_NATIVE_ALS_V1.pt'));out.write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
