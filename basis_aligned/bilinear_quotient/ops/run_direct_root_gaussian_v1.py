#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_solver pred_b_metric pred_c_radial
"""Fixed native hierarchical roots, synthetic Gaussian writer refit.
pred_a_solver finite/residual<1e-8; pred_b_metric improves256 by.03; pred_c_radial beatsradial.
12fits widths8/32/128/256, ridge0/1e-4/1e-2; no textforwards.
"""
import os,sys,json,time,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'
PLAN=dict(widths=[8,32,128,256],ridges=[0.,1e-4,1e-2],train_inputs=4096,evaluation_inputs=1024,evaluation_coefficients=8192,native_forwards=0)
def main():
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print(json.dumps(PLAN));return
 import torch
 from disk_guard import guard_torch_save
 sys.path.insert(0,str(P));from quartic_root_features import root_entries
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;out=P/'NATIVE_ROOT_GAUSSIAN_V1.json';assert not out.exists();start=time.perf_counter();source=torch.load(P/'NATIVE_HIERARCHICAL_ROOT_V1.pt',weights_only=True);state=torch.load('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin',weights_only=True)
 def w(k):return state[k].cuda().float()
 with torch.no_grad():
  _,ru=torch.linalg.qr(w('lm_head.weight'));C=ru@w('transformer.h.17.mlp.Down.weight')/source['teacher_scale'];L=w('transformer.h.17.mlp.Left.weight');R=w('transformer.h.17.mlp.Right.weight');D=w('transformer.h.16.mlp.Down.weight')*w('transformer.h.17.lambdas')[0];A=source['A'].cuda();B=source['B'].cuda();all_ids=source['students'][256]['root_indices'].tolist();lookup={v:i for i,v in enumerate(all_ids)};sets={k:torch.tensor([lookup[v] for v in source['students'][k]['root_indices'].tolist()],device='cuda') for k in PLAN['widths']};ids=torch.tensor(all_ids,device='cuda');left=source['students'][256]['left'].cuda();right=source['students'][256]['right'].cuda()
  def function(x):
   features=[];targets=[]
   for batch in x.split(512):
    m=(batch@A.T)*(batch@B.T);h=m@D.T;targets.append(((h@L.T)*(h@R.T))@C.T);features.append((m@left.T)*(m@right.T))
   return torch.cat(features).double(),torch.cat(targets).double()
  gen=torch.Generator(device='cuda');gen.manual_seed(1736);trainx=torch.randn(4096,1152,device='cuda',generator=gen);evalx=torch.randn(1024,1152,device='cuda',generator=gen);F,Y=function(trainx);E,T=function(evalx);feature_scale=F.square().mean(0).sqrt();Fn=F/feature_scale;gen.manual_seed(1651);indices=torch.randint(1152,(8192,4),device='cuda',generator=gen);coef_features=[];coef_target=[]
  for ix in indices.split(512):
   roots=root_entries(L,R,D,A,B,ix);coef_features.append(roots[:,ids]);coef_target.append(roots@C.T)
  EF=torch.cat(coef_features).double();YT=torch.cat(coef_target).double();rad=trainx.double().square().sum(1).square();radc=(rad[:,None]*Y).sum(0)/rad.square().sum();radial_error=float((evalx.double().square().sum(1).square()[:,None]*radc-T).norm()/T.norm());records=[];writers={};original={}
  for width in PLAN['widths']:
   ix=sets[width];X=Fn[:,ix];G=X.T@X/len(X);cross=Y.T@X/len(X);old=source['students'][width]['writer'].cuda().double();original[width]=float((E[:,ix]@old.T-T).norm()/T.norm())
   for ridge in PLAN['ridges']:
    system=G+ridge*torch.eye(width,device='cuda',dtype=torch.float64);c=torch.linalg.solve(system,cross.T).T;raw=c/feature_scale[ix];residual=float((c@system-cross).norm()/cross.norm());row=dict(width=width,ridge=ridge,training_gaussian_error=float((X@c.T-Y).norm()/Y.norm()),evaluation_gaussian_error=float((E[:,ix]@raw.T-T).norm()/T.norm()),evaluation_coefficient_error=float((EF[:,ix]@raw.T-YT).norm()/YT.norm()),normal_residual=residual,condition=float(torch.linalg.cond(system)),reduced_parameters=2*A.numel()+2*width*A.shape[0]+1152*width);records.append(row);writers[(width,ridge)]=raw.cpu();print(json.dumps(row),flush=True)
   out.write_text(json.dumps(dict(plan=PLAN,records=records,original_coefficient_writer_gaussian_errors=original,radial_gaussian_error=radial_error,seconds=time.perf_counter()-start),indent=2)+'\n')
  best=min(r['evaluation_gaussian_error'] for r in records if r['width']==256);pred=dict(pred_a_solver=all(math.isfinite(r['evaluation_gaussian_error']) and r['normal_residual']<1e-8 for r in records),pred_b_metric=best<original[256]-.03,pred_c_radial=best<radial_error);guard_torch_save(dict(writers=writers,source='NATIVE_HIERARCHICAL_ROOT_V1.pt',radial_writer=radc.cpu()),str(P/'NATIVE_ROOT_GAUSSIAN_V1.pt'));out.write_text(json.dumps(dict(plan=PLAN,records=records,original_coefficient_writer_gaussian_errors=original,radial_gaussian_error=radial_error,seconds=time.perf_counter()-start,predictions=pred,scope='Fixed hierarchy, sampled synthetic Gaussian writer fit; independent coefficient and Gaussian evaluations. No heldout selection for deployment, covariance/empirical activation or circuit claim.'),indent=2)+'\n')
if __name__=='__main__':main()
