#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_finite pred_b_beats_cp pred_c_error
"""Fixed native hierarchy roots, sampled coefficient output refit.
pred_a_finite; pred_b_beats_cp width8; pred_c_error width128<.95.
No activation forwards. Shared full quadratic bank priced explicitly.
"""
import os,sys,json,time,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'
PLAN=dict(candidates=256,top_writer=128,random=128,train_queries=8192,evaluation_queries=8192,function_queries=256,widths=[1,8,32,128,256],native_forwards=0)
def main():
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print(json.dumps(PLAN));return
 import torch
 from disk_guard import guard_torch_save
 sys.path.insert(0,str(P));from quartic_root_features import root_entries
 from cp_dictionary import select_atoms
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;out=P/'NATIVE_HIERARCHICAL_ROOT_V1.json';assert not out.exists();start=time.perf_counter();state=torch.load('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin',weights_only=True)
 def w(k):return state[k].cuda().float()
 with torch.no_grad():
  _,ru=torch.linalg.qr(w('lm_head.weight'));C=ru@w('transformer.h.17.mlp.Down.weight');L=w('transformer.h.17.mlp.Left.weight');R=w('transformer.h.17.mlp.Right.weight');D=w('transformer.h.16.mlp.Down.weight')*w('transformer.h.17.lambdas')[0];A=w('transformer.h.16.mlp.Left.weight');B=w('transformer.h.16.mlp.Right.weight');scale=math.sqrt(json.load(open(P/'NATIVE_QUARTIC_QUERY_V1.json'))['stratified_energy']);C/=scale;top=C.norm(dim=0).topk(128).indices;mask=torch.ones(C.shape[1],dtype=torch.bool,device='cuda');mask[top]=False;remaining=torch.where(mask)[0];torch.manual_seed(1718);candidates=torch.cat([top,remaining[torch.randperm(len(remaining),device='cuda')[:128]]]);d=1152
  def query(indices):
   features=[];targets=[]
   for ix in indices.split(512):
    root=root_entries(L,R,D,A,B,ix);features.append(root[:,candidates]);targets.append(root@C.T)
   return torch.cat(features),torch.cat(targets)
  gen=torch.Generator(device='cuda');gen.manual_seed(1718);train=torch.randint(d,(8192,4),device='cuda',generator=gen);F,Y=query(train);gen.manual_seed(1651);evaluation=torch.randint(d,(8192,4),device='cuda',generator=gen);E,T=query(evaluation);x=torch.randn(256,d,device='cuda',generator=gen);h=((x@A.T)*(x@B.T))@D.T;root=(h@L.T)*(h@R.T);function_target=root@C.T;function_features=root[:,candidates];feature_scale=F.double().square().mean(0).sqrt();Fn=F.double()/feature_scale;K=Fn.T@Fn/len(F);Q=Y.double().T@Fn/len(F);receipts=select_atoms(K,Q,128);selected=[r for r in receipts if r['width'] in [1,8,32,128]];selected.append(dict(width=256,selected=list(range(256))));records=[];students={}
  for row in selected:
   ix=row['selected'];block=K[ix][:,ix];c=torch.linalg.solve(block,Q[:,ix].T).T;raw_c=c/feature_scale[ix];train_error=float((Fn[:,ix]@c.T-Y.double()).norm()/Y.double().norm());test_error=float((E[:,ix].double()@raw_c.T-T.double()).norm()/T.double().norm());function_error=float((function_features[:,ix].double()@raw_c.T-function_target.double()).norm()/function_target.double().norm());width=len(ix);ids=candidates[ix];record=dict(width=width,training_coefficient_error=train_error,evaluation_coefficient_error=test_error,gaussian_function_error=function_error,condition=float(torch.linalg.cond(block)),reduced_parameters=2*4608*d+2*width*4608+d*width,expanded_parameters=2*4608*d+2*width*4608+50304*width);records.append(record);students[width]=dict(root_indices=ids.cpu(),left=(L[ids]@D).cpu(),right=(R[ids]@D).cpu(),writer=raw_c.cpu());print(json.dumps(record),flush=True);out.write_text(json.dumps(dict(plan=PLAN,records=records,seconds=time.perf_counter()-start),indent=2)+'\n')
  bywidth={r['width']:r for r in records};cp=json.load(open(P/'NATIVE_EXACT_CP_GREEDY_V1.json'))['records'][-1]['evaluation_coefficient_error'];pred=dict(pred_a_finite=all(math.isfinite(r['evaluation_coefficient_error']) for r in records),pred_b_beats_cp=bywidth[8]['evaluation_coefficient_error']<cp,pred_c_error=bywidth[128]['evaluation_coefficient_error']<.95);guard_torch_save(dict(A=A.cpu(),B=B.cpu(),students=students,teacher_scale=scale),str(P/'NATIVE_HIERARCHICAL_ROOT_V1.pt'));out.write_text(json.dumps(dict(plan=PLAN,records=records,candidate_indices=candidates.cpu().tolist(),seconds=time.perf_counter()-start,predictions=pred,scope='Fixed native whole quadratic root products with full shared bank. Sampled coefficient training, independent evaluation. More parameters than flat CP; no equal-price or circuit claim.'),indent=2)+'\n')
if __name__=='__main__':main()
