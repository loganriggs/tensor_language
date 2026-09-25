#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_contrast pred_b_order pred_c_projection
"""Frozen-program vocabulary contrast diagnostic, no refitting.
pred_a_contrast: ten-product contrast error<.25 on both panels.
pred_b_order: original quartic contrast error below ten-product on both panels.
pred_c_projection: implicit/direct vocabulary centered squared norms agree<1e-5.
Null: common-logit component explains apparent predictive fidelity. No probability
or whole-model metric claim: final softcap remains explicit.
"""
import os,sys,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'
PLAN=dict(candidates=['quadratic4','quartic26','quartic10'],contexts=[64,256],native_forwards=0)
def main():
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print(json.dumps(PLAN));return
 import torch
 from disk_guard import guard_torch_save
 sys.path.insert(0,str(P))
 from frozen_program_evaluation import quadratic,quartic
 torch.set_num_threads(4);torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False;out=P/'VOCABULARY_CONTRAST_V1.json';assert not out.exists();start=time.perf_counter();state=torch.load('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin',weights_only=True);Q,_=torch.linalg.qr(state['lm_head.weight'].cuda().float());del state;v=Q.double().sum(0)/len(Q)**.5
 def load(file,key):return {k:t.cuda().double() for k,t in torch.load(P/file,weights_only=True)['programs'][key].items()}
 models=dict(quadratic4=load('NATIVE_GAUSSIAN_LINEAR_CONTROL_V1.pt','centered'),quartic26=load('NATIVE_QUARTIC_MEAN_V1.pt',8),quartic10=load('FUSED_ROOT_PROGRAM_V1.pt',4));panels=torch.load(P/'FROZEN_FRESH_VALIDATION_V1.pt',weights_only=True)['panels'];rows=[];checks=[]
 def energy(z):return z.square().sum()-(z@v).square().sum()
 for panel in panels:
  x=panel['rows'].cuda().double();y=panel['targets'].cuda().double();yc=y-y.mean(0)
  for name,s in models.items():
   pred=quadratic(s,x) if name=='quadratic4' else quartic(s,x);delta=pred-y;dc=delta-delta.mean(0);row=dict(candidate=name,context=panel['context'],ordinary_error=float(delta.norm()/y.norm()),contrast_error=float((energy(delta)/energy(y)).sqrt()),ordinary_variation_error=float(dc.norm()/yc.norm()),contrast_variation_error=float((energy(dc)/energy(yc)).sqrt()),target_common_energy_fraction=float((y@v).square().sum()/y.square().sum()),residual_common_energy_fraction=float((delta@v).square().sum()/delta.square().sum()),contrast_mean_error_energy=float(len(y)*energy(delta.mean(0)[None,:])/energy(y)));rows.append(row)
   z=delta[:8];direct=z@Q.double().T;direct-=direct.mean(1,keepdim=True);checks.append(float(abs(direct.square().sum()-energy(z))/direct.square().sum()));print(row,flush=True)
 pred=dict(pred_a_contrast=all(r['contrast_error']<.25 for r in rows if r['candidate']=='quartic10'),pred_b_order=all(next(r for r in rows if r['context']==c and r['candidate']=='quartic26')['contrast_error']<next(r for r in rows if r['context']==c and r['candidate']=='quartic10')['contrast_error'] for c in PLAN['contexts']),pred_c_projection=max(checks)<1e-5);guard_torch_save(dict(common_direction_in_readout_frame=v.cpu()),str(P/'VOCABULARY_CONTRAST_V1.pt'));out.write_text(json.dumps(dict(plan=PLAN,records=rows,predictions=pred,projection_squared_norm_relative_error=max(checks),projected_uniform_direction_norm=float(v.norm()),seconds=time.perf_counter()-start,scope='Frozen isolated polynomial prediction in pre-softcap vocabulary contrast geometry; ordinary error retained, no model intervention/refitting/KL or semantic claim.'),indent=2)+'\n');print(pred,flush=True)
if __name__=='__main__':main()
