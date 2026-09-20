#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_response pred_b_order pred_c_instrument
"""Fused ten-product native response validation on the same registered input screen.
pred_a_response: each candidate learned-direction aggregate response error<.35,
cosine>.9 at each amplitude .25,.5,1 calibration SD (both signs).
pred_b_order: shared quartic response error increases <=.05 versus original quartic at each learned amplitude.
pred_c_instrument: normalized radius relative error<1e-5; response FP32/64
relative discrepancy<1e-3; all target responses nonzero.
Null: good absolute fit hides poor counterfactual changes. Eight learned readers
and eight seeded random directions with matched perturbation L2.128freshrows.
Fused quartic price19632coefficients10products; no fitting; folded-function local responses, not semantic
causal effects or full-model replacements. Prior stability audit perturbed
parameters, whereas this tests native response to input perturbations.
"""
import os,sys,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'
PLAN=dict(amplitudes=[.25,.5,1.],signs=[-1,1],rows=128,directions=8,seed=2040,native_forwards=0)
def main():
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print(json.dumps(PLAN));return
 import torch
 sys.path.insert(0,str(P))
 from frozen_program_evaluation import native,quadratic,quartic,load_teacher
 torch.set_num_threads(4);torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False;out=P/'DIRECTIONAL_RESPONSE_FUSED_V1.json';assert not out.exists();start=time.perf_counter();qs=torch.load(P/'NATIVE_GAUSSIAN_LINEAR_CONTROL_V1.pt',weights_only=True);hs=torch.load(P/'FUSED_ROOT_PROGRAM_V1.pt',weights_only=True);q={k:v.cuda() for k,v in qs['programs']['centered'].items()};h={k:v.cuda() for k,v in hs['programs'][4].items()};teacher=load_teacher('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin',qs['teacher_scale']);panels=torch.load(P/'FROZEN_FRESH_VALIDATION_V1.pt',weights_only=True)['panels'];x=panels[0]['rows'].reshape(32,64,1152)[:,[8,24,40,56]].reshape(128,1152).cuda();cov=torch.load(P/'NATIVE_QUARTIC_COVARIANCE_V1.pt',weights_only=True)['panels'][0]['covariance'].cuda();readers=torch.cat([q['quadratic_left'],q['quadratic_right']]);lengths=readers.norm(dim=1);unit=readers/lengths[:,None];steps=(((readers@cov)*readers).sum(1).sqrt()/lengths);gen=torch.Generator(device='cuda').manual_seed(2040);random=torch.randn(unit.shape,generator=gen,device='cuda');random=random/random.norm(dim=1,keepdim=True);base=dict(teacher=native(teacher,x),quadratic=quadratic(q,x),quartic=quartic(h,x));records=[];maxradius=0.;precision=[]
 for family,directions in [('learned',unit),('random',random)]:
  for amplitude in PLAN['amplitudes']:
   responses={k:[] for k in base};perdirection=[]
   for k,v in enumerate(directions):
    for sign in PLAN['signs']:
     z=x+sign*amplitude*steps[k]*v;z=z/z.norm(dim=1,keepdim=True)*1152**.5;maxradius=max(maxradius,float((z.square().sum(1)/1152-1).abs().max()));preds=dict(teacher=native(teacher,z),quadratic=quadratic(q,z),quartic=quartic(h,z));delta={n:preds[n]-base[n] for n in base}
     if k==0 and sign==1:
      td=[t.double() for t in teacher];exact=native(td,z[:8].double())-native(td,x[:8].double());precision.append(float((exact-delta['teacher'][:8].double()).norm()/exact.norm()))
     for n in base:responses[n].append(delta[n].double())
     y=delta['teacher'].double();perdirection.append(dict(direction=k,sign=sign,target_rms=float(y.square().mean().sqrt()),errors={n:float((delta[n].double()-y).norm()/y.norm()) for n in ['quadratic','quartic']}))
   y=torch.cat(responses['teacher']);metrics={}
   for n in ['quadratic','quartic']:
    p=torch.cat(responses[n]);metrics[n]=dict(error=float((p-y).norm()/y.norm()),cosine=float((p*y).sum()/(p.norm()*y.norm())),response_norm_ratio=float(p.norm()/y.norm()))
   row=dict(family=family,amplitude=amplitude,metrics=metrics,directions=perdirection);records.append(row);print(json.dumps({k:v for k,v in row.items() if k!='directions'}),flush=True)
 learned=[r for r in records if r['family']=='learned'];pred=dict(pred_a_response=all(m['error']<.35 and m['cosine']>.9 for r in learned for m in r['metrics'].values()),pred_b_order=all(r['metrics']['quartic']['error']<=next(old for old in json.loads((P/'DIRECTIONAL_RESPONSE_V1.json').read_text())['records'] if old['family']=='learned' and old['amplitude']==r['amplitude'])['metrics']['quartic']['error']+.05 for r in learned),pred_c_instrument=maxradius<1e-5 and max(precision)<1e-3 and all(d['target_rms']>0 for r in records for d in r['directions']));out.write_text(json.dumps(dict(plan=PLAN,records=records,predictions=pred,maximum_radius_error=maxradius,maximum_response_precision_error=max(precision),seconds=time.perf_counter()-start,scope='Input counterfactual responses of isolated folded function, not semantic/native full-model interventions. Random directions match pre-normalization perturbation L2; normalization can change realized tangent lengths.'),indent=2)+'\n');print(pred,flush=True)
if __name__=='__main__':main()
