#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_replay pred_b_zero pred_c_composition
"""Shared response native replay; SHARED_RESPONSE_NATIVE_PLAN_V1.md.
pred_a_replay: each nonzero native effect relativeerror<1e-4.
pred_b_zero: compiled zero effectexact.
pred_c_composition: coefficient compositionexact/sequential residual effect<1e-5.
Null: cachedwriter/Gram/background interface loses native arithmetic.
Price16nativeforwards;optionalcache219756scalarsinclfeaturelibrary; nativebackgroundextra.
"""
import os,sys,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'
def main():
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print(json.dumps(dict(native_forwards=16,context=256,cases=4)));return
 import torch
 import torch.nn.functional as F
 sys.path.insert(0,str(P))
 from circuit_fast_screen_producer import Bilin18TorchBackend
 from native_feature_capture import capture
 from extract_scalar_modes import evaluate as scalar
 from shared_response_circuit import background,evaluate
 torch.set_num_threads(4);torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False
 out=P/'SHARED_RESPONSE_NATIVE_V1.json';assert not out.exists();start=time.perf_counter();model=Bilin18TorchBackend.load('cuda').model.float();a=torch.load(P/'SHARED_RESPONSE_CIRCUIT_V1.pt',weights_only=True);w=a['residual_writers'].cuda().double();gram=a['writer_gram'].cuda().double();v=a['vocabulary_writes'].cuda().double();s={k:t.cuda().double() for k,t in torch.load(P/'SELECTIVE_SCALAR_READOUT_V1.pt',weights_only=True)['program'].items()};eps=torch.finfo(torch.float32).eps;gen=torch.Generator(device='cuda').manual_seed(261016);records=[];zero=[];composition=[]
 logits=lambda state:30*torch.tanh(model.lm_head(F.rms_norm(state,(1152,)))/30)
 for domain in ['fineweb','code']:
  tokens=torch.load(P/f'SELECTIVE_CONFIRMATION_{domain.upper()}_V1.pt',weights_only=True)[:8]
  for doc,row in enumerate(tokens):
   cache=capture(model,row[None,:256].cuda());state=cache['final'].flatten(0,1);den=cache['h17'].square().mean(-1,keepdim=True).flatten(0,1)+eps;features=scalar(s,cache['x16'].flatten(0,1).double());pre=model.lm_head(F.rms_norm(state,(1152,)));square,projections=background(state.double(),w,eps);compiledbase=30*torch.tanh(pre.double()/30);nativebase=logits(state)
   strengths={'zero':torch.zeros_like(features),'removal':torch.ones_like(features),'signed':features.new_tensor([.5,-1,.25,.75]).expand_as(features),'random':2*torch.rand(features.shape,generator=gen,device='cuda',dtype=torch.float64)-1}
   for name,strength in strengths.items():
    t=-features*strength/den.double();pred=evaluate(pre.double(),square,projections,t,gram,v);delta=(t@w.T).float();actual=logits(state+delta);effect=(actual-nativebase).double();pred_effect=pred-compiledbase
    if name=='zero':zero.append(float(pred_effect.abs().max()));continue
    error=float((pred_effect-effect).norm()/effect.norm());first=t.clone();first[:,2:]=0;second=t-first;joint=evaluate(pre.double(),square,projections,first+second,gram,v);composition.append(float((joint-pred).abs().max()));sequential=logits((state+(first@w.T).float())+(second@w.T).float());sequential_error=float((sequential-actual).double().norm()/effect.norm());separate=evaluate(pre.double(),square,projections,first,gram,v)+evaluate(pre.double(),square,projections,second,gram,v)-compiledbase;nonadditivity=float((pred-separate).norm()/pred_effect.norm());records.append(dict(domain=domain,document=doc,case=name,effect_relative_error=error,sequential_residual_effect_error=sequential_error,joint_logit_nonadditivity=nonadditivity))
 pred=dict(pred_a_replay=max(r['effect_relative_error'] for r in records)<1e-4,pred_b_zero=max(zero)==0,pred_c_composition=max(composition)==0 and max(r['sequential_residual_effect_error'] for r in records)<1e-5)
 result=dict(predictions=pred,records=records,maximum_effect_error=max(r['effect_relative_error'] for r in records),maximum_sequential_error=max(r['sequential_residual_effect_error'] for r in records),zero_max=max(zero),coefficient_composition_max=max(composition),seconds=time.perf_counter()-start,scope='Native interface replay for predicted features with mixed edit strengths. Does not measure accuracy against native true feature effects, semanticselectivity or fullmodel replacement.')
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2))
if __name__=='__main__':main()
