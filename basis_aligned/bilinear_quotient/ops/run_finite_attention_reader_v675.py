#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_native_forward pred_b_finite_closure pred_c_causal_reader
"""Validate exact finite-change readers through native attention12-17.

48 opened rows. 4 prefixes,4 block11,96 attention,24 MLP calls.
96 analytical pullbacks, each recomputing two reference forwards; plus96 direct
reference forwards. Zero fitting, optimizer steps or autograd backwards.
Null: product/norm/rotary handling breaks scalar finite-change closure.
"""
import os,json,sys,time
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[3];POLY=ROOT/'basis_aligned/polynomial_causal'
OUT=ROOT/'basis_aligned/bilinear_quotient/circuits/followups/finite_attention_reader_v675_result.json'
PREDICTIONS=dict(pred_a_native_forward='all native output relative errors<=1e-4',
 pred_b_finite_closure='native closure absolute and norm-scaled errors<=1e-4; float64 closure absolute<=1e-10',
 pred_c_causal_reader='reader at first token has exactly zero pullback on later input tokens')

def main():
 plan=dict(rows=48,prefix_calls=4,block11_calls=4,attention_calls=96,mlp_calls=24,
           analytical_pullbacks=96,reference_attention_forwards=288,optimizer_steps=0,
           autograd_backwards=0,amplitudes=[0.,1.,-1.,.5],predictions=PREDICTIONS,
           execution_policy='managed_queue_only')
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
  print(json.dumps(plan));return
 import torch
 import torch.nn.functional as F
 import circuit_fast_screen_producer as producer
 import subject_number_sparse_graph_token_extraction_v1 as graph
 from disk_guard import guard_write
 sys.path.insert(0,str(POLY))
 from finite_attention_readers import forward,pullback
 if OUT.exists():raise FileExistsError(OUT)
 torch.set_grad_enabled(False);torch.set_num_threads(4);tic=time.perf_counter()
 model=producer.Bilin18TorchBackend.load('cuda').model;eps=torch.finfo(torch.float32).eps
 rows=json.loads((POLY/'SUBJECT_POSITION_STRESS_V644_ROWS.json').read_text())
 generator=torch.Generator(device='cuda').manual_seed(675)
 counts=dict(prefix_calls=0,block11_calls=0,attention_calls=0,mlp_calls=0,analytical_pullbacks=0,reference_attention_forwards=0)
 forward_errors=[];closure_abs=[];closure_scaled=[];double_closure=[];causal=[];per_layer={}
 for template in dict.fromkeys(r['template'] for r in rows):
  entries=[r for r in rows if r['template']==template]
  ids=torch.tensor([r['token_ids'] for r in entries],device='cuda')
  initial=F.rms_norm(model.transformer.wte(ids),(model.config.n_embd,)).float()
  raw,x0,first,_,_=graph._capture(model,initial,torch,F);counts['prefix_calls']+=1
  block=model.transformer.h[11];a,first=block.attn(F.rms_norm(raw,(raw.shape[-1],)),first)
  x=raw+a;x=x+block.mlp(F.rms_norm(x,(x.shape[-1],)));counts['block11_calls']+=1
  for layer in range(12,18):
   block=model.transformer.h[layer];module=block.attn
   background=block.lambdas[0]*x+block.lambdas[1]*x0
   b,t,d=background.shape;heads=module.n_head;hd=module.head_dim
   cos,sin=module.rotary(torch.zeros(b,t,heads,hd,device='cuda'))
   weights={key:getattr(module,name).weight.double() for key,name in [('q','c_q'),('k','c_k'),('q2','c_q2'),('k2','c_k2'),('v','c_v'),('o','c_proj')]}
   args=(first.reshape(b,t,heads,hd).double(),module.lamb.double(),heads,cos[0,:,0].double(),sin[0,:,0].double(),eps,eps)
   delta=torch.randn(background.shape,device='cuda',dtype=torch.float64,generator=generator)
   delta*=.05*background.double().norm(dim=-1,keepdim=True)/delta.norm(dim=-1,keepdim=True)
   g=torch.randn(background.shape,device='cuda',dtype=torch.float64,generator=generator)
   g/=g.square().sum((1,2),keepdim=True).sqrt()
   local=[]
   for amplitude in plan['amplitudes']:
    edited=background+(amplitude*delta).float()
    native,_=module(F.rms_norm(edited,(d,)),first);counts['attention_calls']+=1
    exact,_=forward(weights,edited.double(),*args);counts['reference_attention_forwards']+=1
    forward_errors.append(float((native.double()-exact).norm()/native.double().norm().clamp_min(1e-20)))
    if amplitude==0:
     native0=native;exact0=exact;continue
    q=pullback(weights,background.double(),edited.double(),*args,g)
    counts['analytical_pullbacks']+=1;counts['reference_attention_forwards']+=2
    predicted=(q*(edited.double()-background.double())).sum((1,2))
    actual=(g*(native.double()-native0.double())).sum((1,2))
    reference=(g*(exact-exact0)).sum((1,2))
    error=predicted-actual
    scale=(native.double()-native0.double()).square().sum((1,2)).sqrt().clamp_min(1e-20)
    closure_abs.append(float(error.abs().max()));closure_scaled.append(float((error.abs()/scale).max()))
    double_closure.append(float((predicted-reference).abs().max()));local.append(closure_scaled[-1])
   first_reader=torch.zeros_like(g);first_reader[:,0]=g[:,0]
   q=pullback(weights,background.double(),edited.double(),*args,first_reader)
   counts['analytical_pullbacks']+=1;counts['reference_attention_forwards']+=2
   causal.append(float(q[:,1:].abs().max()))
   per_layer.setdefault(str(layer),[]).extend(local)
   x=background+native0;x=x+block.mlp(F.rms_norm(x,(d,)));counts['mlp_calls']+=1
  print(template,'closure',max(closure_scaled),'forward',max(forward_errors),flush=True)
 expected={k:plan[k] for k in counts}
 a=counts==expected and max(forward_errors)<=1e-4
 b=a and max(closure_abs+closure_scaled)<=1e-4 and max(double_closure)<=1e-10
 c=a and max(causal)==0
 result=dict(created_utc=datetime.now(timezone.utc).isoformat(),plan=plan,counts=counts,
  predictions=dict(zip(PREDICTIONS,[bool(a),bool(b),bool(c)])),max_native_forward_relative=max(forward_errors),
  max_native_closure_absolute=max(closure_abs),max_native_closure_scaled=max(closure_scaled),
  max_float64_closure_absolute=max(double_closure),max_causal_leakage=max(causal),
  layer_max_scaled_closure={k:max(v) for k,v in per_layer.items()},wall_seconds=time.perf_counter()-tic,
  scope='Exact two-endpoint calibration/attribution reader; edited states required, not predictive circuit extraction')
 payload=json.dumps(result,indent=2)+'\n';guard_write(len(payload.encode()),str(OUT.parent),'finite attention readers')
 OUT.write_text(payload);print(json.dumps(result['predictions']));assert a and b and c
if __name__=='__main__':main()
