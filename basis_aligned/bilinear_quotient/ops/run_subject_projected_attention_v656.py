#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_attention_replay pred_b_response_replay pred_c_causal_prefix
"""Exact attention12 projected into frozen v655 response coordinates.

48 opened prompts,4 prefixes,4 block11 evaluations,16 attention12 evaluations.
No fits/backwards; z perturbs every position by5% raw-state norm before amplitudes.
This is an algebraic instrument, not a full circuit or OOD effect test.
"""
import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parents[3];POLY=ROOT/'basis_aligned/polynomial_causal'
OUT=ROOT/'basis_aligned/bilinear_quotient/circuits/followups/subject_projected_attention_v656_result.json'
PREDICTIONS=dict(pred_a_attention_replay='output relativeL2 and maxabs/max(1,RMS)<=1e-4',
 pred_b_response_replay='finite response relativeL2<=1e-4 for each nonzero amplitude',
 pred_c_causal_prefix='changing final-token coordinates leaves earlier projected outputs exactly unchanged')


def main():
    plan=dict(prefix_calls=4,block11_calls=4,attention_calls=16,rows=48,fits=0,backwards=0,
              amplitudes=[0.,1.,-1.,.5],relative_perturbation=.05,predictions=PREDICTIONS,
              execution_policy='managed_queue_only')
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps(plan));return
    import torch
    import torch.nn.functional as F
    import circuit_fast_screen_producer as producer
    import subject_number_sparse_graph_token_extraction_v1 as graph
    from disk_guard import guard_write
    sys.path.insert(0,str(POLY))
    from projected_two_qk_attention import compile_attention,execute
    if OUT.exists():raise FileExistsError(OUT)
    torch.set_grad_enabled(False);torch.set_num_threads(4);tic=time.perf_counter()
    package=torch.load(OUT.parent/'subject_response_v655_program.pt',map_location='cpu',weights_only=True)
    P=package['producer']['programs'][0]['input_basis'].cuda()
    Q=package['producer']['initial_encoder'].cuda()
    model=producer.Bilin18TorchBackend.load('cuda').model
    a=model.transformer.h[12].attn;heads=a.n_head;hd=a.head_dim;eps=torch.finfo(torch.float32).eps
    weights={key:getattr(a,name).weight.double() for key,name in
             [('q','c_q'),('k','c_k'),('q2','c_q2'),('k2','c_k2'),('v','c_v'),('o','c_proj')]}
    rows=json.loads((POLY/'SUBJECT_POSITION_STRESS_V644_ROWS.json').read_text())
    generator=torch.Generator(device='cuda').manual_seed(656)
    relative=[];absolute=[];scaled_absolute=[];responses=[];causal=[];prices=[]
    counts=dict(prefix_calls=0,block11_calls=0,attention_calls=0)
    def nvalues(obj):
        if isinstance(obj,torch.Tensor):return obj.numel()
        if isinstance(obj,dict):return sum(nvalues(v) for v in obj.values())
        return 0
    for template in dict.fromkeys(r['template'] for r in rows):
        entries=[r for r in rows if r['template']==template]
        ids=torch.tensor([r['token_ids'] for r in entries],device='cuda')
        initial=F.rms_norm(model.transformer.wte(ids),(model.config.n_embd,)).float()
        raw,x0,first,_,_=graph._capture(model,initial,torch,F);counts['prefix_calls']+=1
        block=model.transformer.h[11]
        attention,first=block.attn(F.rms_norm(raw,(raw.shape[-1],)),first)
        x=raw+attention;x=x+block.mlp(F.rms_norm(x,(x.shape[-1],)));counts['block11_calls']+=1
        block=model.transformer.h[12];background=block.lambdas[0]*x+block.lambdas[1]*x0
        b,t,d=background.shape
        cos,sin=a.rotary(torch.zeros(b,t,heads,hd,device='cuda'))
        program=compile_attention(weights,background.double(),P,Q,first.reshape(b,t,heads,hd).double(),
            a.lamb.double(),heads,cos[0,:,0].double(),sin[0,:,0].double(),eps,eps)
        prices.append(dict(template=template,batch=b,length=t,compiled_values=nvalues(program)))
        z=torch.randn(b,t,P.shape[1],generator=generator,device='cuda',dtype=torch.float64)
        z=z*(.05*background.double().norm(dim=-1)/(z@P.T).norm(dim=-1))[...,None]
        native_zero=folded_zero=None
        for amplitude in plan['amplitudes']:
            edited=background+(amplitude*z@P.T).float()
            native,_=a(F.rms_norm(edited,(d,)),first);counts['attention_calls']+=1
            native=native.double()@Q;folded=execute(program,amplitude*z)
            err=folded-native
            relative.append(float(err.norm()/native.norm().clamp_min(1e-20)))
            absolute.append(float(err.abs().max()))
            scaled_absolute.append(float(err.abs().max()/native.square().mean().sqrt().clamp_min(1.)))
            if amplitude==0:native_zero=native;folded_zero=folded
            else:
                true=native-native_zero;pred=folded-folded_zero
                responses.append(float((pred-true).norm()/true.norm().clamp_min(1e-20)))
        changed=z.clone();changed[:,-1]*=2
        causal.append(float((execute(program,changed)[:,:-1]-execute(program,z)[:,:-1]).abs().max()))
        print(template,'native',max(relative),'response',max(responses),flush=True)
    a_ok=max(relative+scaled_absolute)<=1e-4 and counts==dict(prefix_calls=4,block11_calls=4,attention_calls=16)
    b_ok=a_ok and max(responses)<=1e-4;c_ok=a_ok and max(causal)==0.
    result=dict(created_utc=datetime.now(timezone.utc).isoformat(),plan=plan,counts=counts,
        predictions=dict(zip(PREDICTIONS,[bool(a_ok),bool(b_ok),bool(c_ok)])),
        max_output_relative_error=max(relative),max_output_absolute_error=max(absolute),
        max_output_scaled_absolute_error=max(scaled_absolute),max_response_relative_error=max(responses),
        max_causal_prefix_difference=max(causal),prices=prices,wall_seconds=time.perf_counter()-tic,
        scope='Exact projected attention at a declared raw-state basis and native background/first-value ports; all feature slots tied. No compressed full-model or fresh semantic claim.')
    payload=json.dumps(result,indent=2)+'\n';guard_write(len(payload.encode()),str(OUT.parent),'projected attention')
    OUT.write_text(payload);print(json.dumps(result['predictions']));assert a_ok


if __name__=='__main__':main()
