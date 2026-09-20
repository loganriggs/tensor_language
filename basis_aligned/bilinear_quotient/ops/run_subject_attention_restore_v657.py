#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_exact_replay pred_b_effect_prediction pred_c_capable_position_transfer
"""Causal restoration of attention12: native versus projected full two-QK fold.

48 new matched-position sequences, eight prefix calls and twenty-four suffix calls, zero fits.
One exact baseline, true edited suffix, frozen edited suffix, frozen baseline replay.
Six baseline attention-write ports remain charged. Not standalone extraction.
Null: small secant attributions conceal material downstream feedback.
"""
import os
import json
import time
from pathlib import Path
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parents[3]
POLY=ROOT/'basis_aligned/polynomial_causal'
OUT=ROOT/'basis_aligned/bilinear_quotient/circuits/followups/subject_attention_freeze_v657_result.json'
PREDICTIONS=dict(pred_a_exact_replay='frozen-baseline replay max absolute error <=1e-4',
 pred_b_effect_prediction='restoring native attention12 halves all-frozen effect error in every postposed cell',
 pred_c_capable_position_transfer='folded attention12 restoration full native effect relativeL2<=.10 every cell')


def main():
    plan=dict(prefix_calls=8,suffix_calls=24,sequences_per_arm=48,fits=0,backwards=0,
              frozen_attention_layers=list(range(12,18)),native_background_ports=6,
              execution_policy='managed_queue_only',predictions=PREDICTIONS)
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps(plan));return
    import torch
    import torch.nn.functional as F
    import numpy as np
    import circuit_fast_screen_producer as producer
    import subject_number_sparse_graph_token_extraction_v1 as graph
    from disk_guard import guard_write
    if OUT.exists():raise FileExistsError(OUT)
    torch.set_grad_enabled(False);torch.set_num_threads(4);tic=time.perf_counter()
    model=producer.Bilin18TorchBackend.load('cuda').model
    decoder=json.loads((POLY/'SUBJECT_NUMBER_EMBEDDING_DECODER_FROZEN_V1_ARTIFACT.json').read_text())['frozen_decoder']
    axis=torch.tensor(decoder['axis'],device='cuda',dtype=torch.float64)
    unit=axis/axis.norm();target_projection=float(decoder['threshold'])/float(axis.norm())
    import hashlib
    rowfile=POLY/'SUBJECT_POSITION_STRESS_V644_ROWS.json'
    assert hashlib.sha256(rowfile.read_bytes()).hexdigest()=='03dd63a4b5aa3e4bf77f41280217d449084e2ad5121b0919f7ab0e07ce6c4db1'
    rows=json.loads(rowfile.read_text());batches=[]
    for template in dict.fromkeys(r['template'] for r in rows):
        batches.append([(r['token_ids'],r['subject_position'],r['answer_ids'],r['family'],r['readout_position'])
                        for r in rows if r['template']==template])
    assert sum(map(len,batches))==48
    import sys
    sys.path.insert(0,str(POLY))
    from projected_two_qk_attention import compile_attention,execute as folded_attention
    package=torch.load(OUT.parent/'subject_response_v655_program.pt',map_location='cpu',weights_only=True)
    P=package['producer']['programs'][0]['input_basis'].cuda()
    Q=package['producer']['initial_encoder'].cuda()
    attn=model.transformer.h[12].attn
    weights={key:getattr(attn,name).weight.double() for key,name in
             [('q','c_q'),('k','c_k'),('q2','c_q2'),('k2','c_k2'),('v','c_v'),('o','c_proj')]}
    context={};eps=torch.finfo(torch.float32).eps
    cells={};replay=[];counts=dict(prefix_calls=0,suffix_calls=0)

    def suffix(raw,x0,first,pos,answers,frozen=None,mode="freeze"):
        counts['suffix_calls']+=1;x=raw;writes={}
        for layer in range(11,18):
            block=model.transformer.h[layer]
            if layer>11:x=block.lambdas[0]*x+block.lambdas[1]*x0
            if layer==12 and frozen is None:
                b,t,d=x.shape;heads=attn.n_head;hd=attn.head_dim
                cos,sin=attn.rotary(torch.zeros(b,t,heads,hd,device=x.device))
                context['background']=x.double().clone()
                context['program']=compile_attention(weights,x.double(),P,Q,first.reshape(b,t,heads,hd).double(),
                    attn.lamb.double(),heads,cos[0,:,0].double(),sin[0,:,0].double(),eps,eps)
                context['zero']=folded_attention(context['program'],torch.zeros(b,t,P.shape[1],device=x.device,dtype=torch.float64))
            if frozen is not None and layer==12 and mode=='folded':
                z=(x.double()-context['background'])@Q
                projected=folded_attention(context['program'],z)-context['zero']
                a=frozen[layer]+(projected@P.T).float()
            elif frozen is not None and layer in frozen and not(layer==12 and mode=='native12'):
                a=frozen[layer]
            else:
                a,first=block.attn(F.rms_norm(x,(x.shape[-1],)),first)
            x=x+a;x=x+block.mlp(F.rms_norm(x,(x.shape[-1],)))
            if layer>=12:writes[layer]=a
        batch=torch.arange(len(x),device=x.device)
        logits=30*torch.tanh(model.lm_head(F.rms_norm(x[batch,pos],(x.shape[-1],)))/30)
        selected=logits.gather(1,answers).double()
        return selected[:,0]-selected[:,1],writes

    for entries in batches:
        tokens=torch.tensor([e[0] for e in entries],device='cuda')
        pos=torch.tensor([e[1] for e in entries],device='cuda')
        readpos=torch.tensor([e[4] for e in entries],device='cuda')
        answers=torch.tensor([e[2] for e in entries],device='cuda')
        batch=torch.arange(len(entries),device='cuda')
        initial=F.rms_norm(model.transformer.wte(tokens),(model.config.n_embd,)).float()
        subject=initial[batch,pos].double();projection=subject@unit
        orthogonal=subject-projection[:,None]*unit
        scale=((subject.square().sum(1)-target_projection**2)/orthogonal.square().sum(1)).sqrt()
        removed=initial.clone();removed[batch,pos]=(target_projection*unit+scale[:,None]*orthogonal).float()
        raw,x0,first,pb,_=graph._capture(model,initial,torch,F)
        _,_,_,pr,_=graph._capture(model,removed,torch,F);counts['prefix_calls']+=2
        edited=raw.clone();edited[batch,pos]+=sum((y-x)[batch,pos] for x,y in zip(pb,pr)).float()
        base,background=suffix(raw,x0,first,readpos,answers)
        true=graph._suffix_margin(model,edited,x0,first,readpos,answers,torch,F);counts['suffix_calls']+=1
        frozen,_=suffix(edited,x0,first,readpos,answers,background)
        restored,_=suffix(edited,x0,first,readpos,answers,background,mode='native12')
        folded,_=suffix(edited,x0,first,readpos,answers,background,mode='folded')
        replayed,_=suffix(raw,x0,first,readpos,answers,background,mode='folded')
        replay.append(float((replayed-base).abs().max()))
        for i,e in enumerate(entries):
            c=cells.setdefault(e[3],dict(target=[],prediction=[],native_margin=[],frozen_prediction=[],restored_prediction=[]))
            c['native_margin'].append(float(base[i]));c['target'].append(float(base[i]-true[i]));c['prediction'].append(float(base[i]-folded[i]));c['frozen_prediction'].append(float(base[i]-frozen[i]));c['restored_prediction'].append(float(base[i]-restored[i]))
        print('batch',len(entries),'baseline replay',replay[-1],flush=True)
    for c in cells.values():
        y=np.asarray(c['target']);p=np.asarray(c['prediction']);yn=np.linalg.norm(y);pn=np.linalg.norm(p)
        c['frozen_relative_error']=float(np.linalg.norm(np.array(c['frozen_prediction'])-y)/max(yn,1e-30))
        c['restored_relative_error']=float(np.linalg.norm(np.array(c['restored_prediction'])-y)/max(yn,1e-30))
        c.update(native_accuracy=float(np.mean(np.asarray(c['native_margin'])>0)),relative_l2=float(np.linalg.norm(p-y)/max(yn,1e-30)),
                 cosine=float(p@y/max(yn*pn,1e-30)),target_rms=float(np.sqrt(np.mean(y*y))),
                 absolute_error_rms=float(np.sqrt(np.mean((p-y)**2))))
    a=max(replay)<=1e-4 and counts==dict(prefix_calls=8,suffix_calls=24)
    b=a and all(c['restored_relative_error']<=.5*c['frozen_relative_error'] for k,c in cells.items() if k.startswith('post'))
    c=a and all(c['relative_l2']<=.10 for c in cells.values())
    result=dict(created_utc=datetime.now(timezone.utc).isoformat(),plan=plan,counts=counts,cells=cells,
                predictions=dict(zip(PREDICTIONS,[bool(a),bool(b),bool(c)])),
                max_baseline_replay_absolute_error=max(replay),wall_seconds=time.perf_counter()-tic,
                scope='Opened position prompts: causal attention12 restoration, dense MLP suffix; six exact attention-write backgrounds plus prefix/first-value ports retained. No computational circuit adoption; template-position transfer tested with capability gate.')
    payload=json.dumps(result,indent=2)+'\n';guard_write(len(payload.encode()),str(OUT.parent),'subject freeze')
    OUT.write_text(payload);print(json.dumps(result['predictions']));assert a


if __name__=='__main__':main()
