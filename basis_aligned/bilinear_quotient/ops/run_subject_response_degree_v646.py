#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_exact_replay pred_b_effect_prediction pred_c_capable_position_transfer
"""Finite-response quadratic product necessity with exact normalization.

48 new matched-position sequences, eight prefix calls and sixteen suffix calls, zero fits.
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
OUT=ROOT/'basis_aligned/bilinear_quotient/circuits/followups/subject_attention_freeze_v646_result.json'
PREDICTIONS=dict(pred_a_exact_replay='frozen-baseline and tensor-only chain replay max absolute error <=1e-4',
 pred_b_effect_prediction='quadratic omission conditional effect relativeL2 <=.05 every cell',
 pred_c_capable_position_transfer='quadratic omission full native effect relativeL2 <=.10 every cell')


def main():
    plan=dict(prefix_calls=8,suffix_calls=16,sequences_per_arm=48,fits=0,backwards=0,
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
    from conditional_mlp_chain import execute, response
    blocks=[dict(left=b.mlp.Left.weight.detach().clone(),right=b.mlp.Right.weight.detach().clone(),
                 down=b.mlp.Down.weight.detach().clone(),bias=b.mlp.Down_bias.detach().clone(),
                 lambdas=b.lambdas.detach().clone()) for b in model.transformer.h[12:18]]
    pair_readout=model.lm_head.weight[[318,389]].detach().clone()
    chain_checks=[];response_checks=[];context={};last_response={}
    double_blocks=[{k:v.double() for k,v in b.items()} for b in blocks]
    parameter_values=sum(t.numel() for b in blocks for t in b.values())+pair_readout.numel()
    cells={};replay=[];counts=dict(prefix_calls=0,suffix_calls=0)

    def suffix(raw,x0,first,pos,answers,frozen=None):
        counts['suffix_calls']+=1;x=raw;writes={}
        for layer in range(11,18):
            block=model.transformer.h[layer]
            if layer>11:x=block.lambdas[0]*x+block.lambdas[1]*x0
            if frozen is not None and layer in frozen:
                a=frozen[layer]
            else:
                a,first=block.attn(F.rms_norm(x,(x.shape[-1],)),first)
            x=x+a;x=x+block.mlp(F.rms_norm(x,(x.shape[-1],)))
            if layer==11:start=x[torch.arange(len(x),device=x.device),pos].clone()
            if layer>=12:writes[layer]=a
        batch=torch.arange(len(x),device=x.device)
        logits=30*torch.tanh(model.lm_head(F.rms_norm(x[batch,pos],(x.shape[-1],)))/30)
        selected=logits.gather(1,answers).double()
        conditional,_=execute(blocks,start,x0[batch,pos],
                              [writes[l][batch,pos] for l in range(12,18)],pair_readout,
                              torch.finfo(torch.float32).eps)
        direction=torch.where(answers[:,0]==318,1.,-1.)
        margin=(conditional[:,0]-conditional[:,1]).double()*direction
        reference=selected[:,0]-selected[:,1]
        chain_checks.append(float((margin-reference).abs().max()))
        if frozen is None:
            context['base_start']=start.double()
        else:
            ports=[writes[l][batch,pos].double() for l in range(12,18)]
            base_response,exact_response,_=response(double_blocks,context['base_start'],start.double(),
                x0[batch,pos].double(),ports,pair_readout.double(),torch.finfo(torch.float32).eps,True)
            _,omitted_response,_=response(double_blocks,context['base_start'],start.double(),
                x0[batch,pos].double(),ports,pair_readout.double(),torch.finfo(torch.float32).eps,False)
            exact_margin=(exact_response[:,0]-exact_response[:,1])*direction
            response_checks.append(float((exact_margin-reference).abs().max()))
            last_response['omitted_margin']=(omitted_response[:,0]-omitted_response[:,1])*direction
        return margin,writes

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
        omitted=last_response['omitted_margin'].clone()
        replayed,_=suffix(raw,x0,first,readpos,answers,background)
        replay.append(float((replayed-base).abs().max()))
        for i,e in enumerate(entries):
            c=cells.setdefault(e[3],dict(target=[],prediction=[],native_margin=[],conditional_target=[]))
            c['native_margin'].append(float(base[i]));c['target'].append(float(base[i]-true[i]));c['prediction'].append(float(base[i]-omitted[i]));c['conditional_target'].append(float(base[i]-frozen[i]))
        print('batch',len(entries),'baseline replay',replay[-1],flush=True)
    for c in cells.values():
        y=np.asarray(c['target']);p=np.asarray(c['prediction']);yn=np.linalg.norm(y);pn=np.linalg.norm(p)
        conditional=np.asarray(c['conditional_target'])
        c['conditional_relative_error']=float(np.linalg.norm(p-conditional)/max(np.linalg.norm(conditional),1e-30))
        c.update(native_accuracy=float(np.mean(np.asarray(c['native_margin'])>0)),relative_l2=float(np.linalg.norm(p-y)/max(yn,1e-30)),
                 cosine=float(p@y/max(yn*pn,1e-30)),target_rms=float(np.sqrt(np.mean(y*y))),
                 absolute_error_rms=float(np.sqrt(np.mean((p-y)**2))))
    a=max(replay+chain_checks+response_checks)<=1e-4 and counts==dict(prefix_calls=8,suffix_calls=16)
    b=a and all(c['conditional_relative_error']<=.05 for c in cells.values())
    c=a and all(c['relative_l2']<=.10 for c in cells.values())
    result=dict(created_utc=datetime.now(timezone.utc).isoformat(),plan=plan,counts=counts,cells=cells,
                predictions=dict(zip(PREDICTIONS,[bool(a),bool(b),bool(c)])),
                max_baseline_replay_absolute_error=max(replay),max_chain_replay_absolute_error=max(chain_checks),max_response_replay_absolute_error=max(response_checks),
                parameter_values=parameter_values,background_values_per_example=8*model.config.n_embd,wall_seconds=time.perf_counter()-tic,
                scope='Quadratic response omission with exact edited RMS; opened48rows, explicit baseline backgrounds. No fit or complete circuit adoption.')
    payload=json.dumps(result,indent=2)+'\n';guard_write(len(payload.encode()),str(OUT.parent),'subject freeze')
    OUT.write_text(payload);print(json.dumps(result['predictions']));assert a


if __name__=='__main__':main()
