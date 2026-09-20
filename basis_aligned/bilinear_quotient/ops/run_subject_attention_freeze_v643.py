#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_exact_replay pred_b_effect_prediction pred_c_direction_prediction
"""Causal test of v642: omit attention12–17 responses, retain native backgrounds.

160 opened sequences, six prefix calls and twelve suffix calls, zero fits.
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
OUT=ROOT/'basis_aligned/bilinear_quotient/circuits/followups/subject_attention_freeze_v643_result.json'
PREDICTIONS=dict(pred_a_exact_replay='frozen-baseline replay max absolute error <=1e-4',
 pred_b_effect_prediction='effect relativeL2 <=.10 in every direction/template cell',
 pred_c_direction_prediction='effect cosine >=.99 in every cell')


def main():
    plan=dict(prefix_calls=6,suffix_calls=12,sequences_per_arm=160,fits=0,backwards=0,
              frozen_attention_layers=list(range(12,18)),native_background_ports=6,
              execution_policy='managed_queue_only',predictions=PREDICTIONS)
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps(plan));return
    import torch
    import torch.nn.functional as F
    import numpy as np
    import circuit_fast_screen_producer as producer
    import circuit_fast_screen_candidate_subject_number_embedding_context_crossed_corrected_v1 as original
    import circuit_fast_screen_candidate_subject_number_rank1_fresh_confirmation as fresh
    import subject_number_sparse_graph_token_extraction_v1 as graph
    from disk_guard import guard_write
    if OUT.exists():raise FileExistsError(OUT)
    torch.set_grad_enabled(False);torch.set_num_threads(4);tic=time.perf_counter()
    model=producer.Bilin18TorchBackend.load('cuda').model
    decoder=json.loads((POLY/'SUBJECT_NUMBER_EMBEDDING_DECODER_FROZEN_V1_ARTIFACT.json').read_text())['frozen_decoder']
    axis=torch.tensor(decoder['axis'],device='cuda',dtype=torch.float64)
    unit=axis/axis.norm();target_projection=float(decoder['threshold'])/float(axis.norm())
    rows=original.build_rows();batches=[]
    for start in (0,64):
        batches.append([(r['token_ids'],r['subject_position'],
                        [r['native_answer_id'],389 if r['native_answer_id']==318 else 318],
                        'original|'+r['number']+'|'+r['template_id']) for r in rows[start:start+64]])
    entries=[]
    for r in fresh.build_rows():
        e=r['endpoints']['recipient']
        entries.append((e['ids'],len(e['ids'])-1,[e['answer_id'],e['foil_id']],
                       'historical_fresh|'+r['direction_id']+'|'+r['template_id']))
    batches.append(entries);assert sum(map(len,batches))==160
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
            if layer>=12:writes[layer]=a
        batch=torch.arange(len(x),device=x.device)
        logits=30*torch.tanh(model.lm_head(F.rms_norm(x[batch,pos],(x.shape[-1],)))/30)
        selected=logits.gather(1,answers).double()
        return selected[:,0]-selected[:,1],writes

    for entries in batches:
        tokens=torch.tensor([e[0] for e in entries],device='cuda')
        pos=torch.tensor([e[1] for e in entries],device='cuda')
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
        base,background=suffix(raw,x0,first,pos,answers)
        true=graph._suffix_margin(model,edited,x0,first,pos,answers,torch,F);counts['suffix_calls']+=1
        frozen,_=suffix(edited,x0,first,pos,answers,background)
        replayed,_=suffix(raw,x0,first,pos,answers,background)
        replay.append(float((replayed-base).abs().max()))
        for i,e in enumerate(entries):
            c=cells.setdefault(e[3],dict(target=[],prediction=[]))
            c['target'].append(float(base[i]-true[i]));c['prediction'].append(float(base[i]-frozen[i]))
        print('batch',len(entries),'baseline replay',replay[-1],flush=True)
    for c in cells.values():
        y=np.asarray(c['target']);p=np.asarray(c['prediction']);yn=np.linalg.norm(y);pn=np.linalg.norm(p)
        c.update(relative_l2=float(np.linalg.norm(p-y)/max(yn,1e-30)),
                 cosine=float(p@y/max(yn*pn,1e-30)),target_rms=float(np.sqrt(np.mean(y*y))),
                 absolute_error_rms=float(np.sqrt(np.mean((p-y)**2))))
    a=max(replay)<=1e-4 and counts==dict(prefix_calls=6,suffix_calls=12)
    b=a and all(c['relative_l2']<=.1 for c in cells.values())
    c=a and all(c['cosine']>=.99 for c in cells.values())
    result=dict(created_utc=datetime.now(timezone.utc).isoformat(),plan=plan,counts=counts,cells=cells,
                predictions=dict(zip(PREDICTIONS,[bool(a),bool(b),bool(c)])),
                max_baseline_replay_absolute_error=max(replay),wall_seconds=time.perf_counter()-tic,
                scope='Opened, same-task conditional effect test; six exact attention-write backgrounds plus prefix/first-value ports retained. No fresh transfer or computational circuit adoption.')
    payload=json.dumps(result,indent=2)+'\n';guard_write(len(payload.encode()),str(OUT.parent),'subject freeze')
    OUT.write_text(payload);print(json.dumps(result['predictions']));assert a


if __name__=='__main__':main()
