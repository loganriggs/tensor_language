#!/usr/bin/env python3
"""Test the L1 previous-key path feeding the L2 backward join payload.

pred_a_instrument: exact single/conditional-joint replay1e-9, accuracy>=.8, cut>=.25.
pred_b_origin: fixed path recovery>=.8, exact complement recovery<=.2.
pred_c_distribution: full native KL mean<=1e-3,p99<=1e-2 all/query, forward live/cut.
pred_d_effects: relative centered RMS<=.01, absolute1e-8 below target1e-6.
Null rejects previous-key subpath sufficiency; no source/head or fitting sweep.
32worlds,2arrangements,8queries,9arms,B4FP64,1800s,<256MiB/tensor;managedGPU.
All400640 native coefficients and live routing/normalizers remain charged.
"""
# BQGATE: EXPERIMENT pred_a_instrument pred_b_origin pred_c_distribution pred_d_effects
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time

ROOT=Path('/workspace/tensor_language'); POLY=ROOT/'basis_aligned/polynomial_causal'; SOURCE=Path(__file__)
OUT=POLY/'JOIN_ORIGIN_WRITER_V1_RESULT.json'; ROWS=POLY/'JOIN_ORIGIN_WRITER_V1_ROWS.pt'
CHECKPOINT=ROOT/'runs_hop/attn4-rms-seed0/model.pt'
EXPECTED_CHECKPOINT='c9388cc8b1ba7e95c6a043cbf4d01592987447867a50781d3babe31ce3f8358b'
BOUND=[POLY/'join_origin_writer_reference.py',POLY/'join_value_producer_reference.py',POLY/'join_contribution_context_reference.py',
       POLY/'JOIN_ORIGIN_WRITER_V1_PREREGISTRATION.md',ROOT/'basis_aligned/bilinear_quotient/ops/run_equality_router_v1.py']
EXPECTED='8a73b6e43146c739beee9694a2969880adcf52bcc288bd4e7e815b42c5c3c7ed'


def bound_hash(): return hashlib.sha256(b''.join(p.read_bytes() for p in BOUND)).hexdigest()


def run(torch,R,score,checks):
    from hop_ablate import load
    os.chdir(ROOT);signal.alarm(1800);started=time.perf_counter()
    assert not OUT.exists() and not ROWS.exists() and score.digest(CHECKPOINT)==EXPECTED_CHECKPOINT
    torch.backends.cuda.matmul.allow_tf32=False
    model,_=load('attn4-rms-seed0');model=model.to(device='cuda',dtype=torch.float64).eval()
    results={};rows={};replay=0.;finite=True
    with torch.inference_mode():
        for pop,worlds in R.populations(seeds=(19909,19910)).items():
            outputs={};metadata=[];saved_tokens=[]
            for w in worlds:
                masks={j:m.cuda() for j,m in w['masks'].items()};heads=w['heads']
                back=next(j for j in heads if heads[j]==2);forward=1-back
                tokall=w['recipient'].cuda();saved_tokens.append(w['recipient'])
                for i in range(0,8,4):
                    tok=tokall[i:i+4];own=R.contributions(model,tok,masks,heads);path=R.origin_write(model,tok,masks,heads)
                    zero=torch.zeros_like(own[forward]);complement={back:own[back]-path[back]}
                    arms={'native':((),None),'cut_b':((back,),None),'restore_p':((back,),path),
                          'restore_complement':((back,),complement),'restore_all':((back,),own),
                          'cut_f':((forward,),None),'cut_both':((back,forward),None),
                          'restore_p_joint':((back,forward),{back:path[back],forward:zero}),
                          'restore_all_joint':((back,forward),{back:own[back],forward:zero})}
                    batch={}
                    for arm,(selected,writes) in arms.items():
                        with R.intervene(model,masks,heads,selected,writes):actual=model(tok)
                        finite &= bool(torch.isfinite(actual).all());batch[arm]=actual
                        outputs.setdefault(arm,[]).append(actual.cpu())
                    replay=max(replay,float((batch['restore_all']-batch['native']).abs().max()),
                               float((batch['restore_all_joint']-batch['cut_f']).abs().max()))
                metadata.extend({'world':w['world'],'arrangement':w['arrangement'],'query':int(q),'backward_query':back,
                                 'hop':int(h),'answer':int(a)} for q,h,a in zip(w['query'],w['hops'],w['answers']))
            logits={a:torch.cat(v) for a,v in outputs.items()};native=logits['native']
            answers=torch.tensor([m['answer'] for m in metadata])
            gold={a:v[:,-1].softmax(-1).gather(-1,answers[:,None]).squeeze(-1) for a,v in logits.items()}
            grouped={}
            for arr in range(2):
                select=torch.tensor([m['arrangement']==arr for m in metadata])
                group=torch.tensor([m['arrangement']==arr and m['query']==m['backward_query'] and m['hop']==3 for m in metadata])
                loss=float((gold['native']-gold['cut_b'])[group].mean())
                recovery={a:float((gold[a]-gold['cut_b'])[group].mean())/max(loss,1e-30) for a in ('restore_p','restore_complement')}
                cases={}
                for case,restore,target,cut in (('single','restore_p','native','cut_b'),('forward_removed','restore_p_joint','cut_f','cut_both')):
                    a,b,c=logits[restore][select],logits[target][select],logits[cut][select]
                    cases[case]={'distribution':{'all':score.distribution(a,b,torch),'query':score.distribution(a[:,-1],b[:,-1],torch)},
                                 'effects':{'all':score.effect_error(a-c,b-c,torch),'query':score.effect_error((a-c)[:,-1],(b-c)[:,-1],torch)}}
                controls={}
                for q in range(2):
                    for hop in range(4):
                        g=torch.tensor([m['arrangement']==arr and m['query']==q and m['hop']==hop for m in metadata])
                        controls[str(q)+'_'+str(hop)]={'native_accuracy':float((native[:,-1].argmax(-1)==answers)[g].double().mean()),
                                                      'cut_loss':float((gold['native']-gold['cut_b'])[g].mean()),
                                                      'restore_p_minus_native_gold':float((gold['restore_p']-gold['native'])[g].mean())}
                grouped[str(arr)]={'native_accuracy':float((native[:,-1].argmax(-1)==answers)[group].double().mean()),
                                   'cut_loss':loss,'recovery':recovery,'cases':cases,'query_hop_controls':controls}
            results[pop]={'arrangements':grouped}
            lp=native.log_softmax(-1)
            rows[pop]={'tokens':torch.cat(saved_tokens),'metadata':metadata,'query_logits':{a:v[:,-1].clone() for a,v in logits.items()},
                       'token_kl':{a:(lp.exp()*(lp-v.log_softmax(-1))).sum(-1) for a,v in logits.items()}}
            print(json.dumps({'population':pop,'arrangements':grouped}),flush=True)
    groups=[g for p in results.values() for g in p['arrangements'].values()]
    pred_a=checks['passed'] and finite and replay<=1e-9 and all(g['native_accuracy']>=.8 and g['cut_loss']>=.25 for g in groups)
    pred_b=pred_a and all(g['recovery']['restore_p']>=.8 and g['recovery']['restore_complement']<=.2 for g in groups)
    pred_c=pred_a and all(v['passed'] for g in groups for c in g['cases'].values() for v in c['distribution'].values())
    pred_d=pred_a and all(v['passed'] for g in groups for c in g['cases'].values() for v in c['effects'].values())
    torch.save(rows,ROWS)
    receipt={'experiment':'join_origin_writer_v1','runner_sha256':score.digest(SOURCE),'bound_sha256':EXPECTED,
             'checkpoint_sha256':EXPECTED_CHECKPOINT,'controls':checks,'native_parameters_retained':sum(p.numel() for p in model.parameters()),
             'self_replay_max_abs_logit':replay,'finite':finite,'populations':results,'rows':str(ROWS.relative_to(ROOT)),
             'rows_sha256':score.digest(ROWS),'independent_worlds':32,'query_variants':512,
             'predictions':{'pred_a_instrument':bool(pred_a),'pred_b_origin':bool(pred_b),'pred_c_distribution':bool(pred_c),'pred_d_effects':bool(pred_d)},
             'terminal':'instrument_invalid' if not pred_a else ('previous_key_origin_path_supported' if pred_b and pred_c and pred_d else 'previous_key_origin_path_not_quantitatively_sufficient'),
             'wall_seconds':time.perf_counter()-started}
    OUT.write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({k:receipt[k] for k in ('terminal','predictions','wall_seconds')}),flush=True)


def main():
    sys.path[:0]=[str(ROOT),str(POLY)];assert bound_hash()==EXPECTED
    import torch
    import join_origin_writer_reference as R
    import run_equality_router_v1 as score
    torch.set_num_threads(2);checks=R.controls();assert checks['passed'],checks
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps({'dryrun':True,'controls':checks,'checkpoint_opened':False}));return
    run(torch,R,score,checks)


if __name__=='__main__':main()
