#!/usr/bin/env python3
"""Audit exact low-order response algebra for disjoint normalized source states.

pred_a_instrument: structural/CPU controls, native suffix replay1e-9/relative1e-10.
pred_b_disjoint: withheld triple closure1e-9/relative1e-10; pair support; live>=1e-6.
pred_c_overlap: overlapping producer third centered RMS>=1e-6 each population.
pred_d_probability: log_softmax pair at unchanged finalquery has max>=1e-6.
Null requires auditing locality premises; this is not learned reduction/adoption.
32worlds,384queryvariants,16arms,B4FP64,1800s,<256MiB/tensor;managedGPU.
All400640 native coefficients and per-input response-generation cost retained.
"""
# BQGATE: EXPERIMENT pred_a_instrument pred_b_disjoint pred_c_overlap pred_d_probability
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time

ROOT=Path('/workspace/tensor_language');POLY=ROOT/'basis_aligned/polynomial_causal';SOURCE=Path(__file__)
OUT=POLY/'SOURCE_SUPPORT_INTERACTION_V1_RESULT.json';ROWS=POLY/'SOURCE_SUPPORT_INTERACTION_V1_ROWS.pt'
CHECKPOINT=ROOT/'runs_hop/attn4-rms-seed0/model.pt'
EXPECTED_CHECKPOINT='c9388cc8b1ba7e95c6a043cbf4d01592987447867a50781d3babe31ce3f8358b'
BOUND=[POLY/n for n in ('source_support_interaction_reference.py','join_contribution_context_reference.py',
                       'join_value_producer_reference.py','SOURCE_SUPPORT_INTERACTION_V1_PREREGISTRATION.md')]
EXPECTED='b2ef489e9b88ef43397145dd02f67e598b222c35c5dcaf71d4c983070f6a4f74'


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def bound_hash():return hashlib.sha256(b''.join(p.read_bytes() for p in BOUND)).hexdigest()


def relative(actual,target):
    return float((actual-target).square().mean().sqrt())/max(float(target.square().mean().sqrt()),1e-6)


def run(torch,R,J,P,checks):
    from hop_ablate import load
    os.chdir(ROOT);signal.alarm(1800);started=time.perf_counter()
    assert not OUT.exists() and not ROWS.exists() and digest(CHECKPOINT)==EXPECTED_CHECKPOINT
    torch.backends.cuda.matmul.allow_tf32=False
    model,_=load('attn4-rms-seed0');model=model.to(device='cuda',dtype=torch.float64).eval()
    results={};rows={};replay=0.;replay_relative=0.;finite=True
    with torch.inference_mode():
        for pop,worlds in R.populations().items():
            triple_error=triple_relative=pair_leak=pair_live=logprob_pair=0.;third_sq=0.;third_count=0
            query_values=[];overlap_query=[];token_rows=[];answers=[];perexample=[]
            for w in worlds:
                masks={j:m.cuda() for j,m in w['masks'].items()};heads=w['heads'];tokall=w['tokens'].cuda()
                supports={j:m.any(-1) for j,m in masks.items()}
                assert all(not bool((supports[i]&supports[j]).any()) for i,j in ((0,1),(0,2),(1,2)))
                assert not any(bool(m.triu(1).any()) for m in masks.values())
                token_rows.append(w['tokens']);answers.append(w['answers'])
                for i in range(0,12,4):
                    tok=tokall[i:i+4];base=model.embed(tok)
                    for layer in model.layers[:3]:base=layer(base)
                    suffix=lambda x:model.head(model.layers[-1](x))
                    writes=J.contributions(model,tok,masks,heads)
                    values=[]
                    # The all-three target is not evaluated until its prediction exists.
                    for subset in range(7):
                        changed=base-sum((writes[j] for j in range(3) if subset&(1<<j)),torch.zeros_like(base))
                        values.append(suffix(changed))
                    predicted=values[3]+values[5]+values[6]-values[1]-values[2]-values[4]+values[0]
                    values.append(suffix(base-sum(writes.values())))
                    terms=R.mobius(values)
                    triple_error=max(triple_error,float((predicted-values[7]).abs().max()))
                    triple_relative=max(triple_relative,relative(predicted,values[7]))
                    native=model(tok)
                    with J.intervene(model,masks,heads,(0,1,2)):native_cut=model(tok)
                    for actual,target in ((values[0],native),(values[7],native_cut)):
                        replay=max(replay,float((actual-target).abs().max()));replay_relative=max(replay_relative,relative(actual,target))
                    for a,b in ((0,1),(0,2),(1,2)):
                        subset=(1<<a)|(1<<b)
                        allowed=((supports[a][:,None]&supports[b][None,:])|(supports[b][:,None]&supports[a][None,:])).tril().any(-1)
                        pair_leak=max(pair_leak,float(terms[subset][:,~allowed].abs().max()))
                        pair_live=max(pair_live,float(terms[subset][:,allowed].abs().max()))
                        lp=values[subset][:,-1].log_softmax(-1)-values[1<<a][:,-1].log_softmax(-1)-values[1<<b][:,-1].log_softmax(-1)+values[0][:,-1].log_softmax(-1)
                        logprob_pair=max(logprob_pair,float(lp.abs().max()))
                    producers,_=P.producer_writes(model,tok,masks,heads)
                    overlap=R.evaluate(suffix,base,[-producers[1][k] for k in P.KINDS])
                    third=R.mobius(overlap)[7];centered=third-third.mean(-1,keepdim=True)
                    third_sq+=float(centered.square().sum());third_count+=centered.numel()
                    finite &= all(bool(torch.isfinite(v).all()) for v in values+overlap)
                    query_values.append(torch.stack([v[:,-1] for v in values],1).cpu())
                    overlap_query.append(torch.stack([v[:,-1] for v in overlap],1).cpu())
                    perexample.extend({'world':w['world'],'query_variant':i+j,
                                       'triple_max_abs':float((predicted[j]-values[7][j]).abs().max()),
                                       'overlap_third_rms':float(centered[j].square().mean().sqrt())} for j in range(len(tok)))
            q=torch.cat(query_values);ans=torch.cat(answers)
            results[pop]={'triple_max_abs_error':triple_error,'triple_max_relative_rms':triple_relative,
                          'pair_outside_support_max_abs':pair_leak,'allowed_pair_max_abs':pair_live,
                          'overlap_third_centered_rms':(third_sq/third_count)**.5,'logprob_finalquery_pair_max_abs':logprob_pair,
                          'native_query_accuracy':float((q[:,0].argmax(-1)==ans).double().mean())}
            rows[pop]={'tokens':torch.cat(token_rows),'answers':ans,'disjoint_query_logits':q,
                       'overlap_query_logits':torch.cat(overlap_query),'perexample':perexample}
            print(json.dumps({'population':pop,**results[pop]}),flush=True)
    pred_a=checks['passed'] and finite and replay<=1e-9 and replay_relative<=1e-10
    pred_b=pred_a and all(p['triple_max_abs_error']<=1e-9 and p['triple_max_relative_rms']<=1e-10 and p['pair_outside_support_max_abs']<=1e-9 and p['allowed_pair_max_abs']>=1e-6 for p in results.values())
    pred_c=pred_a and all(p['overlap_third_centered_rms']>=1e-6 for p in results.values())
    pred_d=pred_a and all(p['logprob_finalquery_pair_max_abs']>=1e-6 for p in results.values())
    torch.save(rows,ROWS)
    receipt={'experiment':'source_support_interaction_v1','runner_sha256':digest(SOURCE),'bound_sha256':EXPECTED,
             'checkpoint_sha256':EXPECTED_CHECKPOINT,'controls':checks,'native_parameters_retained':sum(p.numel() for p in model.parameters()),
             'native_suffix_replay_max_abs':replay,'native_suffix_replay_max_relative_rms':replay_relative,
             'populations':results,'rows':str(ROWS.relative_to(ROOT)),'rows_sha256':digest(ROWS),
             'predictions':{'pred_a_instrument':bool(pred_a),'pred_b_disjoint':bool(pred_b),'pred_c_overlap':bool(pred_c),'pred_d_probability':bool(pred_d)},
             'terminal':'instrument_invalid' if not pred_a else ('source_support_law_verified' if pred_b and pred_c and pred_d else 'source_support_law_or_control_failed'),
             'scope':'fixed-input native-background response algebra; not independent compact extraction','wall_seconds':time.perf_counter()-started}
    OUT.write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({k:receipt[k] for k in ('terminal','predictions','wall_seconds')}),flush=True)


def main():
    sys.path[:0]=[str(ROOT),str(POLY)];assert bound_hash()==EXPECTED
    import torch
    import source_support_interaction_reference as R
    import join_contribution_context_reference as J
    import join_value_producer_reference as P
    torch.set_num_threads(2);checks=R.controls();assert checks['passed'],checks
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps({'dryrun':True,'controls':checks,'checkpoint_opened':False}));return
    run(torch,R,J,P,checks)


if __name__=='__main__':main()
