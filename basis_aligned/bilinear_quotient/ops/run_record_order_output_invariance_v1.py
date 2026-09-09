#!/usr/bin/env python3
"""Fixed record-order full-query test; opened cohort, no fit or new data claim.
pred_a_mechanical: native/export fulloutput max<=1e-9, relative<=1e-10,
finite, token function/suffix preserved and identity exact.
pred_b_function_only: every population/hop/permutation mean pairedJS<=.001.
pred_c_saved_replay: original query logits replay saved native outputs at1e-9/1e-10.
Null: any mean>.001 rejects .001 paired queryKL for an order-invariant model.
All3072 pairs,16 worlds,B8FP64,1800s,256MiB/tensor; no adapter rescue.
All387968 native constants charged, no structural reduction or adoption.
"""
# BQGATE: EXPERIMENT pred_a_mechanical pred_b_function_only pred_c_saved_replay
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time

ROOT=Path('/workspace/tensor_language');POLY=ROOT/'basis_aligned/polynomial_causal';SOURCE=Path(__file__)
OUT=POLY/'RECORD_ORDER_OUTPUT_INVARIANCE_V1_RESULT.json';ROWS=POLY/'RECORD_ORDER_OUTPUT_INVARIANCE_V1_ROWS.pt'
INPUT=POLY/'QUERY_INITIALIZER_FACTORIZATION_V1_ROWS.pt';PACKAGE=POLY/'EXACT_SOURCE_EDIT_V1_PROGRAM.pt'
CHECKPOINT=ROOT/'runs_hop/attn4-rms-seed0/model.pt'
BOUND=[POLY/n for n in ('RECORD_ORDER_OUTPUT_INVARIANCE_V1_PREREGISTRATION.md','record_order_reference.py',
    'entity_equivariance_reference.py','exact_source_edit_reference.py','field_intervention_metrics.py')]+[ROOT/'model.py',ROOT/'deep_model.py']
EXPECTED='2aa5db27b0d5968c3bcf2311a650560815c9fa67adb575fc5358304ee18c98c5'


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    sys.path[:0]=[str(ROOT),str(POLY)]
    assert hashlib.sha256(b''.join(p.read_bytes() for p in BOUND)).hexdigest()==EXPECTED
    import torch
    import record_order_reference as R
    import entity_equivariance_reference as J
    import exact_source_edit_reference as E
    import field_intervention_metrics as M
    torch.set_num_threads(2);checks=J.controls();assert checks['passed']
    assert digest(INPUT)=='899e48cc00c27b966c43dea9bf2406fcdcc6bdb5d7b14436ee661027e3870e7e'
    data=torch.load(INPUT,map_location='cpu',weights_only=True)['populations']
    for block in data.values():
        tokens=block['native_tokens'];assert torch.equal(R.reorder(tokens,torch.arange(24)),tokens)
        for perm in R.permutations().values():
            changed=R.reorder(tokens,perm);assert torch.equal(R.reorder(changed,perm.argsort()),tokens)
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps({'dryrun':True,'controls':checks,'pairs':3072,'checkpoint_opened':False}));return
    from hop_ablate import load
    os.chdir(ROOT);signal.alarm(1800);started=time.perf_counter();assert not OUT.exists() and not ROWS.exists()
    assert digest(PACKAGE)=='e69bff6562c48bd1b59366f118157720734d353626b0a1576626bdd437a1421a'
    assert digest(CHECKPOINT)=='c9388cc8b1ba7e95c6a043cbf4d01592987447867a50781d3babe31ce3f8358b'
    torch.backends.cuda.matmul.allow_tf32=False
    program=E.load_package(torch.load(PACKAGE,map_location='cpu',weights_only=True)).cuda().eval()
    model,_=load('attn4-rms-seed0');model=model.to(device='cuda',dtype=torch.float64).eval()
    audits=[];replays=[];results={};saved={};finite=True
    with torch.inference_mode():
        for pop,block in data.items():
            tokens=block['native_tokens'];logits={};groups={};metadata=block['metadata']
            arms={'native':tokens,**{name:R.reorder(tokens,perm) for name,perm in R.permutations().items()}}
            for name,arm in arms.items():
                values=[]
                for i in range(0,len(arm),8):
                    tok=arm[i:i+8].cuda();native=model(tok);exported=program.prepare(tok)['logits']
                    audits.append(M.correspondence(exported,native));finite &= bool(torch.isfinite(native).all())
                    values.append(native[:,-1].cpu().clone())
                logits[name]=torch.cat(values)
            replay=M.correspondence(logits['native'],block['query_logits']['native']);audits.append(replay);replays.append(replay)
            radii={}
            for name in R.permutations():
                radius,_=J.information_radius(logits['native'],logits[name]);radius=radius.clamp_min(0);radii[name]=radius
                groups[name]={}
                for hop in range(4):
                    selected=torch.tensor([m['hop']==hop for m in metadata]);v=radius[selected]
                    groups[name][str(hop)]={'pairs':int(selected.sum()),'mean_paired_query_kl_lower_bound':float(v.mean()),
                        'max_pair_bound':float(v.max()),'p95_pair_bound':float(torch.quantile(v,.95)),
                        'argmax_disagreements':int(((logits['native'].argmax(-1)!=logits[name].argmax(-1))&selected).sum()),
                        'not_ruled_out_by_pairwise_bound':bool(v.mean()<=.001)}
            results[pop]=groups;saved[pop]={'metadata':metadata,'query_logits':logits,'radii':radii,'native_tokens':tokens}
            print(json.dumps({'population':pop,'groups':groups}),flush=True)
    predictions={'pred_a_mechanical':finite and all(a['passed'] for a in audits),
        'pred_b_function_only':all(g['not_ruled_out_by_pairwise_bound'] for p in results.values() for a in p.values() for g in a.values()),
        'pred_c_saved_replay':all(a['passed'] for a in replays)}
    torch.save(saved,ROWS)
    result={'experiment':'record_order_output_invariance_v1','scope':'opened full-query paired invariance lower bound; no surrogate fit or adoption',
        'predictions':predictions,'controls':checks,'populations':results,'output_oracle_max_abs':max(a['max_abs'] for a in audits),
        'output_oracle_max_relative_rms':max(a['relative_rms'] for a in audits),'independent_worlds':16,'pairs':3072,
        'independent_constants':program.independent_constant_count(),'native_coefficients_removed':0,
        'runner_sha256':digest(SOURCE),'bound_sha256':EXPECTED,'input_sha256':digest(INPUT),'rows_sha256':digest(ROWS),
        'wall_seconds':time.perf_counter()-started,'terminal':'instrument_invalid' if not predictions['pred_a_mechanical'] else
        ('function_only_not_ruled_out_by_pairwise_bound' if predictions['pred_b_function_only'] else 'function_only_query_fidelity_ruled_out')}
    OUT.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ('terminal','predictions','output_oracle_max_abs','wall_seconds')}),flush=True)


if __name__=='__main__':main()
