#!/usr/bin/env python3
"""Fixed individual QK-factor middle-match screen on opened native cases.
pred_a_mechanical: native joint/saved score and factor product <=1e-9/1e-10,
same tokens/metadata/masks, finite; six live primitive controls pass.
pred_b_factor1: both mismatch RMS<=.10 of smaller matched RMS in every group.
pred_c_factor2: same fixed bar for the other factor, no winner selection.
pred_d_joint: known product middle-match gate replays in every group.
All768 cases,32worlds,6orders,4cases,B4FP64,1800s,256MiB/tensor.
Screen only; no native weights removed, no factor-removal identification.
"""
# BQGATE: EXPERIMENT pred_a_mechanical pred_b_factor1 pred_c_factor2 pred_d_joint
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time

ROOT=Path('/workspace/tensor_language');POLY=ROOT/'basis_aligned/polynomial_causal';SOURCE=Path(__file__)
OUT=POLY/'MATCHER_FACTOR_TRUTH_TABLE_V1_RESULT.json';ROWS=POLY/'MATCHER_FACTOR_TRUTH_TABLE_V1_ROWS.pt'
INPUT=POLY/'SUFFIX_JOIN_MIDDLE_MATCH_V1_ROWS.pt';CHECKPOINT=ROOT/'runs_hop/attn4-rms-seed0/model.pt'
BOUND=[POLY/n for n in ('MATCHER_FACTOR_TRUTH_TABLE_V1_PREREGISTRATION.md','matcher_factor_truth_table_reference.py',
    'suffix_join_middle_match_reference.py','suffix_join_writer_reference.py','causal_suffix_join_reference.py',
    'field_intervention_metrics.py')]+[ROOT/'model.py',ROOT/'deep_model.py']
EXPECTED='b9595845a43cd8fac835343b29135906b8c1731097df24a51957b60ea3c01cf2';EXPECTED_INPUT='dde40c697c3f34eb2023425b4dc02aa1b95fd9247c960d864c39e17b270ae265'


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    sys.path[:0]=[str(ROOT),str(POLY)];assert hashlib.sha256(b''.join(p.read_bytes() for p in BOUND)).hexdigest()==EXPECTED
    import torch
    import matcher_factor_truth_table_reference as F
    import suffix_join_middle_match_reference as R
    import field_intervention_metrics as M
    torch.set_num_threads(2);checks=F.controls();assert checks['passed']
    assert digest(INPUT)==EXPECTED_INPUT;previous=torch.load(INPUT,map_location='cpu',weights_only=True);data=R.populations()
    for pop,(tokens,masks,meta) in data.items():
        assert torch.equal(tokens,previous[pop]['tokens']) and meta==previous[pop]['metadata']
        assert bool((masks.sum((-1,-2))==4).all())
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps({'dryrun':True,'controls':checks,'cases':768,'checkpoint_opened':False}));return
    from hop_ablate import load
    os.chdir(ROOT);signal.alarm(1800);started=time.perf_counter();assert not OUT.exists() and not ROWS.exists()
    assert digest(CHECKPOINT)=='c9388cc8b1ba7e95c6a043cbf4d01592987447867a50781d3babe31ce3f8358b'
    torch.backends.cuda.matmul.allow_tf32=False
    model,_=load('attn4-rms-seed0');model=model.to(device='cuda',dtype=torch.float64).eval()
    audits=[];results={};saved={};finite=True
    with torch.inference_mode():
        for pop,(tokens,masks,meta) in data.items():
            values={'factor1':[],'factor2':[],'joint':[]}
            for i in range(0,len(tokens),4):
                assert [m['case'] for m in meta[i:i+4]]==list(R.CASES)
                head=1 if meta[i]['orientation']=='B2_later' else 2
                factors,joint=F.factors(model,tokens[i:i+4].cuda(),masks[i:i+4].cuda(),head)
                audits.append(M.correspondence(factors[0]*factors[1],joint))
                for name,value in zip(values,(*factors,joint)):
                    finite &= bool(torch.isfinite(value).all());values[name].append(value.cpu().clone())
            arrays={name:torch.cat(value) for name,value in values.items()};audits.append(M.correspondence(arrays['joint'],previous[pop]['joint_score_cells']))
            results[pop]={}
            for orientation in ('B2_later','B3_later'):
                selected=torch.tensor([m['orientation']==orientation for m in meta]);cases=[m['case'] for m in meta if m['orientation']==orientation]
                results[pop][orientation]={name:F.truth_table(value[selected],cases) for name,value in arrays.items()}
            saved[pop]={'tokens':tokens,'metadata':meta,'factors':arrays}
            print(json.dumps({'population':pop,'groups':results[pop]}),flush=True)
    all_groups=[g for p in results.values() for g in p.values()]
    predictions={'pred_a_mechanical':finite and all(a['passed'] for a in audits),
        'pred_b_factor1':all(g['factor1']['individual_predicate_nominated'] for g in all_groups),
        'pred_c_factor2':all(g['factor2']['individual_predicate_nominated'] for g in all_groups),
        'pred_d_joint':all(g['joint']['individual_predicate_nominated'] for g in all_groups)}
    torch.save(saved,ROWS)
    result={'experiment':'matcher_factor_truth_table_v1','scope':'opened semantic factor screen, not a causal factor replacement',
        'predictions':predictions,'controls':checks,'populations':results,'native_score_max_abs':max(a['max_abs'] for a in audits),
        'native_score_max_relative_rms':max(a['relative_rms'] for a in audits),'independent_worlds':32,'cases':768,
        'native_parameters_retained':sum(p.numel() for p in model.parameters()),'native_coefficients_removed':0,
        'runner_sha256':digest(SOURCE),'bound_sha256':EXPECTED,'input_sha256':digest(INPUT),'rows_sha256':digest(ROWS),
        'wall_seconds':time.perf_counter()-started,
        'terminal':'instrument_invalid' if not predictions['pred_a_mechanical'] or not predictions['pred_d_joint'] else
        ('individual_matcher_nominated' if predictions['pred_b_factor1'] or predictions['pred_c_factor2'] else 'no_global_individual_matcher')}
    OUT.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ('terminal','predictions','native_score_max_abs','wall_seconds')}),flush=True)


if __name__=='__main__':main()
