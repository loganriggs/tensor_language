#!/usr/bin/env python3
"""Fixed prefix x summary mediation of record-order output dependence.
pred_a_mechanical: allfour fulloutput native cells and saved endpoints <=1e-9/1e-10,
identical local inputs, earlier summary effects<=1e-9, finite and live controls.
pred_b_summary_mediation: every group donorKL mean<=.001,p99<=.01 and centered
summary-only effect error<=.01 relative to native order effect, floor1e-6.
pred_c_mixed: compiled/native mixed term<=1e-9/1e-10; never assume additivity.
Null: summary-only fails; prefix-only remains diagnostic, no field expansion.
3072 opened pairs,16worlds,B8FP64,1800s,256MiB/tensor,387968 retained constants.
"""
# BQGATE: EXPERIMENT pred_a_mechanical pred_b_summary_mediation pred_c_mixed
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time

ROOT=Path('/workspace/tensor_language');POLY=ROOT/'basis_aligned/polynomial_causal';SOURCE=Path(__file__)
OUT=POLY/'SUMMARY_LAYOUT_MEDIATION_V1_RESULT.json';ROWS=POLY/'SUMMARY_LAYOUT_MEDIATION_V1_ROWS.pt'
INPUT=POLY/'RECORD_ORDER_OUTPUT_INVARIANCE_V1_ROWS.pt';PACKAGE=POLY/'EXACT_SOURCE_EDIT_V1_PROGRAM.pt'
CHECKPOINT=ROOT/'runs_hop/attn4-rms-seed0/model.pt'
BOUND=[POLY/n for n in ('SUMMARY_LAYOUT_MEDIATION_V1_PREREGISTRATION.md','summary_layout_mediation_reference.py',
    'query_initializer_factorization_reference.py','local_query_initializer_reference.py','record_order_reference.py',
    'exact_source_edit_reference.py','field_intervention_metrics.py')]+[ROOT/'model.py',ROOT/'deep_model.py']
EXPECTED='8e0d3f81387ecfd0b134eaaabe5caa138ed46da082e46a38c76690ea932619d1'


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    sys.path[:0]=[str(ROOT),str(POLY)];assert hashlib.sha256(b''.join(p.read_bytes() for p in BOUND)).hexdigest()==EXPECTED
    import torch
    import summary_layout_mediation_reference as S
    import query_initializer_factorization_reference as Q
    import record_order_reference as R
    import exact_source_edit_reference as E
    import field_intervention_metrics as M
    torch.set_num_threads(2);checks=S.controls();assert checks['passed']
    assert digest(INPUT)=='beade7e639dcdb2752a7dfb7419866a2d3339d457417841e67902478ee4506c6'
    data=torch.load(INPUT,map_location='cpu',weights_only=True)
    for block in data.values():
        for perm in R.permutations().values():R.reorder(block['native_tokens'],perm)
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps({'dryrun':True,'controls':checks,'pairs':3072,'checkpoint_opened':False}));return
    from hop_ablate import load
    os.chdir(ROOT);signal.alarm(1800);started=time.perf_counter();assert not OUT.exists() and not ROWS.exists()
    assert digest(PACKAGE)=='e69bff6562c48bd1b59366f118157720734d353626b0a1576626bdd437a1421a'
    assert digest(CHECKPOINT)=='c9388cc8b1ba7e95c6a043cbf4d01592987447867a50781d3babe31ce3f8358b'
    torch.backends.cuda.matmul.allow_tf32=False
    program=E.load_package(torch.load(PACKAGE,map_location='cpu',weights_only=True)).cuda().eval()
    model,_=load('attn4-rms-seed0');model=model.to(device='cuda',dtype=torch.float64).eval()
    audits=[];mixed_audits=[];results={};saved={};finite=True;earlier=0.;local_error=0.
    center=lambda x:x-x.mean(-1,keepdim=True)
    mixed=lambda d:d['11']-d['10']-d['01']+d['00']
    with torch.inference_mode():
        for pop,block in data.items():
            tokens=block['native_tokens'];metadata=block['metadata'];results[pop]={};saved[pop]={}
            for name,perm in R.permutations().items():
                donor=R.reorder(tokens,perm);values={k:[] for k in ('00','01','10','11','mixed')}
                for i in range(0,len(tokens),8):
                    endpoints=(tokens[i:i+8].cuda(),donor[i:i+8].cuda());compiled={};oracle={}
                    local_error=max(local_error,float((Q.local_state(program,endpoints[0])-Q.local_state(program,endpoints[1])).abs().max()))
                    for p,prefix in enumerate(endpoints):
                        for s,summary in enumerate(endpoints):
                            key=str(p)+str(s);compiled[key]=S.execute(program,prefix,summary);oracle[key]=S.native(model,prefix,summary)
                            audits.append(M.correspondence(compiled[key],oracle[key]));finite &= bool(torch.isfinite(compiled[key]).all())
                    for p in ('0','1'):earlier=max(earlier,float((compiled[p+'0'][:,:-1]-compiled[p+'1'][:,:-1]).abs().max()))
                    delta=mixed(compiled);mixed_audits.append(M.correspondence(delta,mixed(oracle)))
                    for key,value in {**compiled,'mixed':delta}.items():values[key].append(value[:,-1].cpu().clone())
                logits={k:torch.cat(v) for k,v in values.items()}
                audits.append(M.correspondence(logits['00'],block['query_logits']['native']))
                audits.append(M.correspondence(logits['11'],block['query_logits'][name]))
                groups={}
                for hop in range(4):
                    sel=torch.tensor([m['hop']==hop for m in metadata]);target=center(logits['11'][sel]-logits['00'][sel])
                    norm=float(target.square().mean().sqrt());den=max(norm,1e-6);arms={}
                    for key in ('01','10'):
                        error=center(logits[key][sel]-logits['11'][sel]);lp=logits['11'][sel].log_softmax(-1)
                        kl=(lp.exp()*(lp-logits[key][sel].log_softmax(-1))).sum(-1).clamp_min(0)
                        arms[key]={'mean_donor_query_kl':float(kl.mean()),'p99_donor_query_kl':float(torch.quantile(kl,.99)),
                            'relative_order_effect_error':float(error.square().mean().sqrt())/den}
                    primary=arms['01'];groups[str(hop)]={'pairs':int(sel.sum()),'native_order_effect_rms':norm,'arms':arms,
                        'mixed_over_native_effect_rms':float(center(logits['mixed'][sel]).square().mean().sqrt())/den,
                        'summary_mediation_passed':primary['mean_donor_query_kl']<=.001 and primary['p99_donor_query_kl']<=.01 and primary['relative_order_effect_error']<=.01}
                results[pop][name]=groups;saved[pop][name]={'metadata':metadata,'query_logits':logits}
                print(json.dumps({'population':pop,'permutation':name,'hop3':groups['3']}),flush=True)
    predictions={'pred_a_mechanical':checks['passed'] and finite and all(a['passed'] for a in audits) and earlier<=1e-9 and local_error==0,
        'pred_b_summary_mediation':all(g['summary_mediation_passed'] for p in results.values() for a in p.values() for g in a.values()),
        'pred_c_mixed':all(a['passed'] for a in mixed_audits)}
    torch.save(saved,ROWS)
    result={'experiment':'summary_layout_mediation_v1','scope':'opened first-layer path mediation; all opaque native weights retained',
        'predictions':predictions,'controls':checks,'populations':results,'output_oracle_max_abs':max(a['max_abs'] for a in audits),
        'output_oracle_max_relative_rms':max(a['relative_rms'] for a in audits),'mixed_oracle_max_abs':max(a['max_abs'] for a in mixed_audits),
        'earlier_summary_effect_max_abs':earlier,'local_state_max_abs':local_error,'independent_worlds':16,'pairs':3072,
        'independent_constants':program.independent_constant_count(),'native_coefficients_removed':0,
        'runner_sha256':digest(SOURCE),'bound_sha256':EXPECTED,'input_sha256':digest(INPUT),'rows_sha256':digest(ROWS),
        'wall_seconds':time.perf_counter()-started,'terminal':'instrument_invalid' if not predictions['pred_a_mechanical'] or not predictions['pred_c_mixed'] else
        ('summary_mediation_screen_passed' if predictions['pred_b_summary_mediation'] else 'summary_only_mediation_rejected')}
    OUT.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ('terminal','predictions','output_oracle_max_abs','wall_seconds')}),flush=True)


if __name__=='__main__':main()
