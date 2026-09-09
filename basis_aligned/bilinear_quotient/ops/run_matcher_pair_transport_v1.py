#!/usr/bin/env python3
"""Matched-label QK-pair transfer at fixed native source cells, reused across hops.
pred_a_mechanical: allfull-output native/cut/replay/factorreuse checks<=1e-9/1e-10.
pred_b_pair_portability: joint meanKL<=.001,p99<=.01 for all-token/query and
centered querychange<=.01 of native route-removal RMS, floor1e-6, every group.
pred_c_composition: native/compiled mixed full-output effects<=1e-9/1e-10.
Null: matched-label pair transfer fails; no scalar/entity/head/role/hop rescue.
768 requests,32worlds,6orders,4hops,B4FP64,1800s,256MiB/tensor.
All387968 native export constants retained; exact intervention infrastructure only.
"""
# BQGATE: EXPERIMENT pred_a_mechanical pred_b_pair_portability pred_c_composition
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time

ROOT=Path('/workspace/tensor_language');POLY=ROOT/'basis_aligned/polynomial_causal';SOURCE=Path(__file__)
OUT=POLY/'MATCHER_PAIR_TRANSPORT_V1_RESULT.json';ROWS=POLY/'MATCHER_PAIR_TRANSPORT_V1_ROWS.pt'
INPUT=POLY/'MATCHER_FACTOR_TRUTH_TABLE_V1_ROWS.pt';PREVIOUS=POLY/'SUFFIX_JOIN_MIDDLE_MATCH_V1_ROWS.pt'
PACKAGE=POLY/'EXACT_SOURCE_EDIT_V1_PROGRAM.pt';CHECKPOINT=ROOT/'runs_hop/attn4-rms-seed0/model.pt'
BOUND=[POLY/n for n in ('MATCHER_PAIR_TRANSPORT_V1_PREREGISTRATION.md','matcher_pair_transport_reference.py',
    'matcher_factor_truth_table_reference.py','suffix_join_middle_match_reference.py','suffix_join_writer_reference.py',
    'causal_suffix_join_reference.py','exact_source_edit_reference.py','field_intervention_metrics.py')]+[ROOT/'model.py',ROOT/'deep_model.py']
EXPECTED='8775ec45b00685853b0de5db052d3efac815940ed0b3e1fcff0a33c2ff00bbb5'


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    sys.path[:0]=[str(ROOT),str(POLY)];assert hashlib.sha256(b''.join(p.read_bytes() for p in BOUND)).hexdigest()==EXPECTED
    import torch
    import matcher_pair_transport_reference as T
    import matcher_factor_truth_table_reference as F
    import suffix_join_middle_match_reference as R
    import exact_source_edit_reference as E
    import field_intervention_metrics as M
    torch.set_num_threads(2);checks=T.controls();assert checks['passed']
    assert digest(INPUT)=='d0dd9dd724a054e9a7915b0331d8b5df354004f1ad159b5dfe408edb884a62bf'
    assert digest(PREVIOUS)=='dde40c697c3f34eb2023425b4dc02aa1b95fd9247c960d864c39e17b270ae265'
    data=R.populations();previous=torch.load(INPUT,map_location='cpu',weights_only=True);old_native=torch.load(PREVIOUS,map_location='cpu',weights_only=True)
    for pop,(tokens,masks,metadata) in data.items():assert torch.equal(tokens,previous[pop]['tokens']) and metadata==previous[pop]['metadata']
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps({'dryrun':True,'controls':checks,'requests':768,'checkpoint_opened':False}));return
    from hop_ablate import load
    os.chdir(ROOT);signal.alarm(1800);started=time.perf_counter();assert not OUT.exists() and not ROWS.exists()
    assert digest(PACKAGE)=='e69bff6562c48bd1b59366f118157720734d353626b0a1576626bdd437a1421a'
    assert digest(CHECKPOINT)=='c9388cc8b1ba7e95c6a043cbf4d01592987447867a50781d3babe31ce3f8358b'
    torch.backends.cuda.matmul.allow_tf32=False
    program=E.load_package(torch.load(PACKAGE,map_location='cpu',weights_only=True)).cuda().eval()
    model,_=load('attn4-rms-seed0');model=model.to(device='cuda',dtype=torch.float64).eval()
    audits=[];cut_audits=[];mixed_audits=[];results={};saved={};finite=True
    center=lambda x:x-x.mean(-1,keepdim=True)
    mixed=lambda d:d['11']-d['10']-d['01']+d['00']
    with torch.inference_mode():
        for pop,(tokens,masks,metadata) in data.items():
            values={k:[] for k in ('00','10','01','11','cut','mixed')};kls={k:[] for k in ('10','01','11','cut')};meta=[]
            for i in range(0,len(tokens),4):
                assert [m['case'] for m in metadata[i:i+4]]==list(R.CASES)
                head=1 if metadata[i]['orientation']=='B2_later' else 2
                tok=tokens[i:i+1].expand(4,-1).clone();tok[:,-1]=torch.arange(25,29)
                donor=tokens[i+3:i+4].expand(4,-1).clone();donor[:,-1]=torch.arange(25,29)
                tok=tok.cuda();donor=donor.cuda();mask=masks[i:i+1].expand(4,-1,-1).cuda()
                old,old_joint=F.factors(program.background,tok,mask,head);new,new_joint=F.factors(program.background,donor,mask,head)
                for factors in (old,new):
                    for factor in factors:audits.append(M.correspondence(factor,factor[3:4].expand_as(factor)))
                audits.append(M.correspondence(old_joint,previous[pop]['factors']['joint'][i:i+1].cuda().expand_as(old_joint)))
                audits.append(M.correspondence(new_joint,previous[pop]['factors']['joint'][i+3:i+4].cuda().expand_as(new_joint)))
                scores={'00':old[0]*old[1],'10':new[0]*old[1],'01':old[0]*new[1],'11':new[0]*new[1],'cut':torch.zeros_like(old_joint)}
                compiled={};oracle={}
                for name,score in scores.items():
                    compiled[name]=T.execute(program,tok,mask,head,score);oracle[name]=T.native(model,tok,mask,head,score)
                    audits.append(M.correspondence(compiled[name],oracle[name]));finite &= bool(torch.isfinite(compiled[name]).all())
                audits.append(M.correspondence(compiled['00'][3:4,-1],old_native[pop]['native_query_logits'][i:i+1].cuda()))
                cut_audits.append(M.correspondence(compiled['cut']-compiled['00'],oracle['cut']-oracle['00']))
                delta=mixed(compiled);mixed_audits.append(M.correspondence(delta,mixed(oracle)))
                lp=compiled['00'].log_softmax(-1)
                for name,value in {**compiled,'mixed':delta}.items():values[name].append(value[:,-1].cpu().clone())
                for name in kls:kls[name].append((lp.exp()*(lp-compiled[name].log_softmax(-1))).sum(-1).clamp_min(0).cpu())
                fmap=torch.empty(24,dtype=torch.long).scatter_(0,tokens[i,:48:2],tokens[i,1:48:2]);answer=int(tokens[i,49])
                for hop in range(4):
                    if hop:answer=int(fmap[answer])
                    meta.append({**metadata[i],'hop':hop,'answer':answer})
            logits={k:torch.cat(v) for k,v in values.items()};kl={k:torch.cat(v) for k,v in kls.items()};groups={}
            for orientation in ('B2_later','B3_later'):
                groups[orientation]={}
                for hop in range(4):
                    sel=torch.tensor([m['orientation']==orientation and m['hop']==hop for m in meta]);removal=center(logits['cut'][sel]-logits['00'][sel])
                    norm=float(removal.square().mean().sqrt());den=max(norm,1e-6);arms={}
                    for name in ('10','01','11'):
                        change=center(logits[name][sel]-logits['00'][sel]);v=kl[name][sel]
                        arms[name]={'all_mean_kl':float(v.mean()),'all_p99_kl':float(torch.quantile(v.flatten(),.99)),
                            'query_mean_kl':float(v[:,-1].mean()),'query_p99_kl':float(torch.quantile(v[:,-1],.99)),
                            'query_change_over_removal_rms':float(change.square().mean().sqrt())/den}
                    primary=arms['11'];groups[orientation][str(hop)]={'requests':int(sel.sum()),'removal_effect_rms':norm,'denominator_floored':norm<1e-6,
                        'arms':arms,'mixed_over_removal_rms':float(center(logits['mixed'][sel]).square().mean().sqrt())/den,
                        'pair_portability_passed':primary['all_mean_kl']<=.001 and primary['all_p99_kl']<=.01 and primary['query_mean_kl']<=.001 and primary['query_p99_kl']<=.01 and primary['query_change_over_removal_rms']<=.01}
            results[pop]=groups;saved[pop]={'metadata':meta,'query_logits':logits,'token_kl':kl}
            print(json.dumps({'population':pop,'hop3':{o:g['3'] for o,g in groups.items()}}),flush=True)
    predictions={'pred_a_mechanical':finite and all(a['passed'] for a in audits+cut_audits),
        'pred_b_pair_portability':all(g['pair_portability_passed'] for p in results.values() for o in p.values() for g in o.values()),
        'pred_c_composition':all(a['passed'] for a in mixed_audits)}
    torch.save(saved,ROWS)
    result={'experiment':'matcher_pair_transport_v1','scope':'opened matched-label gate transfer, all native background retained',
        'controls':checks,'predictions':predictions,'populations':results,'output_oracle_max_abs':max(a['max_abs'] for a in audits),
        'cut_oracle_max_abs':max(a['max_abs'] for a in cut_audits),'mixed_oracle_max_abs':max(a['max_abs'] for a in mixed_audits),
        'independent_worlds':32,'requests':768,'independent_constants':program.independent_constant_count(),'native_coefficients_removed':0,
        'runner_sha256':digest(SOURCE),'bound_sha256':EXPECTED,'rows_sha256':digest(ROWS),'wall_seconds':time.perf_counter()-started,
        'terminal':'instrument_invalid' if not predictions['pred_a_mechanical'] or not predictions['pred_c_composition'] else
        ('matched_pair_transport_screen_passed' if predictions['pred_b_pair_portability'] else 'matched_pair_transport_rejected')}
    OUT.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ('terminal','predictions','output_oracle_max_abs','wall_seconds')}),flush=True)


if __name__=='__main__':main()
