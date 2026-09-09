#!/usr/bin/env python3
"""Whole-model gate reuse under graph-preserving entity renaming.
pred_a_mechanical: native/factorial closure<=1e-9/relative1e-10, finite, live controls.
pred_b_distribution: native-renamed to payloadonly KL mean<=1e-3,p95<=1e-2,
querymean<=1e-3 perpopulation. pred_c_effect: centered query rename-effect
relativeRMS<=.01 perpopulation/hop. pred_d_semantics: nativecapability>=.8,
payloadtarget>=.8; gateonly originalaccuracy drop<=.05, |goldP change|<=.10.
Null closes global gate reuse, no selected layer/head/normalizer or fitting rescue.
16worlds1536pairs,4arms,B8FP64,1800s,<256MiB/tensor;managedGPU. All387968 constants
and paired token-derived gatecaches charged; no native parameters removed.
"""
# BQGATE: EXPERIMENT pred_a_mechanical pred_b_distribution pred_c_effect pred_d_semantics
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time

ROOT=Path('/workspace/tensor_language');POLY=ROOT/'basis_aligned/polynomial_causal';SOURCE=Path(__file__)
OUT=POLY/'GLOBAL_ENTITY_GATE_REUSE_V1_RESULT.json';ROWS=POLY/'GLOBAL_ENTITY_GATE_REUSE_V1_ROWS.pt'
PACKAGE=POLY/'EXACT_SOURCE_EDIT_V1_PROGRAM.pt';CHECKPOINT=ROOT/'runs_hop/attn4-rms-seed0/model.pt'
PACKAGE_SHA='e69bff6562c48bd1b59366f118157720734d353626b0a1576626bdd437a1421a'
CHECKPOINT_SHA='c9388cc8b1ba7e95c6a043cbf4d01592987447867a50781d3babe31ce3f8358b'
BOUND=[POLY/n for n in ('frozen_payload_lineage_reference.py','gated_payload_model_reference.py','forward_endpoint_random_layout_reference.py',
                       'field_intervention_metrics.py','exact_source_edit_reference.py','GLOBAL_ENTITY_GATE_REUSE_V1_PREREGISTRATION.md')]+[ROOT/'model.py',ROOT/'deep_model.py']
EXPECTED='b24a5d75d005167a99a734cbe3756ee9810acb009e7fa3f01372c262106d28ec'


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def bound_hash():return hashlib.sha256(b''.join(p.read_bytes() for p in BOUND)).hexdigest()
def worlds(R,G):
    out=R.populations(seeds=(27909,27910))
    for pop in out:
        out[pop]=out[pop][:8]
        for w in out[pop]:
            w['renamed']=G.rename(w['tokens'],w['sigma'])
            for m in w['metadata']:m['renamed_answer']=int(w['sigma'][m['answer']])
            row=w['renamed'][0];f=row.new_empty(24).scatter_(0,row[:48:2],row[1:48:2])
            for tok,m in zip(w['renamed'],w['metadata']):
                e=tok[49]
                for _ in range(m['hop']):e=f[e]
                assert int(e)==m['renamed_answer']
    return out


def run(torch,G,R,E,M,checks,data):
    from hop_ablate import load
    os.chdir(ROOT);signal.alarm(1800);started=time.perf_counter()
    assert not OUT.exists() and not ROWS.exists() and digest(PACKAGE)==PACKAGE_SHA and digest(CHECKPOINT)==CHECKPOINT_SHA
    torch.backends.cuda.matmul.allow_tf32=False
    model,_=load('attn4-rms-seed0');model=model.to(device='cuda',dtype=torch.float64).eval()
    program=E.load_package(torch.load(PACKAGE,map_location='cpu',weights_only=True)).cuda().eval()
    maximum=relative=factorial_error=0.;finite=True;results={};rows={};cache_sizes=set();gate_sizes=set()
    with torch.inference_mode():
        for pop,items in data.items():
            saved={};metadata=[];old_tokens=[];new_tokens=[];kls=[];term_norms={k:0. for k in ('payload','gate','interaction')}
            for w in items:
                metadata.extend(w['metadata']);old_tokens.append(w['tokens']);new_tokens.append(w['renamed'])
                for i in range(0,96,8):
                    tok=w['tokens'][i:i+8].cuda();renamed=w['renamed'][i:i+8].cuda()
                    arms,(old,new)=G.factorial(program,tok,renamed)
                    for name,tokens in (('native',tok),('renamed_native',renamed)):
                        audit=M.correspondence(arms[name],model(tokens));maximum=max(maximum,audit['max_abs']);relative=max(relative,audit['relative_rms']);finite &= audit['finite']
                    de=new['embedding']-old['embedding'];payload=G.execute(program,de,old)
                    gate=arms['gates_only']-arms['native'];interaction=G.execute(program,de,new)-payload
                    factorial_error=max(factorial_error,float((arms['native']+payload+gate+interaction-arms['renamed_native']).abs().max()))
                    for name,value in (('payload',payload),('gate',gate),('interaction',interaction)):
                        centered=value-value.mean(-1,keepdim=True);term_norms[name]+=float(centered.square().sum())
                    for name,value in arms.items():
                        finite &= bool(torch.isfinite(value).all());saved.setdefault(name,[]).append(value[:,-1].cpu().clone())
                    logp=arms['renamed_native'].log_softmax(-1);logq=arms['payload_only'].log_softmax(-1)
                    kls.append((logp.exp()*(logp-logq)).sum(-1).clamp_min(0).cpu())
                    gates=old['gates']+[old['final_gate']];gb=sum(v.numel()*v.element_size() for g in gates for v in g.values())//len(tok)
                    gate_sizes.add(gb);cache_sizes.add(gb+(old['embedding'].numel()+old['native_prefix'].numel())*old['embedding'].element_size()//len(tok))
            logits={k:torch.cat(v) for k,v in saved.items()};kl=torch.cat(kls);groups={}
            for hop in range(4):
                sel=[m['hop']==hop for m in metadata];s=torch.tensor(sel)
                payload=M.panel(logits,'payload_only',metadata,sel,'renamed_answer');gate=M.panel(logits,'gates_only',metadata,sel)
                renamed=M.panel(logits,'renamed_native',metadata,sel,'renamed_answer')
                target=logits['renamed_native'][s]-logits['native'][s];pred=logits['payload_only'][s]-logits['native'][s]
                target-=target.mean(-1,keepdim=True);pred-=pred.mean(-1,keepdim=True)
                effect_error=float((pred-target).square().mean().sqrt())/max(float(target.square().mean().sqrt()),1e-6)
                groups[str(hop)]={'payload':payload,'gates':gate,'renamed_native':renamed,'query_effect_relative_rms':effect_error}
            count=len(metadata)*51*29
            results[pop]={'distribution':{'all_mean_kl':float(kl.mean()),'token_p95_kl':float(torch.quantile(kl.flatten(),.95)),'query_mean_kl':float(kl[:,-1].mean())},
                          'groups':groups,'term_centered_rms':{k:(v/count)**.5 for k,v in term_norms.items()}}
            rows[pop]={'tokens':torch.cat(old_tokens),'renamed_tokens':torch.cat(new_tokens),'metadata':metadata,'query_logits':logits,'token_kl':kl}
            print(json.dumps({'population':pop,**results[pop]}),flush=True)
    groups=[g for p in results.values() for g in p['groups'].values()]
    capability=all(g['payload']['native_accuracy']>=.8 and g['renamed_native']['target_accuracy']>=.8 for g in groups)
    a=checks['passed'] and finite and max(maximum,factorial_error)<=1e-9 and relative<=1e-10 and all(g['payload']['n']>0 for g in groups)
    b=all(p['distribution']['all_mean_kl']<=1e-3 and p['distribution']['token_p95_kl']<=1e-2 and p['distribution']['query_mean_kl']<=1e-3 for p in results.values())
    c=all(g['query_effect_relative_rms']<=.01 for g in groups)
    d=capability and all(g['payload']['target_accuracy']>=.8 and g['gates']['target_accuracy']>=g['gates']['native_accuracy']-.05 and abs(g['gates']['gold_probability_change'])<=.10 for g in groups)
    torch.save(rows,ROWS)
    receipt={'experiment':'global_entity_gate_reuse_v1','runner_sha256':digest(SOURCE),'bound_sha256':EXPECTED,'package_sha256':PACKAGE_SHA,'checkpoint_sha256':CHECKPOINT_SHA,
             'controls':checks,'native_oracle_max_abs':maximum,'native_oracle_max_relative_rms':relative,'factorial_max_abs':factorial_error,'native_capability_licensed':capability,
             'populations':results,'gate_bytes_per_context':sorted(gate_sizes),'cache_bytes_per_context':sorted(cache_sizes),'paired_contexts':2,
             'independent_program_constants':program.independent_constant_count(),'native_coefficients_removed':0,'independent_worlds':16,'query_pairs':1536,'rows_sha256':digest(ROWS),
             'predictions':{'pred_a_mechanical':bool(a),'pred_b_distribution':bool(b),'pred_c_effect':bool(c),'pred_d_semantics':bool(d)},
             'terminal':'mechanically_invalid' if not a else ('global_gate_reuse_supported' if b and c and d else 'global_gate_reuse_not_established'),
             'scope':'paired-context whole-model gate interchange; no standalone gate simplification','wall_seconds':time.perf_counter()-started}
    OUT.write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({k:receipt[k] for k in ('terminal','predictions','wall_seconds')}),flush=True)


def main():
    sys.path[:0]=[str(ROOT),str(POLY)];assert bound_hash()==EXPECTED
    import torch
    import gated_payload_model_reference as G
    import forward_endpoint_random_layout_reference as R
    import exact_source_edit_reference as E
    import field_intervention_metrics as M
    torch.set_num_threads(2);gc=G.controls();mc=M.controls();data=worlds(R,G);checks={'passed':gc['passed'] and mc['passed'],'algebra':gc,'metrics':mc,'valid_conjugate_pairs':1536};assert checks['passed']
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps({'dryrun':True,'controls':checks,'checkpoint_opened':False}));return
    run(torch,G,R,E,M,checks,data)


if __name__=='__main__':main()
