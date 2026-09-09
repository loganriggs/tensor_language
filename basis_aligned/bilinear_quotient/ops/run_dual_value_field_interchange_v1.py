#!/usr/bin/env python3
"""Original-value/computed-endpoint interchange at overlapping postL2 states.
pred_a_mechanical: controls, finite, full oracle<=1e-9/relative1e-10, identity,
prefix producer identity, nonempty panels; no semantic criterion in validity.
pred_b_raw: native>=.8, raw removal loss>=.25, raw map accuracy>=.8/gain>=.5,
other consumer |goldP change|<=.10. pred_c_join: same bars for joined consumer.
pred_d_joint: distinct joint targets accuracy>=.8/gain>=.5; hop0 collateral<=.10.
Null rejects literal E/8 versus endpoint-field abstraction; no scale/producer tune.
32worlds3072queries,8arms,B8FP64,1800s,<256MiB/tensor; managed GPU only.
387968 native constants plus24576byte derived dictionary; no structural saving.
"""
# BQGATE: EXPERIMENT pred_a_mechanical pred_b_raw pred_c_join pred_d_joint
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time

ROOT=Path('/workspace/tensor_language');POLY=ROOT/'basis_aligned/polynomial_causal';SOURCE=Path(__file__)
OUT=POLY/'DUAL_VALUE_FIELD_INTERCHANGE_V1_RESULT.json';ROWS=POLY/'DUAL_VALUE_FIELD_INTERCHANGE_V1_ROWS.pt'
PACKAGE=POLY/'EXACT_SOURCE_EDIT_V1_PROGRAM.pt';CHECKPOINT=ROOT/'runs_hop/attn4-rms-seed0/model.pt'
PACKAGE_SHA='e69bff6562c48bd1b59366f118157720734d353626b0a1576626bdd437a1421a'
CHECKPOINT_SHA='c9388cc8b1ba7e95c6a043cbf4d01592987447867a50781d3babe31ce3f8358b'
BOUND=[POLY/n for n in ('forward_endpoint_program_reference.py','forward_endpoint_random_layout_reference.py','dual_value_field_reference.py',
                       'field_intervention_metrics.py','exact_source_edit_reference.py','DUAL_VALUE_FIELD_INTERCHANGE_V1_PREREGISTRATION.md')]+[ROOT/'model.py',ROOT/'deep_model.py']
EXPECTED='476ee992b5a37c98b1084ec341843828fa496e60503dd20f47c847a8fb3ee6cf'


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def bound_hash():return hashlib.sha256(b''.join(p.read_bytes() for p in BOUND)).hexdigest()
def edits(raw,joined,new_raw,new_joined):
    a=new_raw-raw;b=new_joined-joined
    return {'native':raw*0,'identity':-raw-joined+raw+joined,'map_raw':a,'map_join':b,'map_both':a+b,
            'remove_raw':-raw,'remove_join':-joined,'remove_both':-raw-joined}


def run(torch,F,D,E,M,checks):
    from hop_ablate import load
    os.chdir(ROOT);signal.alarm(1800);started=time.perf_counter()
    assert not OUT.exists() and not ROWS.exists() and digest(PACKAGE)==PACKAGE_SHA and digest(CHECKPOINT)==CHECKPOINT_SHA
    torch.backends.cuda.matmul.allow_tf32=False
    model,_=load('attn4-rms-seed0');model=model.to(device='cuda',dtype=torch.float64).eval()
    program=E.load_package(torch.load(PACKAGE,map_location='cpu',weights_only=True)).cuda().eval()
    maximum=max_relative=prefix_error=identity_error=0.;finite=True;rows={};results={}
    with torch.inference_mode():
        table=F.dictionary(program.background)
        for pop,worlds in D.populations().items():
            saved={};refs={};metadata=[];tokens=[]
            for w in worlds:
                metadata.extend(w['metadata']);tokens.append(w['tokens']);rm=w['raw_map'].cuda();jm=w['join_map'].cuda()
                for i in range(0,96,8):
                    tok=w['tokens'][i:i+8].cuda();context=program.prepare(tok);mask=w['mask'].cuda()[None].expand(len(tok),-1,-1)
                    raw=D.raw_field(program.background,tok);joined=F.messages(program.background,tok,mask,table=table)
                    candidate=edits(raw,joined,D.raw_field(program.background,tok,rm),F.messages(program.background,tok,mask,jm,table=table))
                    oracle=edits(D.raw_field(model,tok),F.native_messages(model,tok,mask),D.raw_field(model,tok,rm),F.native_messages(model,tok,mask,jm))
                    base,identity=D.prefix_identity(model,tok);prefix_error=max(prefix_error,float((base-identity).abs().max()))
                    native=model(tok);batch={}
                    for arm,delta in candidate.items():
                        actual=program.edit(context,delta);expected=native if arm in ('native','identity') else model.head(model.layers[-1](base+oracle[arm]))
                        audit=M.correspondence(actual,expected);maximum=max(maximum,audit['max_abs']);max_relative=max(max_relative,audit['relative_rms']);finite &= audit['finite']
                        saved.setdefault(arm,[]).append(actual[:,-1].cpu().clone());refs.setdefault(arm,[]).append(expected[:,-1].cpu().clone());batch[arm]=actual
                    identity_error=max(identity_error,float((batch['identity']-batch['native']).abs().max()))
            logits={a:torch.cat(v) for a,v in saved.items()};refs={a:torch.cat(v) for a,v in refs.items()};groups={}
            for field,hops in (('raw',(1,2,3)),('join',(3,))):
                for hop in hops:
                    sel=[m[field+'_consumer'] and m['hop']==hop for m in metadata];other='join' if field=='raw' else 'raw'
                    groups[f'{field}_h{hop}']={'own_map':M.panel(logits,'map_'+field,metadata,sel,field+'_desired'),
                                               'own_remove':M.panel(logits,'remove_'+field,metadata,sel),
                                               'other_map':M.panel(logits,'map_'+other,metadata,sel),
                                               'both_map':M.panel(logits,'map_both',metadata,sel,'both_desired')}
            controls={a:M.panel(logits,a,metadata,[m['hop']==0 for m in metadata]) for a in ('map_raw','map_join','map_both')}
            interaction=(logits['map_both']-logits['native'])-(logits['map_raw']-logits['native'])-(logits['map_join']-logits['native'])
            centered=interaction-interaction.mean(-1,keepdim=True)
            results[pop]={'groups':groups,'hop0_controls':controls,'joint_interaction_max_abs':float(interaction.abs().max()),'joint_interaction_centered_rms':float(centered.square().mean().sqrt())}
            rows[pop]={'tokens':torch.cat(tokens),'metadata':metadata,'query_logits':logits,'native_oracle_query_logits':refs,
                       'raw_maps':torch.stack([w['raw_map'] for w in worlds]),'join_maps':torch.stack([w['join_map'] for w in worlds])}
            print(json.dumps({'population':pop,**results[pop]}),flush=True)
    raw=[g for p in results.values() for k,g in p['groups'].items() if k.startswith('raw')];joined=[p['groups']['join_h3'] for p in results.values()];all_groups=raw+joined
    def target(g):return g['target_accuracy']>=.8 and g['target_probability_gain']>=.5
    pred_a=checks['passed'] and finite and max(maximum,prefix_error,identity_error)<=1e-9 and max_relative<=1e-10 and all(g['own_map']['n']>0 for g in all_groups)
    pred_b=all(g['own_map']['native_accuracy']>=.8 and target(g['own_map']) and g['own_remove']['removal_gold_loss']>=.25 for g in raw) and all(abs(g['other_map']['gold_probability_change'])<=.10 for g in joined)
    pred_c=all(g['own_map']['native_accuracy']>=.8 and target(g['own_map']) and g['own_remove']['removal_gold_loss']>=.25 for g in joined) and all(abs(g['other_map']['gold_probability_change'])<=.10 for g in raw)
    pred_d=all(target(g['both_map']) for g in all_groups) and all(abs(c['gold_probability_change'])<=.10 for p in results.values() for c in p['hop0_controls'].values())
    torch.save(rows,ROWS)
    receipt={'experiment':'dual_value_field_interchange_v1','runner_sha256':digest(SOURCE),'bound_sha256':EXPECTED,'package_sha256':PACKAGE_SHA,'checkpoint_sha256':CHECKPOINT_SHA,
             'controls':checks,'oracle_max_abs_logit':maximum,'oracle_max_relative_rms':max_relative,'prefix_identity_max_abs':prefix_error,'identity_max_abs_logit':identity_error,
             'program_independent_constants':program.independent_constant_count(),'derived_dictionary_bytes':table.numel()*table.element_size(),'native_coefficients_removed':0,
             'populations':results,'rows_sha256':digest(ROWS),'independent_worlds':32,'query_variants':3072,
             'predictions':{'pred_a_mechanical':bool(pred_a),'pred_b_raw':bool(pred_b),'pred_c_join':bool(pred_c),'pred_d_joint':bool(pred_d)},
             'terminal':'mechanically_invalid' if not pred_a else ('dual_value_fields_supported' if pred_b and pred_c and pred_d else 'dual_value_fields_not_established'),
             'scope':'literal raw E/8 and composed endpoint with native context; no structural coefficient reduction','wall_seconds':time.perf_counter()-started}
    OUT.write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({k:receipt[k] for k in ('terminal','predictions','wall_seconds')}),flush=True)


def main():
    sys.path[:0]=[str(ROOT),str(POLY)];assert bound_hash()==EXPECTED
    import torch
    import forward_endpoint_program_reference as F
    import dual_value_field_reference as D
    import exact_source_edit_reference as E
    import field_intervention_metrics as M
    torch.set_num_threads(2);dc=D.controls();mc=M.controls();checks={'passed':dc['passed'] and mc['passed'],'fields':dc,'metrics':mc};assert checks['passed']
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps({'dryrun':True,'controls':checks,'checkpoint_opened':False}));return
    run(torch,F,D,E,M,checks)


if __name__=='__main__':main()
