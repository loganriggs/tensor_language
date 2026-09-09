#!/usr/bin/env python3
"""Automatic forward endpoint-family extraction on random record layouts.
pred_a_instrument: CPU controls, all-logit oracle1e-9/relative1e-10, native>=.8,
eligible groups nonempty, endpoint-only removal goldP loss>=.25 perpop/hop2/3.
pred_b_retarget: mapAll desiredaccuracy>=.8 and probability gain>=.5 perpop/hop.
pred_c_reuse: matching parity meets same bars; other parity |goldP change|<=.10.
pred_d_specificity: hop0/1 and backward h2/3 |goldP change|<=.10; rawquery joint
composition max<=1e-9. Null rejects generalized semantic family, no layout rescue.
32worlds3072queries,6arms,B8FP64,1800s,<256MiB/tensor; managed GPU only.
387968 native constants plus24576byte derived dictionary, zero parameters removed.
"""
# BQGATE: EXPERIMENT pred_a_instrument pred_b_retarget pred_c_reuse pred_d_specificity
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time

ROOT=Path('/workspace/tensor_language');POLY=ROOT/'basis_aligned/polynomial_causal';SOURCE=Path(__file__)
OUT=POLY/'FORWARD_ENDPOINT_RANDOM_LAYOUT_V1_RESULT.json';ROWS=POLY/'FORWARD_ENDPOINT_RANDOM_LAYOUT_V1_ROWS.pt'
PACKAGE=POLY/'EXACT_SOURCE_EDIT_V1_PROGRAM.pt';CHECKPOINT=ROOT/'runs_hop/attn4-rms-seed0/model.pt'
PACKAGE_SHA='e69bff6562c48bd1b59366f118157720734d353626b0a1576626bdd437a1421a'
CHECKPOINT_SHA='c9388cc8b1ba7e95c6a043cbf4d01592987447867a50781d3babe31ce3f8358b'
BOUND=[POLY/n for n in ('forward_endpoint_program_reference.py','forward_endpoint_random_layout_reference.py',
                       'exact_source_edit_reference.py','FORWARD_ENDPOINT_RANDOM_LAYOUT_V1_PREREGISTRATION.md')]+[ROOT/'model.py',ROOT/'deep_model.py']
EXPECTED='d12d0f8854c9fa555011bf28d4e934d95b331d26b1a322bba94f58c54ef88275'


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def bound_hash():return hashlib.sha256(b''.join(p.read_bytes() for p in BOUND)).hexdigest()
def relative(a,b):return float((a-b).square().mean().sqrt())/max(float(b.square().mean().sqrt()),1e-6)


def edits(own,mapped):
    a=mapped[0]-own[0];b=mapped[1]-own[1]
    return {'native':own[0]*0,'identity':-own[0]-own[1]+own[0]+own[1],
            'map_even':a,'map_odd':b,'map_all':a+b,'remove_all':-own[0]-own[1]}


def run(torch,F,R,E,checks):
    from hop_ablate import load
    os.chdir(ROOT);signal.alarm(1800);started=time.perf_counter()
    assert not OUT.exists() and not ROWS.exists() and digest(PACKAGE)==PACKAGE_SHA and digest(CHECKPOINT)==CHECKPOINT_SHA
    torch.backends.cuda.matmul.allow_tf32=False
    model,_=load('attn4-rms-seed0');model=model.to(device='cuda',dtype=torch.float64).eval()
    program=E.load_package(torch.load(PACKAGE,map_location='cpu',weights_only=True)).cuda().eval()
    maximum=max_relative=path_error=identity_error=0.;finite=True;rows={};results={}
    with torch.inference_mode():
        table=F.dictionary(program.background)
        for pop,worlds in R.populations().items():
            saved={};refs={};metadata=[];tokens=[];counts=[]
            for w in worlds:
                metadata.extend(w['metadata']);tokens.append(w['tokens']);counts.append(len(w['records']))
                sigma=w['sigma'].cuda()
                for i in range(0,96,8):
                    tok=w['tokens'][i:i+8].cuda();context=program.prepare(tok)
                    masks={j:m.cuda()[None].expand(len(tok),-1,-1) for j,m in w['masks'].items()}
                    own={j:F.messages(program.background,tok,m,table=table) for j,m in masks.items()}
                    mapped={j:F.messages(program.background,tok,m,sigma,table=table) for j,m in masks.items()}
                    original={j:F.native_messages(model,tok,m) for j,m in masks.items()}
                    changed={j:F.native_messages(model,tok,m,sigma) for j,m in masks.items()}
                    path_error=max(path_error,*(float((own[j]-original[j]).abs().max()) for j in masks),
                                   *(float((mapped[j]-changed[j]).abs().max()) for j in masks))
                    candidate=edits(own,mapped);oracle=edits(original,changed)
                    base=model.embed(tok)
                    for layer in model.layers[:-1]:base=layer(base)
                    native=model(tok);batch={}
                    for arm,delta in candidate.items():
                        actual=program.edit(context,delta)
                        expected=native if arm in ('native','identity') else model.head(model.layers[-1](base+oracle[arm]))
                        maximum=max(maximum,float((actual-expected).abs().max()));max_relative=max(max_relative,relative(actual,expected))
                        finite &= bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
                        saved.setdefault(arm,[]).append(actual[:,-1].cpu().clone());refs.setdefault(arm,[]).append(expected[:,-1].cpu().clone());batch[arm]=actual
                    identity_error=max(identity_error,float((batch['identity']-batch['native']).abs().max()))
            logits={a:torch.cat(v) for a,v in saved.items()};refs={a:torch.cat(v) for a,v in refs.items()}
            answer=torch.tensor([m['answer'] for m in metadata]);desired=torch.tensor([m['desired_answer'] for m in metadata])
            gold={a:v.softmax(-1).gather(-1,answer[:,None]).squeeze(-1) for a,v in logits.items()}
            target={a:v.softmax(-1).gather(-1,desired[:,None]).squeeze(-1) for a,v in logits.items()}
            def select(hop,eligible,parity=None):return torch.tensor([m['hop']==hop and m['eligible']==eligible and (parity is None or m['parity']==parity) for m in metadata])
            def score(sel,arm):
                return {'n':int(sel.sum()),'native_accuracy':float((logits['native'].argmax(-1)==answer)[sel].double().mean()),
                        'removal_gold_loss':float((gold['native']-gold['remove_all'])[sel].mean()),
                        'target_accuracy':float((logits[arm].argmax(-1)==desired)[sel].double().mean()),
                        'target_probability_gain':float((target[arm]-target['native'])[sel].mean())}
            groups={};parities={}
            for h in (2,3):
                groups[str(h)]=score(select(h,True),'map_all')
                for parity in (0,1):
                    arm='map_even' if parity==0 else 'map_odd';other='map_odd' if parity==0 else 'map_even';sel=select(h,True,parity)
                    parities[f'{h}_{parity}']={**score(sel,arm),'other_group_gold_change':float((gold[other]-gold['native'])[sel].mean())}
            controls={str(h):{'n':int(select(h,False).sum()),'map_all_gold_change':float((gold['map_all']-gold['native'])[select(h,False)].mean())} for h in range(4)}
            def residual(v):return (v['map_all']-v['native'])-(v['map_even']-v['native'])-(v['map_odd']-v['native'])
            composition=max(float(residual(logits).abs().max()),float(residual(refs).abs().max()))
            results[pop]={'eligible_groups':groups,'parity_groups':parities,'controls':controls,'raw_query_composition_max_abs':composition,'join_counts':counts}
            rows[pop]={'tokens':torch.cat(tokens),'metadata':metadata,'query_logits':logits,'native_oracle_query_logits':refs,'endpoint_maps':torch.stack([w['sigma'] for w in worlds])}
            print(json.dumps({'population':pop,**results[pop]}),flush=True)
    groups=[g for p in results.values() for g in p['eligible_groups'].values()];parities=[g for p in results.values() for g in p['parity_groups'].values()]
    pred_a=checks['passed'] and finite and max(maximum,path_error,identity_error)<=1e-9 and max_relative<=1e-10 and all(g['n']>0 and g['native_accuracy']>=.8 and g['removal_gold_loss']>=.25 for g in groups) and all(g['n']>0 for g in parities)
    pred_b=pred_a and all(g['target_accuracy']>=.8 and g['target_probability_gain']>=.5 for g in groups)
    pred_c=pred_a and all(g['target_accuracy']>=.8 and g['target_probability_gain']>=.5 and abs(g['other_group_gold_change'])<=.10 for g in parities)
    pred_d=pred_a and all(p['raw_query_composition_max_abs']<=1e-9 and all(c['n']>0 and abs(c['map_all_gold_change'])<=.10 for c in p['controls'].values()) for p in results.values())
    torch.save(rows,ROWS)
    receipt={'experiment':'forward_endpoint_random_layout_v1','runner_sha256':digest(SOURCE),'bound_sha256':EXPECTED,'package_sha256':PACKAGE_SHA,'checkpoint_sha256':CHECKPOINT_SHA,
             'controls':checks,'oracle_max_abs_logit':maximum,'oracle_max_relative_rms':max_relative,'path_oracle_max_abs':path_error,'identity_max_abs_logit':identity_error,
             'program_independent_constants':program.independent_constant_count(),'derived_dictionary_bytes':table.numel()*table.element_size(),'native_coefficients_removed':0,
             'populations':results,'rows_sha256':digest(ROWS),'independent_worlds':32,'query_variants':3072,
             'predictions':{'pred_a_instrument':bool(pred_a),'pred_b_retarget':bool(pred_b),'pred_c_reuse':bool(pred_c),'pred_d_specificity':bool(pred_d)},
             'terminal':'instrument_invalid' if not pred_a else ('random_layout_endpoint_family_supported' if pred_b and pred_c and pred_d else 'random_layout_endpoint_family_not_established'),
             'scope':'token-parsed endpoint family with native context; no structural coefficient reduction','wall_seconds':time.perf_counter()-started}
    OUT.write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({k:receipt[k] for k in ('terminal','predictions','wall_seconds')}),flush=True)


def main():
    sys.path[:0]=[str(ROOT),str(POLY)];assert bound_hash()==EXPECTED
    import torch
    import forward_endpoint_program_reference as F
    import forward_endpoint_random_layout_reference as R
    import exact_source_edit_reference as E
    torch.set_num_threads(2);fc=F.controls();rc=R.controls();checks={'passed':fc['passed'] and rc['passed'],'algebra':fc,'generator':rc};assert checks['passed']
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps({'dryrun':True,'controls':checks,'checkpoint_opened':False}));return
    run(torch,F,R,E,checks)


if __name__=='__main__':main()
