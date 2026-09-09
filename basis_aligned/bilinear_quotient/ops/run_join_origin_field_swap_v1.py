#!/usr/bin/env python3
"""Test whether copied origin-field interchange retargets two backward joins.

pred_a_instrument: native oracle closure1e-9/relative1e-10, capability>=.8,cut>=.25.
pred_b_retarget: bothSwap target accuracy>=.8 andtarget-prob gain>=.5 perquery/pop.
pred_c_reuse: isolated single transfers meet the same retarget bars.
pred_d_specificity: lowerhop |goldP change|<=.1 andrawquery composition max<=1e-9.
Null rejects semantic retargeting, not generic causal damage.32worlds256variants,
9arms,B4FP64,1800s,<256MiB/tensor;managedGPU. Exported387968 native constants,
no fitted adapter or new semantic parameter reduction.
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
OUT=POLY/'JOIN_ORIGIN_FIELD_SWAP_V1_RESULT.json';ROWS=POLY/'JOIN_ORIGIN_FIELD_SWAP_V1_ROWS.pt'
PACKAGE=POLY/'EXACT_SOURCE_EDIT_V1_PROGRAM.pt'
EXPECTED_PACKAGE='e69bff6562c48bd1b59366f118157720734d353626b0a1576626bdd437a1421a'
CHECKPOINT=ROOT/'runs_hop/attn4-rms-seed0/model.pt'
EXPECTED_CHECKPOINT='c9388cc8b1ba7e95c6a043cbf4d01592987447867a50781d3babe31ce3f8358b'
BOUND=[POLY/n for n in ('join_origin_field_swap_reference.py','join_contribution_context_reference.py',
                       'exact_source_edit_reference.py','JOIN_ORIGIN_FIELD_SWAP_V1_PREREGISTRATION.md')]+[ROOT/'model.py',ROOT/'deep_model.py']
EXPECTED='2a9d49b8846592f4c8b066b3b7cc364d8cf8164a75eace01b6a121148addea83'


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def bound_hash():return hashlib.sha256(b''.join(p.read_bytes() for p in BOUND)).hexdigest()
def relative(a,b):return float((a-b).square().mean().sqrt())/max(float(b.square().mean().sqrt()),1e-6)


def edits(own,swapped,full):
    a=swapped[0]-own[0];b=swapped[1]-own[1];zero=own[0]*0
    return {'native':zero,'identity':-own[0]-own[1]+own[0]+own[1],
            'swap_a':a,'swap_b':b,'both_swap':a+b,'swap_a_cut_b':a-full[1],
            'swap_b_cut_a':b-full[0],'cut_a':-full[0],'cut_b':-full[1]}


def run(torch,R,E,J,checks):
    from hop_ablate import load
    os.chdir(ROOT);signal.alarm(1800);started=time.perf_counter()
    assert not OUT.exists() and not ROWS.exists() and digest(CHECKPOINT)==EXPECTED_CHECKPOINT and digest(PACKAGE)==EXPECTED_PACKAGE
    torch.backends.cuda.matmul.allow_tf32=False
    model,_=load('attn4-rms-seed0');model=model.to(device='cuda',dtype=torch.float64).eval()
    program=E.load_package(torch.load(PACKAGE,map_location='cpu',weights_only=True)).cuda().eval()
    results={};rows={};maximum=max_relative=path_error=identity_error=0.;finite=True
    with torch.inference_mode():
        for pop,worlds in R.populations().items():
            outputs={};reference_query={};metadata=[];tokens_saved=[]
            for w in worlds:
                tokall=w['recipient'].cuda();masks={j:m.cuda() for j,m in w['masks'].items()};heads=w['heads']
                tokens_saved.append(w['recipient'])
                desired=w['answers'].clone();desired[3]=w['answers'][7];desired[7]=w['answers'][3]
                metadata.extend({'world':w['world'],'query':int(q),'hop':int(h),'answer':int(a),'desired_answer':int(d)}
                                for q,h,a,d in zip(w['query'],w['hops'],w['answers'],desired))
                for i in range(0,8,4):
                    tok=tokall[i:i+4];context=program.prepare(tok)
                    own=R.paths(program.background,tok,masks);swapped=R.paths(program.background,tok,masks,True)
                    native_own=R.native_hook_paths(model,tok,masks);native_swapped=R.native_hook_paths(model,tok,masks,True)
                    path_error=max(path_error,*(float((v-native_own[j]).abs().max()) for j,v in own.items()),
                                   *(float((v-native_swapped[j]).abs().max()) for j,v in swapped.items()))
                    candidate=edits(own,swapped,J.contributions(program.background,tok,masks,heads))
                    oracle=edits(native_own,native_swapped,J.contributions(model,tok,masks,heads))
                    base=model.embed(tok)
                    for layer in model.layers[:-1]:base=layer(base)
                    batch={}
                    for arm,delta in candidate.items():
                        actual=program.edit(context,delta)
                        if arm in ('cut_a','cut_b'):
                            with J.intervene(model,masks,heads,(0 if arm=='cut_a' else 1,)):expected=model(tok)
                        else:expected=model.head(model.layers[-1](base+oracle[arm]))
                        maximum=max(maximum,float((actual-expected).abs().max()));max_relative=max(max_relative,relative(actual,expected))
                        finite &= bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
                        outputs.setdefault(arm,[]).append(actual.cpu());reference_query.setdefault(arm,[]).append(expected[:,-1].cpu().clone());batch[arm]=actual
                    identity_error=max(identity_error,float((batch['identity']-batch['native']).abs().max()))
            logits={a:torch.cat(v) for a,v in outputs.items()};reference_query={a:torch.cat(v) for a,v in reference_query.items()}
            answer=torch.tensor([m['answer'] for m in metadata]);desired=torch.tensor([m['desired_answer'] for m in metadata])
            gold={a:v[:,-1].softmax(-1).gather(-1,answer[:,None]).squeeze(-1) for a,v in logits.items()}
            target={a:v[:,-1].softmax(-1).gather(-1,desired[:,None]).squeeze(-1) for a,v in logits.items()}
            groups={}
            for query in range(2):
                select=torch.tensor([m['query']==query and m['hop']==3 for m in metadata]);cut='cut_a' if query==0 else 'cut_b'
                single='swap_b_cut_a' if query==0 else 'swap_a_cut_b'
                groups[str(query)]={'native_accuracy':float((logits['native'][:,-1].argmax(-1)==answer)[select].double().mean()),
                                    'own_cut_gold_loss':float((gold['native']-gold[cut])[select].mean()),
                                    'joint_target_accuracy':float((logits['both_swap'][:,-1].argmax(-1)==desired)[select].double().mean()),
                                    'joint_target_probability_gain':float((target['both_swap']-target['native'])[select].mean()),
                                    'single_arm':single,'single_target_accuracy':float((logits[single][:,-1].argmax(-1)==desired)[select].double().mean()),
                                    'single_target_probability_gain':float((target[single]-target['native'])[select].mean())}
            lower={}
            for q in range(2):
                for h in range(3):
                    select=torch.tensor([m['query']==q and m['hop']==h for m in metadata])
                    lower[str(q)+'_'+str(h)]=float((gold['both_swap']-gold['native'])[select].mean())
            def joint_residual(v):return (v['both_swap']-v['native'])-(v['swap_a']-v['native'])-(v['swap_b']-v['native'])
            composition=max(float(joint_residual({a:v[:,-1] for a,v in logits.items()}).abs().max()),float(joint_residual(reference_query).abs().max()))
            results[pop]={'queries':groups,'lowerhop_gold_changes':lower,'raw_query_composition_max_abs':composition}
            rows[pop]={'tokens':torch.cat(tokens_saved),'metadata':metadata,'query_logits':{a:v[:,-1].clone() for a,v in logits.items()},'native_oracle_query_logits':reference_query}
            print(json.dumps({'population':pop,**results[pop]}),flush=True)
    groups=[q for p in results.values() for q in p['queries'].values()]
    pred_a=checks['passed'] and finite and max(maximum,path_error,identity_error)<=1e-9 and max_relative<=1e-10 and all(q['native_accuracy']>=.8 and q['own_cut_gold_loss']>=.25 for q in groups)
    pred_b=pred_a and all(q['joint_target_accuracy']>=.8 and q['joint_target_probability_gain']>=.5 for q in groups)
    pred_c=pred_a and all(q['single_target_accuracy']>=.8 and q['single_target_probability_gain']>=.5 for q in groups)
    pred_d=pred_a and all(p['raw_query_composition_max_abs']<=1e-9 and all(abs(v)<=.10 for v in p['lowerhop_gold_changes'].values()) for p in results.values())
    torch.save(rows,ROWS)
    receipt={'experiment':'join_origin_field_swap_v1','runner_sha256':digest(SOURCE),'bound_sha256':EXPECTED,'package_sha256':EXPECTED_PACKAGE,
             'checkpoint_sha256':EXPECTED_CHECKPOINT,'controls':checks,'oracle_max_abs_logit':maximum,'oracle_max_relative_rms':max_relative,
             'path_oracle_max_abs':path_error,'identity_max_abs_logit':identity_error,'program_independent_constants':program.independent_constant_count(),
             'populations':results,'rows':str(ROWS.relative_to(ROOT)),'rows_sha256':digest(ROWS),'independent_worlds':32,'query_variants':256,
             'predictions':{'pred_a_instrument':bool(pred_a),'pred_b_retarget':bool(pred_b),'pred_c_reuse':bool(pred_c),'pred_d_specificity':bool(pred_d)},
             'terminal':'instrument_invalid' if not pred_a else ('origin_field_interchange_supported' if pred_b and pred_c and pred_d else 'origin_field_interchange_not_established'),
             'scope':'explicit native-background field manipulation; no semantic parameter reduction','wall_seconds':time.perf_counter()-started}
    OUT.write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({k:receipt[k] for k in ('terminal','predictions','wall_seconds')}),flush=True)


def main():
    sys.path[:0]=[str(ROOT),str(POLY)];assert bound_hash()==EXPECTED
    import torch
    import join_origin_field_swap_reference as R
    import exact_source_edit_reference as E
    import join_contribution_context_reference as J
    torch.set_num_threads(2);checks=R.controls();assert checks['passed'],checks
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps({'dryrun':True,'controls':checks,'checkpoint_opened':False}));return
    run(torch,R,E,J,checks)


if __name__=='__main__':main()
