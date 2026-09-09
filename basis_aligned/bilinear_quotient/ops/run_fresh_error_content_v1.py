#!/usr/bin/env python3
"""Fresh complete-cohort conditional error-content confirmation.
pred_a_mechanical: all3072 native states retained, full-output oracle<=1e-9/1e-10,
fixed-gain additive composition<=1e-9, finite/identity/restored controls.
pred_b_coverage: >=3 errors/population and supported cases/field, >=.8 coverage/pop.
pred_c_mediation: named path margin>=.5 whole, wrongP loss>=.25, >=.5 whole repairs;
>=.5 of ALL errors repaired perpopulation. pred_d_specificity: >=5 correct peers,
mean absolute goldP change<=.10, >=.95 staycorrect. Empty evidence is untested.
Null: fixed raw/join error explanation fails; no source/head/dose/field expansion.
16worlds1536pairs3072states, B8FP64,1800s,256MiB/tensor; managedGPU only.
All387968 opaque constants/native background charged, no structural saving.
"""
# BQGATE: EXPERIMENT pred_a_mechanical pred_b_coverage pred_c_mediation pred_d_specificity
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time

ROOT=Path('/workspace/tensor_language');POLY=ROOT/'basis_aligned/polynomial_causal';SOURCE=Path(__file__)
OUT=POLY/'FRESH_ERROR_CONTENT_V1_RESULT.json';ROWS=POLY/'FRESH_ERROR_CONTENT_V1_ROWS.pt'
PACKAGE=POLY/'EXACT_SOURCE_EDIT_V1_PROGRAM.pt';CHECKPOINT=ROOT/'runs_hop/attn4-rms-seed0/model.pt'
BOUND=[POLY/n for n in ('FRESH_ERROR_CONTENT_V1_PREREGISTRATION.md','bad_source_content_reference.py',
       'final_source_failure_reference.py','forward_endpoint_program_reference.py','forward_endpoint_random_layout_reference.py',
       'gated_payload_model_reference.py','source_port_interchange_reference.py','exact_source_edit_reference.py','field_intervention_metrics.py')]+[ROOT/'model.py',ROOT/'deep_model.py']
EXPECTED='e6d6ca36c6f01d74f52e5c75f8706efbbae27c728c30e5743ada8ea285409d27'


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def bound_hash():return hashlib.sha256(b''.join(p.read_bytes() for p in BOUND)).hexdigest()


def field_type(tokens,j,wrong):
    values=tokens[:48].reshape(24,2).tolist();f=dict(values);positions={k:i for i,(k,v) in enumerate(values)};key=values[j][0]
    if f[key]==wrong:return 'raw'
    if f[f[key]]==wrong and positions[f[key]]<j:return 'join'
    return 'unsupported'


def main():
    sys.path[:0]=[str(ROOT),str(POLY)];assert bound_hash()==EXPECTED
    import torch
    import bad_source_content_reference as B
    import exact_source_edit_reference as E
    import final_source_failure_reference as F
    import field_intervention_metrics as M
    import forward_endpoint_program_reference as J
    import forward_endpoint_random_layout_reference as R
    import gated_payload_model_reference as G
    torch.set_num_threads(2);checks=B.controls();assert checks['passed']
    data={pop:worlds[:8] for pop,worlds in R.populations(seeds=(30909,30910)).items()}
    # Model-free nomination control includes a backward join that must be rejected.
    t=torch.stack((torch.arange(24),torch.arange(24).roll(-1)),-1).flatten()
    assert field_type(t,3,4)=='raw' and field_type(t,3,5)=='unsupported'
    reordered=t.reshape(24,2).clone();reordered[[3,4]]=reordered[[4,3]];reordered=reordered.flatten()
    assert field_type(reordered,4,5)=='join' and field_type(reordered,4,7)=='unsupported'
    for worlds in data.values():
        for w in worlds:
            w['renamed']=G.rename(w['tokens'],w['sigma'])
            f=w['renamed'][0].new_empty(24).scatter_(0,w['renamed'][0,:48:2],w['renamed'][0,1:48:2])
            for tok,m in zip(w['renamed'],w['metadata']):
                e=int(tok[49])
                for _ in range(m['hop']):e=int(f[e])
                assert e==int(w['sigma'][m['answer']])
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps({'dryrun':True,'controls':checks,'states':3072,'nomination_controls':4,'checkpoint_opened':False}));return
    from hop_ablate import load
    os.chdir(ROOT);signal.alarm(1800);started=time.perf_counter()
    assert not OUT.exists() and not ROWS.exists()
    assert digest(PACKAGE)=='e69bff6562c48bd1b59366f118157720734d353626b0a1576626bdd437a1421a'
    assert digest(CHECKPOINT)=='c9388cc8b1ba7e95c6a043cbf4d01592987447867a50781d3babe31ce3f8358b'
    torch.backends.cuda.matmul.allow_tf32=False
    program=E.load_package(torch.load(PACKAGE,map_location='cpu',weights_only=True)).cuda().eval()
    model,_=load('attn4-rms-seed0');model=model.to(device='cuda',dtype=torch.float64).eval()
    audits=[];all_native={};error_rows=[];saved={k:[] for k in ('native','whole','nominated','opposite','joint','identity','raw','join')};composition=0.;cache_bytes=set()
    with torch.inference_mode():
        for pop,worlds in data.items():
            cohort=[]
            for w in worlds:
                natives={}
                for side,key in (('old','tokens'),('renamed','renamed')):
                    parts=[]
                    for start in range(0,96,8):
                        tok=w[key][start:start+8].cuda();context=program.prepare(tok);native=model(tok)
                        audits.append(M.correspondence(context['logits'],native));parts.append(native[:,-1].cpu().clone())
                    natives[side]=torch.cat(parts)
                for i,m in enumerate(w['metadata']):
                    answers={'old':m['answer'],'renamed':int(w['sigma'][m['answer']])}
                    for side,key in (('old','tokens'),('renamed','renamed')):
                        winner=int(natives[side][i].argmax());answer=answers[side]
                        cohort.append({'world':w['world'],'query':m['query'],'hop':m['hop'],'side':side,'answer':answer,'prediction':winner,
                                       'tokens':w[key][i].clone(),'logits':natives[side][i].clone()})
                        if winner==answer:continue
                        tok=w[key][i:i+1].cuda();context=program.prepare(tok);source,_=F.messages(program,context)
                        pairs=source[:,:48].reshape(1,24,2,29).sum(2)[0];j=int((pairs[:,winner]-pairs[:,answer]).argmax())
                        field=field_type(w[key][i],j,winner)
                        inverse=torch.argsort(w['sigma']);canonical=winner if side=='old' else int(inverse[winner]) if winner<24 else winner
                        for target_side,target_key in (('old','tokens'),('renamed','renamed')):
                            target=w[target_key][i:i+1].cuda();context=program.prepare(target)
                            tensors=[v for v in context.values() if torch.is_tensor(v)]+list(context['features'].values())
                            storages={v.untyped_storage().data_ptr():v.untyped_storage().nbytes() for v in tensors}
                            cache_bytes.add(sum(storages.values()))
                            base=model.embed(target)
                            for layer in model.layers[:3]:base=layer(base)
                            raw=torch.zeros_like(context['x']);raw[:,2*j+1]=program.background.embed(target)[:,2*j+1]/8
                            mask,_=J.joins(target);mask[:,:2*j]=False;mask[:,2*j+2:]=False
                            joined=J.messages(program.background,target,mask);native_join=J.native_messages(model,target,mask)
                            native_raw=torch.zeros_like(base);native_raw[:,2*j+1]=model.embed(target)[:,2*j+1]/8
                            content={'raw':B.execute(program,context,raw),'join':B.execute(program,context,joined),'unsupported':context['logits']}
                            joint=B.execute(program,context,raw+joined);identity=B.execute(program,context,raw*0)
                            for actual,expected in ((joined,native_join),(content['raw'],B.native(model,base,native_raw)),
                                (content['join'],B.native(model,base,native_join)),(joint,B.native(model,base,native_raw+native_join)),(identity,context['logits'])):
                                audits.append(M.correspondence(actual,expected))
                            composition=max(composition,float((joint-content['raw']-content['join']+context['logits']).abs().max()))
                            source,residual=F.messages(program,context);whole=context['logits'][:,-1]-source[:,2*j:2*j+2].sum(1)
                            audits.append(M.correspondence(whole,F.native_binding_cuts(model,base)[j:j+1]))
                            audits.append(M.correspondence(source.sum(1)+residual,context['logits'][:,-1]))
                            opposite='join' if field=='raw' else 'raw'
                            outputs={'native':context['logits'][:,-1],'whole':whole,'nominated':content[field][:,-1],
                                     'opposite':content[opposite][:,-1],'joint':joint[:,-1],'identity':identity[:,-1],
                                     'raw':content['raw'][:,-1],'join':content['join'][:,-1]}
                            for arm,value in outputs.items():saved[arm].append(value[0].cpu().clone())
                            gold=answers[target_side];foil=canonical if target_side=='old' or canonical>=24 else int(w['sigma'][canonical]);p=outputs['native'].softmax(-1)[0]
                            metrics={}
                            for arm,value in outputs.items():
                                q=value.softmax(-1)[0]
                                metrics[arm]={'prediction':int(value.argmax()),'gold_probability_change':float(q[gold]-p[gold]),
                                    'wrong_probability_loss':float(p[foil]-q[foil]),'margin_reduction':float(outputs['native'][0,foil]-outputs['native'][0,gold]-value[0,foil]+value[0,gold])}
                            whole_effect=outputs['whole']-outputs['native'];named_effect=outputs['nominated']-outputs['native']
                            whole_effect-=whole_effect.mean(-1,keepdim=True);named_effect-=named_effect.mean(-1,keepdim=True)
                            effect_error=float((named_effect-whole_effect).square().mean().sqrt())/max(float(whole_effect.square().mean().sqrt()),1e-6)
                            error_rows.append({'population':pop,'world':w['world'],'query':m['query'],'hop':m['hop'],'primary_side':side,'side':target_side,
                                'is_primary':side==target_side,'binding':j,'field':field,'answer':gold,'foil':foil,'native_correct':int(outputs['native'].argmax())==gold,
                                'join_path_present':bool(mask.any()),'source_key':int(target[0,2*j]),'source_value':int(target[0,2*j+1]),
                                'nominated_vs_whole_query_effect_relative_rms':effect_error,'arms':metrics})
            all_native[pop]=cohort
            print(json.dumps({'population':pop,'states':len(cohort),'errors':sum(c['prediction']!=c['answer'] for c in cohort)}),flush=True)
    logits={k:torch.stack(v) if v else torch.empty(0,29) for k,v in saved.items()}
    primary=[r for r in error_rows if r['is_primary']];coverage={};groups={};b_tests=[];c_tests=[]
    for pop in data:
        group=[r for r in primary if r['population']==pop];n=len(group);supported=sum(r['field']!='unsupported' for r in group)
        repaired=sum(r['arms']['nominated']['prediction']==r['answer'] for r in group)
        coverage[pop]={'native_states':len(all_native[pop]),'errors':n,'supported':supported,'unsupported':n-supported,'nominated_repaired':repaired}
        b_tests.append(n>=3 and supported>=.8*n);c_tests.append(n>0 and repaired>=.5*n)
    for field in ('raw','join'):
        sel=[r['is_primary'] and r['field']==field for r in error_rows];group=[r for r,s in zip(error_rows,sel) if s];n=len(group)
        mean=lambda arm,key:sum(r['arms'][arm][key] for r in group)/n if n else None
        whole=mean('whole','margin_reduction');nominated=mean('nominated','margin_reduction');loss=mean('nominated','wrong_probability_loss')
        repairable=[r for r in group if r['arms']['whole']['prediction']==r['answer']];repaired=sum(r['arms']['nominated']['prediction']==r['answer'] for r in repairable)
        b_tests.append(n>=3);c_tests.append(n>0 and whole>0 and nominated>=.5*whole and loss>=.25 and bool(repairable) and repaired>=.5*len(repairable))
        groups[field]={'n':n,'whole_margin_reduction':whole,'nominated_margin_reduction':nominated,'wrong_probability_loss':loss,
            'whole_repairable':len(repairable),'also_repaired_by_content':repaired,'tested':n>0 and bool(repairable),
            'arms':{arm:M.panel(logits,arm,error_rows,sel) for arm in saved} if n else {}}
    peer=[not r['is_primary'] and r['native_correct'] for r in error_rows];controls_rows=[r for r,s in zip(error_rows,peer) if s];n=len(controls_rows)
    specificity=M.panel(logits,'nominated',error_rows,peer) if n else {'n':0}
    specificity.update({'mean_absolute_goldP_change':sum(abs(r['arms']['nominated']['gold_probability_change']) for r in controls_rows)/n if n else None,
        'worst_goldP_loss':max([0.]+[-r['arms']['nominated']['gold_probability_change'] for r in controls_rows])})
    cells={}
    for pop in data:
        for field in ('raw','join','unsupported'):
            for role in ('primary','correct_peer'):
                sel=[r['population']==pop and r['field']==field and (r['is_primary'] if role=='primary' else not r['is_primary'] and r['native_correct']) for r in error_rows]
                cells[pop+'/'+field+'/'+role]={arm:M.panel(logits,arm,error_rows,sel) for arm in saved} if any(sel) else {'n':0,'tested':False}
    assert len(primary)==sum(c['prediction']!=c['answer'] for cohort in all_native.values() for c in cohort)
    torch.save({'all_native':all_native,'error_metadata':error_rows,'error_query_logits':logits},ROWS)
    a=checks['passed'] and all(c['passed'] for c in audits) and composition<=1e-9 and sum(map(len,all_native.values()))==3072
    d=n>=5 and specificity['mean_absolute_goldP_change']<=.10 and specificity['target_accuracy']>=.95
    predictions={'pred_a_mechanical':a,'pred_b_coverage':all(b_tests),'pred_c_mediation':all(c_tests),'pred_d_specificity':d}
    result={'experiment':'fresh_error_content_v1','scope':'fresh conditional error explanation; native background retained, no deployable detector or structural saving',
        'predictions':predictions,'terminal':'mechanically_invalid' if not a else 'conditional_explanation_supported' if all(predictions.values()) else 'conditional_explanation_not_established',
        'controls':checks,'coverage':coverage,'field_groups':groups,'correct_peer_specificity':specificity,'cells':cells,'cases':error_rows,
        'oracle_max_abs':max(c['max_abs'] for c in audits),'oracle_max_relative_rms':max(c['relative_rms'] for c in audits),'composition_max_abs':composition,
        'independent_constants':program.independent_constant_count(),'native_coefficients_removed':0,'independent_worlds':16,
        'export_context_storage_bytes':sorted(cache_bytes),'derived_endpoint_dictionary_values':3072,'derived_endpoint_dictionary_bytes':24576,
        'runner_sha256':digest(SOURCE),'bound_sha256':EXPECTED,'rows_sha256':digest(ROWS),'wall_seconds':time.perf_counter()-started}
    OUT.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ('terminal','predictions','coverage','oracle_max_abs','wall_seconds')}),flush=True)


if __name__=='__main__':main()
