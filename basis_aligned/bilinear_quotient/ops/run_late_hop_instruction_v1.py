#!/usr/bin/env python3
"""Literal E(hop)/8 as a reusable late instruction field.
pred_a_mechanical: native/export everyarm andjoint=donor max<=1e-9/relative1e-10,
source identity, unchanged earlieroutputs, finite/live controls.
pred_b_distribution: directswap teacherKL mean<=.001,p99<=.01,querymean<=.001,
everypopulation/orderedhoppair. pred_c_effect: centered queryeffect relative<=.01.
pred_d_erasure: erasedhop differences/nativehop differences relativeRMS<=.01.
Null closes literal lateinstruction; no scale/head/producer or computed-only rescue.
Fullmodel output target,4608transitions16worlds,B8FP64,1800s,256MiB/tensor,
managedGPU only. All387968 constants andpairedcontexts charged, zero weights removed.
"""
# BQGATE: EXPERIMENT pred_a_mechanical pred_b_distribution pred_c_effect pred_d_erasure
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time

ROOT=Path('/workspace/tensor_language');POLY=ROOT/'basis_aligned/polynomial_causal';SOURCE=Path(__file__)
OUT=POLY/'LATE_HOP_INSTRUCTION_V1_RESULT.json';ROWS=POLY/'LATE_HOP_INSTRUCTION_V1_ROWS.pt'
PACKAGE=POLY/'EXACT_SOURCE_EDIT_V1_PROGRAM.pt';CHECKPOINT=ROOT/'runs_hop/attn4-rms-seed0/model.pt'
BOUND=[POLY/n for n in ('LATE_HOP_INSTRUCTION_V1_PREREGISTRATION.md','hop_instruction_field_reference.py',
    'exact_source_edit_reference.py','forward_endpoint_random_layout_reference.py','forward_endpoint_program_reference.py','field_intervention_metrics.py')]+[ROOT/'model.py',ROOT/'deep_model.py']
EXPECTED='86846e27f304f7b75d7f44911251166a58d3389ee6429b324836799ce03cb675'


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def bound_hash():return hashlib.sha256(b''.join(p.read_bytes() for p in BOUND)).hexdigest()


def main():
    sys.path[:0]=[str(ROOT),str(POLY)];assert bound_hash()==EXPECTED
    import torch
    import hop_instruction_field_reference as H
    import exact_source_edit_reference as E
    import field_intervention_metrics as M
    import forward_endpoint_random_layout_reference as R
    torch.set_num_threads(2);checks=H.controls();assert checks['passed']
    data={pop:worlds[:8] for pop,worlds in R.populations(seeds=(33909,33910)).items()}
    for worlds in data.values():
        for w in worlds:
            for query in range(24):assert torch.equal(w['tokens'][query*4:query*4+4,:-1],w['tokens'][query*4:query*4+1,:-1].expand(4,-1))
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps({'dryrun':True,'controls':checks,'transitions':4608,'checkpoint_opened':False}));return
    from hop_ablate import load
    os.chdir(ROOT);signal.alarm(1800);started=time.perf_counter();assert not OUT.exists() and not ROWS.exists()
    assert digest(PACKAGE)=='e69bff6562c48bd1b59366f118157720734d353626b0a1576626bdd437a1421a'
    assert digest(CHECKPOINT)=='c9388cc8b1ba7e95c6a043cbf4d01592987447867a50781d3babe31ce3f8358b'
    torch.backends.cuda.matmul.allow_tf32=False;program=E.load_package(torch.load(PACKAGE,map_location='cpu',weights_only=True)).cuda().eval()
    model,_=load('attn4-rms-seed0');model=model.to(device='cuda',dtype=torch.float64).eval()
    audits=[];earlier=0.;results={};rows={};cache_bytes=set();finite=True
    with torch.inference_mode():
        for pop,worlds in data.items():
            saved={};metadata=[];kls=[];tokens_all=[]
            for w in worlds:
                tokens_all.append(w['tokens'])
                for start in range(0,24,8):
                    contexts={};nativebases={};native={};tokens={};erased={}
                    for hop in range(4):
                        ix=torch.arange(start,start+8)*4+hop;tok=w['tokens'][ix].cuda();tokens[hop]=tok
                        context=program.prepare(tok);contexts[hop]=context;native[hop]=model(tok)
                        audits.append(M.correspondence(context['logits'],native[hop]))
                        x=model.embed(tok)
                        for layer in model.layers[:3]:x=layer(x)
                        nativebases[hop]=x;erase=H.removal(program,context,tok);erased[hop]=program.edit(context,erase)
                        audits.append(M.correspondence(erased[hop],model.head(model.layers[-1](x+erase))))
                        audits.append(M.correspondence(context['x'][:,:-1],contexts[0]['x'][:,:-1]))
                        vals=[v for v in context.values() if torch.is_tensor(v)]+list(context['features'].values())
                        cache_bytes.add(sum({v.untyped_storage().data_ptr():v.untyped_storage().nbytes() for v in vals}.values())//len(tok))
                    for recipient in range(4):
                        for donor in range(4):
                            if recipient==donor:continue
                            a=contexts[recipient];b=contexts[donor];direct,computed,total=H.fields(program,a,b,tokens[recipient],tokens[donor])
                            arms={'native':a['logits'],'removal':erased[recipient],'donor':native[donor],'donor_erased':erased[donor]}
                            for name,delta in (('direct',direct),('computed',computed),('joint',total),('identity',direct*0)):
                                arms[name]=program.edit(a,delta)
                                audits.append(M.correspondence(arms[name],model.head(model.layers[-1](nativebases[recipient]+delta))))
                                earlier=max(earlier,float((arms[name][:,:-1]-a['logits'][:,:-1]).abs().max()))
                            audits.append(M.correspondence(arms['joint'],native[donor]))
                            arms['interaction']=arms['joint']-arms['direct']-arms['computed']+arms['native']
                            for name,value in arms.items():
                                finite &= bool(torch.isfinite(value).all());saved.setdefault(name,[]).append(value[:,-1].cpu().clone())
                            lp=native[donor].log_softmax(-1);lq=arms['direct'].log_softmax(-1)
                            kls.append((lp.exp()*(lp-lq)).sum(-1).clamp_min(0).cpu())
                            metadata.extend({'world':w['world'],'query':query,'recipient_hop':recipient,'donor_hop':donor,
                                'answer':w['metadata'][query*4+recipient]['answer'],'donor_answer':w['metadata'][query*4+donor]['answer']} for query in range(start,start+8))
            logits={k:torch.cat(v) for k,v in saved.items()};kl=torch.cat(kls);groups={}
            for recipient in range(4):
                for donor in range(4):
                    if recipient==donor:continue
                    sel=[m['recipient_hop']==recipient and m['donor_hop']==donor for m in metadata];s=torch.tensor(sel);assert int(s.sum())==192
                    center=lambda x:x-x.mean(-1,keepdim=True)
                    target=center(logits['donor'][s]-logits['native'][s]);den=max(float(target.square().mean().sqrt()),1e-6)
                    effect=center(logits['direct'][s]-logits['native'][s]);erasure=center(logits['donor_erased'][s]-logits['removal'][s])
                    interaction=center(logits['interaction'][s])
                    groups[str(recipient)+'_to_'+str(donor)]={'n':int(s.sum()),'all_mean_kl':float(kl[s].mean()),'token_p99_kl':float(torch.quantile(kl[s].flatten(),.99)),
                        'query_mean_kl':float(kl[s,-1].mean()),'maximum_kl':float(kl[s].max()),'effect_relative_rms':float((effect-target).square().mean().sqrt())/den,
                        'erasure_relative_rms':float(erasure.square().mean().sqrt())/den,'interaction_centered_rms':float(interaction.square().mean().sqrt()),
                        'panels':{arm:M.panel(logits,arm,metadata,sel,'donor_answer') for arm in ('direct','computed','joint','donor','removal')}}
            results[pop]={'groups':groups};rows[pop]={'metadata':metadata,'query_logits':logits,'token_kl':kl,'native_tokens':torch.cat(tokens_all)}
            print(json.dumps({'population':pop,'max_query_kl':max(g['query_mean_kl'] for g in groups.values()),
                'max_effect_error':max(g['effect_relative_rms'] for g in groups.values()),'max_erasure_error':max(g['erasure_relative_rms'] for g in groups.values())}),flush=True)
    groups=[g for p in results.values() for g in p['groups'].values()]
    predictions={'pred_a_mechanical':checks['passed'] and finite and all(a['passed'] for a in audits) and earlier<=1e-9,
        'pred_b_distribution':all(g['all_mean_kl']<=.001 and g['token_p99_kl']<=.01 and g['query_mean_kl']<=.001 for g in groups),
        'pred_c_effect':all(g['effect_relative_rms']<=.01 for g in groups),'pred_d_erasure':all(g['erasure_relative_rms']<=.01 for g in groups)}
    torch.save(rows,ROWS)
    result={'experiment':'late_hop_instruction_v1','scope':'literal instruction field full-model prediction/removal; no computed-only rescue or structural saving assumed',
        'controls':checks,'predictions':predictions,'populations':results,'oracle_max_abs':max(a['max_abs'] for a in audits),
        'oracle_max_relative_rms':max(a['relative_rms'] for a in audits),'earlier_output_max_abs':earlier,
        'context_bytes_per_input':sorted(cache_bytes),'independent_constants':program.independent_constant_count(),'native_coefficients_removed':0,
        'independent_worlds':16,'transitions':4608,'runner_sha256':digest(SOURCE),'bound_sha256':EXPECTED,'rows_sha256':digest(ROWS),
        'terminal':'mechanically_invalid' if not predictions['pred_a_mechanical'] else 'late_instruction_supported' if all(predictions.values()) else 'late_instruction_not_established',
        'wall_seconds':time.perf_counter()-started}
    OUT.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ('terminal','predictions','oracle_max_abs','wall_seconds')}),flush=True)


if __name__=='__main__':main()
