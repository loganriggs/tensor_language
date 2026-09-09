#!/usr/bin/env python3
"""Fresh hop control in Q1/Q2, all4608 ordered request transitions.
pred_a_mechanical: all four query-factor cells/native oracle and both=donor read
max<=1e-9/relative1e-10, source identity, finite controls.
pred_b_q1: Q1-only centered effect relativeRMS<=.01 everypop/orderedhoppair.
pred_c_q2: same for Q2-only. pred_d_composition: explicit mixed bilinear term
closes both-Q output<=1e-9. Null closes separate native-factor hop-control claims;
no head/tasksubset/rank/decoder fit. Initial48binding-read target only, no suffix
or residual/fullmodel equivalence. B8FP64,16worlds,1800s,256MiB/tensor,managedGPU.
All387968 constants andpaired token-derived contexts charged; no weights removed.
"""
# BQGATE: EXPERIMENT pred_a_mechanical pred_b_q1 pred_c_q2 pred_d_composition
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time

ROOT=Path('/workspace/tensor_language');POLY=ROOT/'basis_aligned/polynomial_causal';SOURCE=Path(__file__)
OUT=POLY/'HOP_QUERY_FACTOR_TRANSFER_V1_RESULT.json';ROWS=POLY/'HOP_QUERY_FACTOR_TRANSFER_V1_ROWS.pt'
PACKAGE=POLY/'EXACT_SOURCE_EDIT_V1_PROGRAM.pt';CHECKPOINT=ROOT/'runs_hop/attn4-rms-seed0/model.pt'
BOUND=[POLY/n for n in ('HOP_QUERY_FACTOR_TRANSFER_V1_PREREGISTRATION.md','query_factor_transfer_reference.py',
    'exact_source_edit_reference.py','forward_endpoint_random_layout_reference.py','forward_endpoint_program_reference.py','field_intervention_metrics.py')]+[ROOT/'model.py',ROOT/'deep_model.py']
EXPECTED='ce3ff7124dd9760274747090d57a4a0fdf09b97297d6b1cd9891999ecacbc3b2'


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def bound_hash():return hashlib.sha256(b''.join(p.read_bytes() for p in BOUND)).hexdigest()


def main():
    sys.path[:0]=[str(ROOT),str(POLY)];assert bound_hash()==EXPECTED
    import torch
    import query_factor_transfer_reference as Q
    import exact_source_edit_reference as E
    import field_intervention_metrics as M
    import forward_endpoint_random_layout_reference as R
    torch.set_num_threads(2);checks=Q.controls();assert checks['passed']
    data={pop:ws[:8] for pop,ws in R.populations(seeds=(31909,31910)).items()}
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
    audits=[];composition=0.;results={};rows={};cache_bytes=set()
    with torch.inference_mode():
        for pop,worlds in data.items():
            storage={};metadata=[];capability={h:[] for h in range(4)};tokens_all=[]
            for w in worlds:
                tokens_all.append(w['tokens'])
                for start in range(0,24,8):
                    contexts={};nativebases={};native_reads={}
                    for hop in range(4):
                        ix=torch.arange(start,start+8)*4+hop;tok=w['tokens'][ix].cuda();context=program.prepare(tok);contexts[hop]=context
                        native=model(tok);audits.append(M.correspondence(context['logits'],native))
                        gold=torch.tensor([w['metadata'][int(i)]['answer'] for i in ix],device='cuda')
                        capability[hop].append((native[:,-1].argmax(-1)==gold).cpu())
                        x=model.embed(tok)
                        for layer in model.layers[:3]:x=layer(x)
                        nativebases[hop]=x;native_reads[hop]=Q.native(model,x,x,0)
                        audits.append(M.correspondence(context['x'][:,:48],contexts[0]['x'][:,:48]))
                        vals=[v for v in context.values() if torch.is_tensor(v)]+list(context['features'].values())
                        cache_bytes.add(sum({v.untyped_storage().data_ptr():v.untyped_storage().nbytes() for v in vals}.values())//len(tok))
                    for recipient in range(4):
                        for donor in range(4):
                            if recipient==donor:continue
                            a=contexts[recipient];b=contexts[donor]
                            arms={mask:Q.read(program,a,b,mask) for mask in range(4)}
                            for mask in range(4):
                                oracle=native_reads[recipient] if mask==0 else Q.native(model,nativebases[recipient],nativebases[donor],mask)
                                audits.append(M.correspondence(arms[mask],oracle))
                            target=Q.read(program,b,b,0);audits.append(M.correspondence(arms[3],target))
                            q={k:b['features'][k][:,-1:]-a['features'][k][:,-1:] for k in ('q1','q2')}
                            source={k:v[:,:48] for k,v in a['features'].items()}
                            mixed=(.5*program.aggregate(q,source,a['positions'][-1:],a['positions'][:48])@program.folded.T)[:,0]
                            composition=max(composition,float((arms[1]+arms[2]-arms[0]+mixed-arms[3]).abs().max()))
                            for name,value in {**{str(k):v for k,v in arms.items()},'mixed':mixed,'donor':target}.items():storage.setdefault(name,[]).append(value.cpu().clone())
                            metadata.extend({'world':w['world'],'query':query,'recipient_hop':recipient,'donor_hop':donor} for query in range(start,start+8))
            values={k:torch.cat(v) for k,v in storage.items()};groups={}
            for recipient in range(4):
                for donor in range(4):
                    if recipient==donor:continue
                    sel=torch.tensor([m['recipient_hop']==recipient and m['donor_hop']==donor for m in metadata]);assert int(sel.sum())==192
                    target=values['donor'][sel]-values['0'][sel];target-=target.mean(-1,keepdim=True);den=max(float(target.square().mean().sqrt()),1e-6)
                    errors={}
                    for mask in (1,2):
                        predicted=values[str(mask)][sel]-values['0'][sel];predicted-=predicted.mean(-1,keepdim=True)
                        errors['q'+str(mask)+'_relative_rms']=float((predicted-target).square().mean().sqrt())/den
                    mixed=values['mixed'][sel];mixed-=mixed.mean(-1,keepdim=True)
                    groups[str(recipient)+'_to_'+str(donor)]={'n':int(sel.sum()),**errors,'target_centered_rms':den,
                        'mixed_centered_rms':float(mixed.square().mean().sqrt()),'mixed_relative_to_target':float(mixed.square().mean().sqrt())/den}
            results[pop]={'groups':groups,'native_accuracy':{str(h):float(torch.cat(v).double().mean()) for h,v in capability.items()}}
            rows[pop]={'metadata':metadata,'binding_reads':values,'native_tokens':torch.cat(tokens_all)}
            print(json.dumps({'population':pop,**results[pop]}),flush=True)
    groups=[g for p in results.values() for g in p['groups'].values()]
    predictions={'pred_a_mechanical':checks['passed'] and all(a['passed'] for a in audits),
        'pred_b_q1':all(g['q1_relative_rms']<=.01 for g in groups),'pred_c_q2':all(g['q2_relative_rms']<=.01 for g in groups),
        'pred_d_composition':composition<=1e-9}
    torch.save(rows,ROWS)
    result={'experiment':'hop_query_factor_transfer_v1','scope':'paired-context initialbinding read only; no full-model or independent hop program equivalence',
        'controls':checks,'predictions':predictions,'populations':results,'oracle_max_abs':max(a['max_abs'] for a in audits),
        'oracle_max_relative_rms':max(a['relative_rms'] for a in audits),'composition_max_abs':composition,'context_bytes_per_input':sorted(cache_bytes),
        'independent_constants':program.independent_constant_count(),'native_coefficients_removed':0,'paired_contexts':2,'independent_worlds':16,'transitions':4608,
        'terminal':'mechanically_invalid' if not predictions['pred_a_mechanical'] else 'separate_query_factor_control_supported' if predictions['pred_b_q1'] or predictions['pred_c_q2'] else 'separate_query_factor_control_not_established',
        'runner_sha256':digest(SOURCE),'bound_sha256':EXPECTED,'rows_sha256':digest(ROWS),'wall_seconds':time.perf_counter()-started}
    OUT.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ('terminal','predictions','oracle_max_abs','wall_seconds')}),flush=True)


if __name__=='__main__':main()
