#!/usr/bin/env python3
"""Fresh exact document/local initializer extraction, removals and shared reuse.
pred_a_extraction: fulloutput native closure<=1e-9/relative1e-10, native KL<=1e-12.
pred_b_removals: all three compiled/native removal effects same1e-9/1e-10 bars.
pred_c_joint: neither/native closure, zeroquery absorbing and mixedeffect replay.
pred_d_reuse: same4document summaries across24queries, same96local states across
16worlds; cached/direct equality<=1e-9/1e-10. No semanticselectivity assumed.
1536requests,4keepcells,B8FP64,1800s,256MiB/tensor; managedGPU only.
All387968 constants retained; generic additive/CSE reuse is infrastructure,
not novel structuralmodel reduction. Cache/edge prices reported separately.
"""
# BQGATE: EXPERIMENT pred_a_extraction pred_b_removals pred_c_joint pred_d_reuse
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time

ROOT=Path('/workspace/tensor_language');POLY=ROOT/'basis_aligned/polynomial_causal';SOURCE=Path(__file__)
OUT=POLY/'QUERY_INITIALIZER_FACTORIZATION_V1_RESULT.json';ROWS=POLY/'QUERY_INITIALIZER_FACTORIZATION_V1_ROWS.pt'
PACKAGE=POLY/'EXACT_SOURCE_EDIT_V1_PROGRAM.pt';CHECKPOINT=ROOT/'runs_hop/attn4-rms-seed0/model.pt'
BOUND=[POLY/n for n in ('QUERY_INITIALIZER_FACTORIZATION_V1_PREREGISTRATION.md','query_initializer_factorization_reference.py',
    'local_query_initializer_reference.py','exact_source_edit_reference.py','forward_endpoint_random_layout_reference.py',
    'forward_endpoint_program_reference.py','field_intervention_metrics.py')]+[ROOT/'model.py',ROOT/'deep_model.py']
EXPECTED='0485aacf24ef49a0f67d59f1482cb1ef7867913f3c8cad8d6fcb94859fad1e35'


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def bound_hash():return hashlib.sha256(b''.join(p.read_bytes() for p in BOUND)).hexdigest()


def main():
    sys.path[:0]=[str(ROOT),str(POLY)];assert bound_hash()==EXPECTED
    import torch
    import query_initializer_factorization_reference as Q
    import exact_source_edit_reference as E
    import field_intervention_metrics as M
    import forward_endpoint_random_layout_reference as R
    torch.set_num_threads(2);checks=Q.controls();assert checks['passed']
    data={pop:worlds[:8] for pop,worlds in R.populations(seeds=(34909,34910)).items()}
    local_tokens=data['iid'][0]['tokens'][:,-3:]
    for worlds in data.values():
        for w in worlds:assert torch.equal(w['tokens'][:,-3:],local_tokens)
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps({'dryrun':True,'controls':checks,'requests':1536,'local_inputs':96,'checkpoint_opened':False}));return
    from hop_ablate import load
    os.chdir(ROOT);signal.alarm(1800);started=time.perf_counter();assert not OUT.exists() and not ROWS.exists()
    assert digest(PACKAGE)=='e69bff6562c48bd1b59366f118157720734d353626b0a1576626bdd437a1421a'
    assert digest(CHECKPOINT)=='c9388cc8b1ba7e95c6a043cbf4d01592987447867a50781d3babe31ce3f8358b'
    torch.backends.cuda.matmul.allow_tf32=False;program=E.load_package(torch.load(PACKAGE,map_location='cpu',weights_only=True)).cuda().eval()
    model,_=load('attn4-rms-seed0');model=model.to(device='cuda',dtype=torch.float64).eval()
    audits=[];removal_audits=[];reuse_audits=[];mixed_audits=[];results={};rows={};zero=earlier=0.;finite=True
    with torch.inference_mode():
        local_cache=Q.local_state(program,data['iid'][0]['tokens'].cuda())
        for pop,worlds in data.items():
            saved={};metadata=[];kls={};tokens_all=[];summaries=[]
            for w in worlds:
                tokens_all.append(w['tokens']);metadata.extend(w['metadata'])
                binding=w['tokens'][0,:48].cuda().expand(4,-1);hop_tokens=torch.arange(25,29,device='cuda')
                summary_cache=Q.document_summary(program,binding,hop_tokens);summaries.append(summary_cache.cpu().clone())
                for i in range(0,96,8):
                    tok=w['tokens'][i:i+8].cuda();hop=tok[:,-1]-25;summary=summary_cache[hop];local=local_cache[i:i+8]
                    reuse_audits.append(M.correspondence(summary,Q.document_summary(program,tok[:,:48],tok[:,-1])))
                    reuse_audits.append(M.correspondence(local,Q.local_state(program,tok)))
                    native=model(tok);arms={keep:Q.execute(program,tok,keep,summary,local) for keep in range(4)}
                    oracle={keep:Q.native(model,tok,keep) for keep in range(4)}
                    for keep in range(4):
                        audits.append(M.correspondence(arms[keep],oracle[keep]))
                        earlier=max(earlier,float((arms[keep][:,:-1]-native[:,:-1]).abs().max()))
                        if keep!=3:removal_audits.append(M.correspondence(arms[keep]-arms[3],oracle[keep]-native))
                    audits.append(M.correspondence(arms[3],native));zero=max(zero,float(arms[0][:,-1].abs().max()),float(oracle[0][:,-1].abs().max()))
                    mixed=arms[0]-arms[1]-arms[2]+arms[3];native_mixed=oracle[0]-oracle[1]-oracle[2]+native
                    mixed_audits.append(M.correspondence(mixed,native_mixed))
                    outputs={'native':native,**{str(k):v for k,v in arms.items()},'mixed':mixed}
                    for name,value in outputs.items():
                        finite &= bool(torch.isfinite(value).all());saved.setdefault(name,[]).append(value[:,-1].cpu().clone())
                    lp=native.log_softmax(-1)
                    for keep,value in arms.items():kls.setdefault(str(keep),[]).append((lp.exp()*(lp-value.log_softmax(-1))).sum(-1).clamp_min(0).cpu())
            logits={k:torch.cat(v) for k,v in saved.items()};kl={k:torch.cat(v) for k,v in kls.items()};groups={}
            for hop in range(4):
                sel=[m['hop']==hop for m in metadata];s=torch.tensor(sel);groups[str(hop)]={}
                base=logits['native'][s];base=base-base.mean(-1,keepdim=True);den=max(float(base.square().mean().sqrt()),1e-6)
                for keep in range(4):
                    arm=str(keep);effect=logits[arm][s]-logits['native'][s];effect-=effect.mean(-1,keepdim=True)
                    groups[str(hop)][arm]={**M.panel(logits,arm,metadata,sel),'all_mean_kl':float(kl[arm][s].mean()),
                        'query_mean_kl':float(kl[arm][s,-1].mean()),'query_vector_relative_change':float(effect.square().mean().sqrt())/den}
                mixed=logits['mixed'][s];mixed-=mixed.mean(-1,keepdim=True)
                groups[str(hop)]['mixed_centered_rms']=float(mixed.square().mean().sqrt())
            results[pop]={'groups':groups,'native_mean_kl':float(kl['3'].mean())}
            rows[pop]={'metadata':metadata,'query_logits':logits,'token_kl':kl,'native_tokens':torch.cat(tokens_all),'document_summaries':torch.stack(summaries)}
            print(json.dumps({'population':pop,'native_mean_kl':results[pop]['native_mean_kl'],
                'hop3_remove_summary':groups['3']['2'],'hop3_remove_local':groups['3']['1']}),flush=True)
    predictions={'pred_a_extraction':checks['passed'] and finite and all(a['passed'] for a in audits) and earlier<=1e-9 and all(p['native_mean_kl']<=1e-12 for p in results.values()),
        'pred_b_removals':all(a['passed'] for a in removal_audits),'pred_c_joint':zero<=1e-9 and all(a['passed'] for a in mixed_audits),
        'pred_d_reuse':all(a['passed'] for a in reuse_audits)}
    torch.save({'populations':rows,'local_cache':local_cache.cpu().clone()},ROWS)
    maxima=lambda entries:{'max_abs':max(a['max_abs'] for a in entries),'max_relative_rms':max(a['relative_rms'] for a in entries)}
    result={'experiment':'query_initializer_factorization_v1','scope':'exact partial extraction/CSE reuse; no novel structuralmodel reduction',
        'controls':checks,'predictions':predictions,'populations':results,'output_oracle':maxima(audits),'removal_oracle':maxima(removal_audits),
        'mixed_oracle':maxima(mixed_audits),'reuse_oracle':maxima(reuse_audits),'zero_query_max_abs':zero,'earlier_output_max_abs':earlier,
        'independent_constants':program.independent_constant_count(),'native_coefficients_removed':0,'independent_worlds':16,'requests':1536,
        'document_summary_bytes_per_document':4*128*8,'global_local_cache_bytes':local_cache.numel()*local_cache.element_size(),
        'query_initializer_edges_uncached_cohort':1536*4*51,'query_initializer_edges_cached_cohort':16*4*4*48+96*4*3,
        'price_scope':'initializer source contractions only, excluding all other native execution; standard CSE baseline, no single-request speedup claim',
        'runner_sha256':digest(SOURCE),'bound_sha256':EXPECTED,'rows_sha256':digest(ROWS),'wall_seconds':time.perf_counter()-started,
        'terminal':'exact_initializer_extraction_validated' if all(predictions.values()) else 'initializer_instrument_not_validated'}
    OUT.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ('terminal','predictions','output_oracle','wall_seconds')}),flush=True)


if __name__=='__main__':main()
