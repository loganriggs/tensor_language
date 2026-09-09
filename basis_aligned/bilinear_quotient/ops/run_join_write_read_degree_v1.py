#!/usr/bin/env python3
"""Exact cubic-numerator join read with native RMS; fixed degree hypotheses.
pred_a_exact: alpha=-1,0,.5,1,2 curves/native/sourceedit query outputs<=1e-9/1e-10.
pred_b_degree: forward C1/backward C2 plus C0 and native RMS predict centered
effects from alpha0 within .01 relativeRMS, floor1e-6, every nonzero scale/group.
pred_c_partition: normalization+linear+quadratic+cubic equal removal<=1e-9/1e-10.
768 opened requests,32worlds,6orders,4hops,B4FP64,1800s,256MiB/tensor.
All387968 weights retained; response cache and compile time charged; no rank fit.
"""
# BQGATE: EXPERIMENT pred_a_exact pred_b_degree pred_c_partition
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time

ROOT=Path('/workspace/tensor_language');POLY=ROOT/'basis_aligned/polynomial_causal';SOURCE=Path(__file__)
OUT=POLY/'JOIN_WRITE_READ_DEGREE_V1_RESULT.json';ROWS=POLY/'JOIN_WRITE_READ_DEGREE_V1_ROWS.pt'
PACKAGE=POLY/'EXACT_SOURCE_EDIT_V1_PROGRAM.pt';CHECKPOINT=ROOT/'runs_hop/attn4-rms-seed0/model.pt'
BOUND=[POLY/n for n in ('JOIN_WRITE_READ_DEGREE_V1_PREREGISTRATION.md','join_write_read_degree_reference.py',
    'join_contribution_context_reference.py','matcher_pair_transport_reference.py','matcher_factor_truth_table_reference.py',
    'suffix_join_middle_match_reference.py','suffix_join_writer_reference.py','causal_suffix_join_reference.py',
    'exact_source_edit_reference.py','field_intervention_metrics.py')]+[ROOT/'model.py',ROOT/'deep_model.py']
EXPECTED='90cdf8c1dc62c0ea550798d571d14e815d8034ace0d840090d4c9a095a23bcaf';ALPHAS=(-1.,0.,.5,1.,2.)


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    sys.path[:0]=[str(ROOT),str(POLY)];assert hashlib.sha256(b''.join(p.read_bytes() for p in BOUND)).hexdigest()==EXPECTED
    import torch
    import join_write_read_degree_reference as D
    import join_contribution_context_reference as J
    import matcher_pair_transport_reference as T
    import matcher_factor_truth_table_reference as F
    import suffix_join_middle_match_reference as R
    import exact_source_edit_reference as E
    import field_intervention_metrics as M
    torch.set_num_threads(2);checks=D.controls();assert checks['passed'];data=R.populations()
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps({'dryrun':True,'controls':checks,'requests':768,'checkpoint_opened':False}));return
    from hop_ablate import load
    os.chdir(ROOT);signal.alarm(1800);started=time.perf_counter();assert not OUT.exists() and not ROWS.exists()
    assert digest(PACKAGE)=='e69bff6562c48bd1b59366f118157720734d353626b0a1576626bdd437a1421a'
    assert digest(CHECKPOINT)=='c9388cc8b1ba7e95c6a043cbf4d01592987447867a50781d3babe31ce3f8358b'
    torch.backends.cuda.matmul.allow_tf32=False
    program=E.load_package(torch.load(PACKAGE,map_location='cpu',weights_only=True)).cuda().eval()
    model,_=load('attn4-rms-seed0');model=model.to(device='cuda',dtype=torch.float64).eval()
    audits=[];partitions=[];results={};saved={};finite=True;compile_seconds=0.;cache_bytes=0
    center=lambda x:x-x.mean(-1,keepdim=True)
    with torch.inference_mode():
        for pop,(tokens,masks,metadata) in data.items():
            exact={str(a):[] for a in ALPHAS};approx={str(a):[] for a in ALPHAS};terms={n:[] for n in ('normalization','linear','quadratic','cubic')};meta=[];curves=[]
            for i in range(0,len(tokens),4):
                head=1 if metadata[i]['orientation']=='B2_later' else 2;degree=1 if head==1 else 2
                tok=tokens[i:i+1].expand(4,-1).clone();tok[:,-1]=torch.arange(25,29);tok=tok.cuda();mask=masks[i:i+1].expand(4,-1,-1).cuda()
                context=program.prepare(tok);write=J.contributions(program.background,tok,{0:mask},{0:head})[0]
                _,scores=F.factors(program.background,tok,mask,head)
                torch.cuda.synchronize();t0=time.perf_counter();curve=D.compile_curve(program,context,write);torch.cuda.synchronize();compile_seconds+=time.perf_counter()-t0
                cache_bytes+=sum(v.numel()*v.element_size() for v in curve.values() if isinstance(v,torch.Tensor))
                for alpha in ALPHAS:
                    actual=D.evaluate(curve,alpha);physical=program.edit(context,(alpha-1)*write)[:,-1];native=T.native(model,tok,mask,head,alpha*scores)[:,-1]
                    audits.extend((M.correspondence(actual,physical),M.correspondence(actual,native)))
                    finite &= bool(torch.isfinite(actual).all());exact[str(alpha)].append(actual.cpu().clone());approx[str(alpha)].append(D.evaluate_degree(curve,alpha,degree).cpu().clone())
                parts=D.removal_terms(curve);partitions.append(M.correspondence(sum(parts.values()),D.evaluate(curve,1.)-D.evaluate(curve,0.)))
                for name,value in parts.items():terms[name].append(value.cpu().clone())
                curves.append({k:v.cpu().clone() if isinstance(v,torch.Tensor) else v for k,v in curve.items()})
                for hop in range(4):meta.append({**metadata[i],'hop':hop})
            exact={k:torch.cat(v) for k,v in exact.items()};approx={k:torch.cat(v) for k,v in approx.items()};terms={k:torch.cat(v) for k,v in terms.items()};groups={}
            for orientation in ('B2_later','B3_later'):
                groups[orientation]={}
                for hop in range(4):
                    sel=torch.tensor([m['orientation']==orientation and m['hop']==hop for m in meta]);scales={}
                    for alpha in ALPHAS:
                        if alpha==0:continue
                        key=str(alpha);effect=center(exact[key][sel]-exact['0.0'][sel]);den=max(float(effect.square().mean().sqrt()),1e-6)
                        error=center(approx[key][sel]-exact[key][sel]);lp=exact[key][sel].log_softmax(-1)
                        kl=(lp.exp()*(lp-approx[key][sel].log_softmax(-1))).sum(-1).clamp_min(0)
                        relative=float(error.square().mean().sqrt())/den
                        scales[key]={'native_effect_rms':float(effect.square().mean().sqrt()),'relative_effect_error':relative,
                            'query_mean_kl':float(kl.mean()),'query_p99_kl':float(torch.quantile(kl,.99)),'degree_passed':relative<=.01}
                    removal=center(exact['1.0'][sel]-exact['0.0'][sel]);den2=max(float(removal.square().sum()),1e-12)
                    attribution={name:{'rms_over_removal':float(center(v[sel]).norm())/max(float(removal.norm()),1e-6),
                        'projection_fraction':float((center(v[sel])*removal).sum())/den2} for name,v in terms.items()}
                    groups[orientation][str(hop)]={'requests':int(sel.sum()),'fixed_degree':1 if orientation=='B2_later' else 2,'scales':scales,'removal_terms':attribution}
            results[pop]=groups;saved[pop]={'metadata':meta,'exact_query_logits':exact,'approximate_query_logits':approx,'removal_terms':terms,'curves':curves}
            print(json.dumps({'population':pop,'hop3':{o:g['3'] for o,g in groups.items()}}),flush=True)
    predictions={'pred_a_exact':finite and all(a['passed'] for a in audits),'pred_b_degree':all(s['degree_passed'] for p in results.values() for o in p.values() for g in o.values() for s in g['scales'].values()),
        'pred_c_partition':all(a['passed'] for a in partitions)}
    torch.save(saved,ROWS)
    result={'experiment':'join_write_read_degree_v1','scope':'exact final-query response curves and fixed degree screen; no model reduction',
        'controls':checks,'predictions':predictions,'populations':results,'native_curve_max_abs':max(a['max_abs'] for a in audits),
        'native_curve_max_relative_rms':max(a['relative_rms'] for a in audits),'partition_max_abs':max(a['max_abs'] for a in partitions),
        'curve_compile_seconds_excluding_native_prefix_and_context':compile_seconds,'curve_tensor_cache_bytes_cohort':cache_bytes,
        'independent_worlds':32,'requests':768,'independent_constants':program.independent_constant_count(),'native_coefficients_removed':0,
        'runner_sha256':digest(SOURCE),'bound_sha256':EXPECTED,'rows_sha256':digest(ROWS),'wall_seconds':time.perf_counter()-started,
        'terminal':'instrument_invalid' if not predictions['pred_a_exact'] or not predictions['pred_c_partition'] else
        ('fixed_degree_read_screen_passed' if predictions['pred_b_degree'] else 'fixed_degree_read_rejected')}
    OUT.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ('terminal','predictions','native_curve_max_abs','wall_seconds')}),flush=True)


if __name__=='__main__':main()
