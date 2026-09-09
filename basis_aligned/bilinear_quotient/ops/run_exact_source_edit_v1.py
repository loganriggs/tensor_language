#!/usr/bin/env python3
"""Validate token-derived exact source editing, standalone export and literal price.

pred_a_preparation: CPU controls and native token-to-logit closure1e-9/relative1e-10.
pred_b_edits: fresh disjoint/overlap/half-dose all-logit closure at same bars;
additional full distribution meanKL<=1e-3,p99<=1e-2 all/query.
pred_c_export: fresh CPU process replay passes with original checkpoint reads denied.
pred_d_price:387968 independent constants, final O absent, explicit cache/timing price.
No semantic-reduction or speedup prediction.32worlds384queryvariants17arms,
B4FP64,1800s,<256MiB/tensor, managedGPU; export replay CPU2threads.
"""
# BQGATE: EXPERIMENT pred_a_preparation pred_b_edits pred_c_export pred_d_price
import hashlib
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time

ROOT=Path('/workspace/tensor_language');POLY=ROOT/'basis_aligned/polynomial_causal';SOURCE=Path(__file__)
OUT=POLY/'EXACT_SOURCE_EDIT_V1_RESULT.json';ROWS=POLY/'EXACT_SOURCE_EDIT_V1_ROWS.pt'
PACKAGE=POLY/'EXACT_SOURCE_EDIT_V1_PROGRAM.pt';FIXTURES=POLY/'EXACT_SOURCE_EDIT_V1_REPLAY_INPUTS.pt'
CPU_RESULT=POLY/'EXACT_SOURCE_EDIT_V1_CPU_REPLAY.json'
CPU_RUNNER=SOURCE.parent/'replay_exact_source_edit_v1_cpu.py'
CHECKPOINT=ROOT/'runs_hop/attn4-rms-seed0/model.pt'
EXPECTED_CHECKPOINT='c9388cc8b1ba7e95c6a043cbf4d01592987447867a50781d3babe31ce3f8358b'
BOUND=[POLY/n for n in ('exact_source_edit_reference.py','source_support_interaction_reference.py',
                       'join_contribution_context_reference.py','join_value_producer_reference.py','EXACT_SOURCE_EDIT_V1_PREREGISTRATION.md')]+[CPU_RUNNER]
EXPECTED='00b30d8fac7db8cb946d6a34061ed71be23ab20f88baea305f227cda5486c8a2'


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def bound_hash():return hashlib.sha256(b''.join(p.read_bytes() for p in BOUND)).hexdigest()
def relative(a,b):return float((a-b).square().mean().sqrt())/max(float(b.square().mean().sqrt()),1e-6)


def cache_bytes(value):
    seen={}
    def visit(x):
        if isinstance(x,dict):
            for v in x.values():visit(v)
        elif hasattr(x,'untyped_storage'):
            storage=x.untyped_storage();seen[storage.data_ptr()]=storage.nbytes()
    visit(value);return sum(seen.values())


def benchmark(torch,model,program,tokens,native_base,context,edits):
    d2=edits['disjoint_1'];d6=edits['disjoint_7']
    counts={'k2':int(d2.ne(0).any((0,2)).sum()),'k6':int(d6.ne(0).any((0,2)).sum())}
    handle=model.layers[2].register_forward_hook(lambda _m,_a,out:out+d6)
    methods={'native_full_edit_k6':lambda:model(tokens), 'program_prepare':lambda:program.prepare(tokens),
             'native_cached_suffix_k2':lambda:model.head(model.layers[-1](native_base+d2)),
             'native_cached_suffix_k6':lambda:model.head(model.layers[-1](native_base+d6)),
             'program_edit_k2':lambda:program.edit(context,d2),'program_edit_k6':lambda:program.edit(context,d6)}
    samples={k:[] for k in methods}
    try:
        for fn in methods.values():
            for _ in range(3):fn()
        torch.cuda.synchronize()
        for _ in range(20):
            for name,fn in methods.items():
                torch.cuda.synchronize();start=time.perf_counter();value=fn();torch.cuda.synchronize()
                samples[name].append((time.perf_counter()-start)*1000);del value
    finally:handle.remove()
    peak={}
    for name,delta in (('k2',d2),('k6',d6)):
        torch.cuda.synchronize();before=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
        value=program.edit(context,delta);torch.cuda.synchronize()
        peak[name]=torch.cuda.max_memory_allocated()-before;del value
    return {'milliseconds':{k:{'median':statistics.median(v),'min':min(v),'max':max(v),'trials':len(v)} for k,v in samples.items()},
            'incremental_peak_allocated_bytes':peak,'context_bytes':cache_bytes(context),
            'shape':{'batch':len(tokens),'tokens':tokens.shape[1],'residual':native_base.shape[-1],'heads':4,'head_width':32},
            'observed_edited_positions':counts,
            'projection_rows_per_matrix':{'native_suffix':len(tokens)*tokens.shape[1],**{'edit_'+k:len(tokens)*v for k,v in counts.items()}},
            'attention_pair_cells_per_head_batch':{'native_suffix':tokens.shape[1]**2,**{'edit_'+k+'_total':3*tokens.shape[1]*v for k,v in counts.items()}},
            'note':'Three source/query aggregate calls per edit; dense folded readout/residual correction and cache copies remain. Timing includes Python and CUDA synchronization; prefix preparation is separate.'}


def run(torch,R,D,J,checks):
    from hop_ablate import load
    os.chdir(ROOT);signal.alarm(1800);started=time.perf_counter()
    assert not any(p.exists() for p in (OUT,ROWS,PACKAGE,FIXTURES,CPU_RESULT)) and digest(CHECKPOINT)==EXPECTED_CHECKPOINT
    torch.backends.cuda.matmul.allow_tf32=False
    model,config=load('attn4-rms-seed0');model=model.to(device='cuda',dtype=torch.float64).eval()
    program=R.SourceEditProgram(model).eval();results={};row_data={};fixtures=[];bench=None
    prep_max=prep_rel=0.;finite=True
    with torch.inference_mode():
        for pop,worlds in D.populations(seeds=(21909,21910)).items():
            errors={};kl_rows={};query_rows={};tokens_saved=[]
            for wi,w in enumerate(worlds):
                masks={j:m.cuda() for j,m in w['masks'].items()};heads=w['heads'];tokall=w['tokens'].cuda()
                tokens_saved.append(w['tokens'])
                for i in range(0,12,4):
                    tok=tokall[i:i+4];context=program.prepare(tok);native=model(tok)
                    prep_max=max(prep_max,float((context['logits']-native).abs().max()));prep_rel=max(prep_rel,relative(context['logits'],native))
                    native_base=model.embed(tok)
                    for layer in model.layers[:-1]:native_base=layer(native_base)
                    own_edits=R.make_edits(program.background,tok,masks,heads)
                    native_edits=R.make_edits(model,tok,masks,heads)
                    expected_fixture={}
                    for arm,delta in own_edits.items():
                        actual=program.edit(context,delta)
                        if arm.startswith('disjoint_'):
                            subset=int(arm.rsplit('_',1)[1]);selected=tuple(j for j in range(3) if subset&(1<<j))
                            with J.intervene(model,masks,heads,selected):expected=model(tok)
                        else:expected=model.head(model.layers[-1](native_base+native_edits[arm]))
                        finite &= bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
                        e=errors.setdefault(arm,{'max_abs_logit':0.,'max_relative_rms':0.})
                        e['max_abs_logit']=max(e['max_abs_logit'],float((actual-expected).abs().max()))
                        e['max_relative_rms']=max(e['max_relative_rms'],relative(actual,expected))
                        lp,lq=expected.log_softmax(-1),actual.log_softmax(-1)
                        kl_rows.setdefault(arm,[]).append((lp.exp()*(lp-lq)).sum(-1).cpu())
                        query_rows.setdefault(arm,[]).append(actual[:,-1].cpu().clone())
                        if wi==0 and i==0:expected_fixture[arm]=expected.cpu().clone()
                    if expected_fixture:fixtures.append({'population':pop,'tokens':tok.cpu().clone(),'masks':w['masks'],'heads':heads,'expected':expected_fixture})
                    if bench is None:bench=(tok.clone(),native_base.clone(),context,own_edits)
            for arm,e in errors.items():
                kl=torch.cat(kl_rows[arm]);query=kl[:,-1]
                e['all_distribution']={'mean_kl':float(kl.mean()),'p99_kl':float(torch.quantile(kl.flatten(),.99)),'max_kl':float(kl.max())}
                e['query_distribution']={'mean_kl':float(query.mean()),'p99_kl':float(torch.quantile(query,.99)),'max_kl':float(query.max())}
                e['passed']=e['max_abs_logit']<=1e-9 and e['max_relative_rms']<=1e-10 and all(d['mean_kl']<=1e-3 and d['p99_kl']<=1e-2 for d in (e['all_distribution'],e['query_distribution']))
            results[pop]={'arms':errors};row_data[pop]={'tokens':torch.cat(tokens_saved),'query_logits':{k:torch.cat(v) for k,v in query_rows.items()},'token_kl':{k:torch.cat(v) for k,v in kl_rows.items()}}
            print(json.dumps({'population':pop,'max_abs_logit':max(e['max_abs_logit'] for e in errors.values()),'all_arms_passed':all(e['passed'] for e in errors.values())}),flush=True)
        price=benchmark(torch,model,program,*bench)
        package=R.export_package(program,config);torch.save(package,PACKAGE);torch.save(fixtures,FIXTURES);torch.save(row_data,ROWS)
    price.update({'native_independent_constants':sum(p.numel() for p in model.parameters()),'program_independent_constants':program.independent_constant_count(),
                  'fixed_buffer_bytes':sum(v.numel()*v.element_size() for v in package['fixed_buffers'].values()),'package_file_bytes':PACKAGE.stat().st_size,
                  'native_O_physically_absent':not hasattr(program.background.layers[-1],'o'),'generic_fold_saving':12672,'new_semantic_parameter_saving':0})
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=''
    replay=subprocess.run(['/venv/main/bin/python',str(CPU_RUNNER)],cwd=ROOT,env=env,capture_output=True,text=True,timeout=120)
    cpu=json.loads(CPU_RESULT.read_text()) if CPU_RESULT.exists() else {'passed':False,'error':replay.stderr[-2000:]}
    pred_a=checks['passed'] and finite and prep_max<=1e-9 and prep_rel<=1e-10
    pred_b=pred_a and all(a['passed'] for p in results.values() for a in p['arms'].values())
    pred_c=pred_a and replay.returncode==0 and cpu['passed']
    pred_d=price['program_independent_constants']==387968 and price['native_independent_constants']==400640 and price['native_O_physically_absent'] and price['context_bytes']<256*1024**2
    receipt={'experiment':'exact_source_edit_v1','runner_sha256':digest(SOURCE),'bound_sha256':EXPECTED,'checkpoint_sha256':EXPECTED_CHECKPOINT,
             'controls':checks,'preparation':{'max_abs_logit':prep_max,'max_relative_rms':prep_rel},'populations':results,'cpu_export_replay':cpu,'price':price,
             'program':str(PACKAGE.relative_to(ROOT)),'program_sha256':digest(PACKAGE),'rows':str(ROWS.relative_to(ROOT)),'rows_sha256':digest(ROWS),
             'predictions':{'pred_a_preparation':bool(pred_a),'pred_b_edits':bool(pred_b),'pred_c_export':bool(pred_c),'pred_d_price':bool(pred_d)},
             'terminal':'exact_source_edit_executor_verified' if pred_a and pred_b and pred_c and pred_d else 'exact_source_edit_validation_failed',
             'scope':'native-background intervention executor, no new semantic parameter reduction','wall_seconds':time.perf_counter()-started}
    OUT.write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({k:receipt[k] for k in ('terminal','predictions','price','wall_seconds')}),flush=True)


def main():
    sys.path[:0]=[str(ROOT),str(POLY)];assert bound_hash()==EXPECTED
    import torch
    import exact_source_edit_reference as R
    import source_support_interaction_reference as D
    import join_contribution_context_reference as J
    torch.set_num_threads(2);checks=R.controls();assert checks['passed'],checks
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps({'dryrun':True,'controls':checks,'checkpoint_opened':False}));return
    run(torch,R,D,J,checks)


if __name__=='__main__':main()
