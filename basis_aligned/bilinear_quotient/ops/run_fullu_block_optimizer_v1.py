#!/usr/bin/env python3
# BQGATE: 0 body forwards,0 text sequences; two exact weight-only optimizations,1300second alarm.
"""pred_a replay/descent/orthogonality; pred_b balanced cut<=.1 and20%controlgain;
pred_c cross-seed overlap>=.9; pred_d both gradient norms<=1e-6.
Known planted local minima; no global or circuit claim.
"""
import os,sys,time,json,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal';sys.path.insert(0,str(P))
import torch
from sparse_path_stability_atlas_v1 import digest
from quadratic_producer_projection_v1 import atom_gram
from fullu_block_optimizer_v1 import optimize
STEM='FULLU_BLOCK_OPTIMIZER_V1'

@torch.no_grad()
def main():
    binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files']
    assert all(digest(k)==v for k,v in binding.items())
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps(dict(body_forwards=0,text_sequences=0,seeds=2,max_steps=1500,seconds_per_seed=600)));return
    out=P/(STEM+'_RESULT.json');artifact=P/(STEM+'_READERS.pt');assert not out.exists() and not artifact.exists()
    signal.alarm(1300);start=time.perf_counter();torch.set_num_threads(2);torch.set_default_dtype(torch.float64);torch.backends.cuda.matmul.allow_tf32=False
    state=torch.load(next(k for k in binding if k.endswith('/pytorch_model.bin')),weights_only=True,mmap=True)
    u=state['lm_head.weight'].double().cuda();u-=u.mean(0)
    l0,r0,d0,l1,r1,d1=[state[f'transformer.h.{layer}.mlp.{name}.weight'].double().cuda() for layer in (16,17) for name in ('Left','Right','Down')]
    scale=float(state['transformer.h.17.lambdas'][0]);h=scale**2*d0@atom_gram(l0,r0)@d0.T
    he,hv=torch.linalg.eigh(h);hs=(hv*he.sqrt())@hv.T;l,r=l1@hs,r1@hs
    g=d1.T@(u.T@u)@d1
    frozen=torch.load(P/'FULLU_INPUT_BLOCKS_V1_PROJECTORS.pt',weights_only=True)
    previous=json.loads((P/'FULLU_INPUT_BLOCKS_V1_RESULT.json').read_text())['reports']
    reports=[];readers=[];errors=[]
    for seed in (120423,120424):
        _,v=torch.linalg.eigh(frozen[f'producer_{seed}'].cuda());q=v[:,-576:]
        def checkpoint(row,current):
            torch.save(dict(seed=seed,readers=current.cpu(),row=row),P/(STEM+'_PROGRESS.pt'))
            print(json.dumps(dict(seed=seed,**row)),flush=True)
        q,report=optimize(l,r,g,q,callback=checkpoint)
        original=next(x for x in previous if x['metric']=='producer' and x['seed']==seed and x['kind']=='candidate')
        control=next(x for x in previous if x['metric']=='producer' and x['seed']==seed and x['kind']=='random_basis')
        errors.append(abs(report['initial']-original['normalized_cut']))
        report.update(seed=seed,matched_control=control['normalized_cut'])
        reports.append(report);readers.append(q.cpu())
    overlap=float((readers[0].T@readers[1]).square().sum()/576);overlap=max(overlap,1-overlap)
    a=max(errors)<1e-8 and all(x['orthogonality_error']<1e-8 and all(b['objective']<=a['objective']+1e-12 for a,b in zip(x['history'],x['history'][1:])) for x in reports)
    a=a and all(torch.isfinite(torch.tensor([x['normalized_cut'],x['gradient_norm'],x['incident_fraction']])).all().item() for x in reports)
    b=all(x['normalized_cut']<=.1 and .1<=x['incident_fraction']<=.9 and x['normalized_cut']<=.8*x['matched_control'] for x in reports)
    result={'pred_a':a,'pred_b':b,'pred_c':overlap>=.9,'pred_d':all(x['converged'] for x in reports)}
    torch.save(dict(readers=readers,seeds=[120423,120424]),artifact)
    result.update(reports=reports,overlap=overlap,replay_errors=errors,execution_seconds=time.perf_counter()-start,artifact_sha=digest(artifact),source_shas=binding)
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('reports','source_shas')}),flush=True)

if __name__=='__main__':main()
