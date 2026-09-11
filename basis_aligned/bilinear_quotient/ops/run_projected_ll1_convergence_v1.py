#!/usr/bin/env python3
# BQGATE: 0forwards0seq; full-U weight-only projected LL1 optimization, two starts.
"""pred_a numeric; pred_b local convergence; pred_c function/group stability.
Two64x16LL1starts, exact output solve,1200softseconds/start,max10000iterations.
No corpus, model-body forwards, or semantic circuit claim.
"""
import os,sys,json,time,signal,hashlib
from pathlib import Path
HERE=Path(__file__).resolve();ROOT=HERE.parents[3];P=ROOT/'basis_aligned/polynomial_causal'
sys.path[:0]=[str(P),str(ROOT)]
import numpy as np
import torch
import scipy
from scipy.optimize import linear_sum_assignment
from equilibrated_ll1_projected_v1 import Objective
from projected_ll1_fit_v1 import fit
from symmetric_ll1_objective_v1 import cp
from structured_branch_amplitudes_v1 import inner
from ll1_group_matching_v1 import group_inner
PREFIX='PROJECTED_LL1_CONVERGENCE_V1'
CK=Path('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin')


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def write(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2);f.write('\n')


def main():
    binding=json.loads((P/f'{PREFIX}_BINDING.json').read_text())
    assert all(digest(path)==sha for path,sha in binding.items())
    control=json.loads((P/'EQUILIBRATED_LL1_PROJECTED_V1_CONTROL.json').read_text())
    assert all(control[k] for k in ('pred_a','pred_b','pred_c'))
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps(dict(model_loaded=False,gpu_accessed=False,body_forwards=0,corpus_access=False,
                              groups=64,input_rank=16,starts=['spectral','native'],fit_seconds_per_start=1200,
                              penalty=.01,parameter_floats=1254400,local_gradient_bar=1e-7,progress_bar=1e-6)));return
    out=P/f'{PREFIX}_RESULT.json';assert not out.exists();signal.alarm(3000)
    started=time.perf_counter();torch.set_default_dtype(torch.float64);torch.set_num_threads(2);torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32=False
    sd=torch.load(CK,weights_only=True,map_location='cpu',mmap=True)
    l,r,d=[sd[f'transformer.h.17.mlp.{key}.weight'].double().cuda() for key in ('Left','Right','Down')]
    assert l.shape==r.shape==(4608,1152) and d.shape==(1152,4608)
    total=json.loads((P/'NATIVE_READER_METRIC_V1_AUDIT.json').read_text())['native_total']
    prior=json.loads((P/'MATCHED_SHARED_GROUPS_V1_RESULT.json').read_text())
    rows=[];programs=[]
    for label in ('spectral','native'):
        source=Path(f'/dev/shm/bilin18_matched_shared_groups_v1_ll1_{label}.pt')
        saved=torch.load(source,weights_only=True,map_location='cpu')
        a,s,c=[x.double().cuda() for x in saved['parts']];wh=saved['output_whitener'].double().cuda()
        target=(l,r,wh@d);objective=Objective(target,a,s,total,.01)
        initial,gradient=objective.evaluate(objective.initial)
        original=next(row for row in prior['rows'] if row['family']=='ll1' and row['label']==label)['optimization']['final']
        assert initial<=original+1e-8
        direction=np.random.default_rng(2402).normal(size=len(gradient));direction/=np.linalg.norm(direction);h=1e-6
        finite=(objective.evaluate(objective.initial+h*direction)[0]-objective.evaluate(objective.initial-h*direction)[0])/(2*h)
        fd=abs(finite-gradient@direction)/max(1.,abs(finite),abs(gradient@direction))
        assert fd<=1e-6
        write(P/f'{PREFIX}_{label.upper()}_PREFLIGHT.json',dict(initial=initial,previous_joint_objective=original,finite_difference_error=float(fd)))
        point,stats=fit(objective,seconds=1200,max_iterations=10000)
        aa,ss=objective.physical(point);cc=objective.writer;parts=(aa,ss,cc)
        torch.manual_seed(2403);x=torch.randn(13,1152,device='cuda')
        scalar=(torch.einsum('nd,gkd->ngk',x,aa).square()*ss[None]).sum(-1)
        bank=cp(*parts);reference=((x@bank[0].T)*(x@bank[1].T))@bank[2].T
        replay=float((scalar@cc-reference).norm()/reference.norm())
        details=stats['details']
        assert details['output_system_condition']<=details['output_condition_bound']*(1+1e-6)
        numeric=max(float(fd),replay,details['reduced_identity_error'],details['output_normal_residual'])<=1e-6 and max(replay,details['reduced_identity_error'],details['output_normal_residual'])<=1e-8 and stats['maximum_increase']<=1e-10
        artifact=P/f'{PREFIX}_{label.upper()}.pt'
        assert not artifact.exists()
        torch.save(dict(parts=tuple(v.detach().cpu() for v in parts),output_whitener=wh.cpu(),penalty=.01,
                        bias=sd['transformer.h.17.mlp.Down_bias'].cpu(),source_sha256=digest(source)),artifact)
        row=dict(label=label,pred_a=bool(numeric),pred_b=stats['local_converged'],capture=1-details['residual'],
                 finite_difference_error=float(fd),executor_replay=replay,optimization=stats,
                 artifact=dict(path=str(artifact),sha256=digest(artifact)))
        write(P/f'{PREFIX}_{label.upper()}.json',row);rows.append(row);programs.append(parts)
        print(json.dumps({k:v for k,v in row.items() if k!='optimization'}),flush=True)
    first,second=[cp(*parts) for parts in programs]
    function_cos=float(inner(first,second)/(inner(first,first)*inner(second,second)).sqrt())
    cross=group_inner(*programs);g0=group_inner(programs[0],programs[0]);g1=group_inner(programs[1],programs[1])
    sims=cross/(g0.diag()[:,None]*g1.diag()[None,:]).sqrt();i,j=linear_sum_assignment(-sims.cpu().numpy());matched=sims[i,j]
    result=dict(pred_a=all(row['pred_a'] for row in rows),pred_b=all(row['pred_b'] for row in rows),
                pred_c=function_cos>=.95 and int((matched>=.8).sum())>=16,rows=rows,function_cosine=function_cos,
                groups_matched_ge_08=int((matched>=.8).sum()),matching=[dict(spectral=int(a),native=int(b),cosine=float(sims[a,b])) for a,b in zip(i,j)],
                versions=dict(torch=torch.__version__,scipy=scipy.__version__),seconds=time.perf_counter()-started,
                scope='Weight-only projected LL1 fitting at fixed family/budget. Explicit local convergence and group stability are separate; no global optimum, OOD, extraction, selective manipulation or circuit adoption claim.')
    write(out,result)


if __name__=='__main__':main()
