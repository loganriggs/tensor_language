#!/usr/bin/env python3
# BQGATE: 0forwards0seq; native full-U weight-only kernel timing and numeric checks.
"""pred_a numeric; pred_b runtime/memory; pred_c matched CPU objective replay.

Two supplied 64-group rank16 graphs; no optimization or corpus access.
A relative directional FD<=1e-4, graph replay<=1e-8, inner normal<=1e-10.
B median uncached evaluation<=10s and peak CUDA allocation<=8GiB.
C initial normalized penalized objective agrees CPU conditional solve<=1e-8.
Null: kernel invalid or impractical; no inference of absent native structure.
Literal graph prices remain 1239793/1236439 floats plus90/119int64 indices,
with required common unembedding/whitener/bias/native background counted apart.
"""
import os,sys,json,time,signal,hashlib,statistics
from pathlib import Path
HERE=Path(__file__).resolve();ROOT=HERE.parents[3];P=ROOT/'basis_aligned/polynomial_causal'
sys.path[:0]=[str(P),str(ROOT)]
import numpy as np
import torch
from shared_reader_variable_projection_v2 import Objective
from ll1_joint_parent_graph_v2 import execute,factors
from symmetric_ll1_objective_v1 import cp
PREFIX='SHARED_READER_PROJECTED_PREFLIGHT_V1'
CK=Path('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin')


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    binding=json.loads((P/f'{PREFIX}_BINDING.json').read_text())
    assert all(digest(path)==sha for path,sha in binding.items())
    control=json.loads((P/'SHARED_READER_VARIABLE_PROJECTION_V2_CONTROL.json').read_text())
    assert control['held']
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps(dict(model_loaded=False,gpu_accessed=False,body_forwards=0,corpus_access=False,
                              groups=64,input_rank=16,starts=['spectral','native'],optimizer_steps=0,
                              penalty=.01,inner_tolerance=1e-11,maximum_evaluations_per_start=5)));return
    out=P/f'{PREFIX}_RESULT.json';assert not out.exists();signal.alarm(600)
    torch.set_default_dtype(torch.float64);torch.set_num_threads(2);torch.set_grad_enabled(True)
    torch.backends.cuda.matmul.allow_tf32=False
    sd=torch.load(CK,weights_only=True,mmap=True,map_location='cpu')
    l,r,d=[sd[f'transformer.h.17.mlp.{key}.weight'].double().cuda() for key in ('Left','Right','Down')]
    graphs=torch.load(P/'LL1_JOINT_CORE_SOLVE_V1_GRAPHS.pt',weights_only=True,map_location='cpu')
    prior=json.loads((P/'LL1_JOINT_CORE_SOLVE_V1_AUDIT.json').read_text())
    total=99245061353.47293;rows=[]
    for label in ('spectral','native'):
        graph=graphs[label];target=(l,r,graph['output_whitener'].double().cuda()@d)
        objective=Objective(target,graph,total,penalty=.01)
        torch.cuda.reset_peak_memory_stats();timings=[]
        def measured(point):
            torch.cuda.synchronize();start=time.perf_counter();value,gradient=objective.evaluate(point)
            torch.cuda.synchronize();timings.append(time.perf_counter()-start)
            return value,gradient
        initial,_=measured(objective.initial)
        expected=next(row for row in prior['rows'] if row['label']==label)['arms']['graph']['after_objective']
        direction=np.random.default_rng(3001).normal(size=len(objective.initial));direction/=np.linalg.norm(direction)
        point=objective.initial+1e-3*direction;value,gradient=measured(point);h=1e-4
        finite=(measured(point+h*direction)[0]-measured(point-h*direction)[0])/(2*h)
        analytic=float(gradient@direction)
        fd=abs(finite-analytic)/max(abs(finite),abs(analytic),1e-8)
        measured(point);fitted=objective.physical(point)
        with torch.no_grad():
            bank=cp(*factors(fitted));torch.manual_seed(3002);x=torch.randn(13,1152,device='cuda')
            reference=((x@bank[0].T)*(x@bank[1].T))@bank[2].T
            replay=float((execute(fitted,x)-reference).norm()/reference.norm())
        row=dict(label=label,initial_objective=initial,cpu_objective=expected,cpu_replay=abs(initial-expected),
                 displaced_objective=value,finite_derivative=float(finite),analytic_derivative=analytic,
                 relative_finite_difference_error=float(fd),executor_replay=replay,
                 inner_normal_residual=objective.last['inner']['normal_residual'],
                 inner_iterations=objective.last['inner']['iterations'],uncached_evaluation_seconds=timings,
                 median_evaluation_seconds=statistics.median(timings),
                 peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3)
        rows.append(row);print(json.dumps(row),flush=True)
        del fitted,objective,target
    predictions={'pred_a':all(max(row['relative_finite_difference_error']/1e-4,row['executor_replay']/1e-8,row['inner_normal_residual']/1e-10)<=1 for row in rows),
                 'pred_b':all(row['median_evaluation_seconds']<=10 and row['peak_allocated_gib']<=8 for row in rows),
                 'pred_c':all(row['cpu_replay']<=1e-8 for row in rows)}
    with out.open('x') as f:json.dump(dict(**predictions,rows=rows,scope='Native weight-only numeric/runtime preflight; zero optimizer steps and zero model-body forwards.'),f,indent=2);f.write('\n')


if __name__=='__main__':main()
