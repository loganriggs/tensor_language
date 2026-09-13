#!/usr/bin/env python3
# BQGATE:0bodyforwards;0texttokens;2configurations;1sweep;900seconds.
"""Native weight-only timing and first-sweep screen, not convergence evidence.
pred_a folded trace/replay and GPU Gram controls <=1e-8.
pred_b both exact sweeps nonincreasing normalized loss (tolerance1e-8).
pred_c both complete initialization+sweep <=180seconds per configuration.
Null: expensive or weak first conditional fit; does not rule out converged structure.
Price U-only: shared/private banks + all codes + mean + int32 token group IDs.
"""
import os,sys,json,time,signal,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
P=ROOT/'basis_aligned/polynomial_causal'
sys.path.insert(0,str(P))
import torch
import shared_local_subspaces_v1 as solver
from shared_local_gram_bank_v1 import gram_bank
from check_shared_local_gram_bank_v1 import controls
from audit_fullu_output_functions_v1 import CK
from joint_quadratic_fit_v1 import product_cross
from fullu_shared_local_feasibility_v1 import price
STEM='FULLU_SHARED_LOCAL_TIMING_V1'


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files']
    assert all(digest(f)==sha for f,sha in binding.items())
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print('0forwards0tokens;full50304tokens weight rows;2configurations one exact sweep;900seconds');return
    out=P/(STEM+'_RESULT.json')
    assert not out.exists()
    signal.alarm(900)
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    started=time.perf_counter();checks=controls('cuda')
    sd=torch.load(CK,map_location='cpu',weights_only=True,mmap=True)
    u=sd['lm_head.weight'].to(device='cuda',dtype=torch.float64)
    l,r,down=[sd[f'transformer.h.17.mlp.{key}.weight'].to(device='cuda',dtype=torch.float64)
              for key in ['Left','Right','Down']]
    metric=down@product_cross(l,r,l,r)@down.T
    root=torch.linalg.cholesky((metric+metric.T)/2)
    mean=u.mean(0);x=(u-mean)@root
    energy=float(x.square().sum());mean_energy=float(len(u)*(mean@metric@mean))
    reference=json.loads((P/'FULLU_SHARED_LOCAL_FEASIBILITY_V1_RESULT.json').read_text())
    replay=abs((energy+mean_energy)/reference['coefficient_energy']-1)
    assert replay<=1e-8
    x=x/energy**.5
    del u,l,r,down,metric,sd
    solver.bank=gram_bank
    torch.cuda.synchronize();build_seconds=time.perf_counter()-started
    rows=[]
    for g,groups,local in [(64,32,8),(128,64,16)]:
        tic=time.perf_counter()
        state=solver.initialize(x,g,groups,local,9518)
        initial=solver.objective(x,state)
        torch.cuda.synchronize();initial_seconds=time.perf_counter()-tic
        tic=time.perf_counter();updated=solver.sweep(x,state)
        loss=solver.objective(x,updated)
        torch.cuda.synchronize();sweep_seconds=time.perf_counter()-tic
        assert loss<=initial+1e-8
        row=dict(global_width=g,groups=groups,local_width=local,
                 initial_centered_squared_error=initial,final_centered_squared_error=loss,
                 full_coefficient_relative_error=(loss*energy/(energy+mean_energy))**.5,
                 initial_seconds=initial_seconds,sweep_seconds=sweep_seconds,
                 group_counts=torch.bincount(updated['labels'],minlength=groups).tolist(),
                 **price(len(x),x.shape[1],g,groups,local))
        rows.append(row);print(json.dumps(row),flush=True)
        del state,updated
    result={'pred_a':replay<=1e-8,'pred_b':all(z['final_centered_squared_error']<=z['initial_centered_squared_error']+1e-8 for z in rows),
            'pred_c':all(z['initial_seconds']+z['sweep_seconds']<=180 for z in rows),
            'configurations':rows,'gpu_controls':checks,'native_trace_relative_error':replay,
            'build_seconds':build_seconds,'wall_seconds':time.perf_counter()-started,
            'peak_gpu_bytes':torch.cuda.max_memory_allocated(),'body_forwards':0,
            'scope':'One exact sweep, not convergence or behavioral evidence. No saved fit artifact.'}
    with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
