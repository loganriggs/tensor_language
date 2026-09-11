#!/usr/bin/env python3
# BQGATE: 0forwards0seq; weight-only full-U shared-reader group pilot.
"""pred_a numerical; pred_b objective gain; pred_c capture; pred_d cost.
64 rank8 groups, spectral/native-reader starts,120softseconds each, no text.
"""
import os,sys,json,time,signal,hashlib
from pathlib import Path
HERE=Path(__file__).resolve();ROOT=HERE.parents[3];P=ROOT/'basis_aligned/polynomial_causal'
sys.path[:0]=[str(P),str(ROOT)]
import torch
import numpy as np
from native_reader_msp_generalization_v1 import CK
from shared_reader_conditional_v1 import partner_update
from shared_reader_group_objective_v1 import cp
from shared_reader_group_fit_v1 import Objective,fit
from joint_quadratic_fit_v1 import product_cross
from structured_branch_amplitudes_v1 import inner
PREFIX='SHARED_READER_GROUP_PILOT_V1';GROUPS=64;RANK=8;PENALTY=.01


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2);f.write('\n')


def main():
    binding=json.loads((P/f'{PREFIX}_BINDING.json').read_text())
    assert all(digest(path)==sha for path,sha in binding.items())
    assert all(json.loads((P/'SHARED_READER_GROUP_OBJECTIVE_V1_CONTROL.json').read_text())['predictions'].values())
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps(dict(model_loaded=False,gpu_accessed=False,body_forwards=0,corpus_access=False,
                              groups=GROUPS,rank=RANK,fit_seconds=120,matrix_floats=GROUPS*1152*(1+2*RANK))));return
    out=P/f'{PREFIX}_RESULT.json';assert not out.exists();signal.alarm(1800)
    torch.set_default_dtype(torch.float64);torch.set_num_threads(2);torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32=False;start=time.perf_counter()
    sd=torch.load(CK,weights_only=True,map_location='cpu',mmap=True)
    l,r,d=[sd[f'transformer.h.17.mlp.{key}.weight'].double().cuda() for key in ('Left','Right','Down')]
    metric=torch.load('/dev/shm/bilin18_native_product_energy_v1.pt',weights_only=True,map_location='cpu')['unembedding_gram'].double().cuda()
    wh=torch.linalg.cholesky(metric).T;target=(l,r,wh@d)
    total=json.loads((P/'NATIVE_READER_METRIC_V1_AUDIT.json').read_text())['native_total']
    assert l.shape==r.shape==(4608,1152) and d.shape==(1152,4608)
    spectrum=torch.load('/dev/shm/bilin18_shared_input_subspace_native_v1.pt',map_location='cpu',weights_only=True)
    spectral=spectrum['full']['eigenvectors'][:,-GROUPS:].T.flip(0).cuda()
    energy=target[2].square().sum(0)*.5*(l.square().sum(1)*r.square().sum(1)+(l*r).sum(1).square())
    ids=torch.multinomial(energy.cpu(),GROUPS,replacement=False,generator=torch.Generator().manual_seed(1711))
    native=l[ids.cuda()];native/=native.norm(dim=1,keepdim=True)
    rows=[];functions=[]
    for label,a in [('spectral',spectral),('native',native)]:
        begin=time.perf_counter();us=[];vs=[]
        for j,reader in enumerate(a):
            u,v=partner_update(reader,*target,RANK);us.append(u);vs.append(v.T)
            if j%8==0:print(json.dumps(dict(arm=label,initialized_groups=j+1,seconds=time.perf_counter()-begin)),flush=True)
        w,v=torch.stack(us),torch.stack(vs);raw=cp(a,v,w)
        # Exact amplitudes combine overlapping initialized groups; eta penalizes whole-group energy.
        g=((raw[2].T@raw[2])*product_cross(raw[0],raw[1],raw[0],raw[1])).reshape(GROUPS,RANK,GROUPS,RANK).sum((1,3))
        rhs=((target[2].T@raw[2])*product_cross(target[0],target[1],raw[0],raw[1])).sum(0).reshape(GROUPS,RANK).sum(1)
        matrix=g+PENALTY*torch.diag(g.diag());amplitudes=torch.linalg.solve(matrix,rhs)
        normal=float((matrix@amplitudes-rhs).norm()/rhs.norm())
        w*=amplitudes[:,None,None]
        initialization_seconds=time.perf_counter()-begin
        objective=Objective(target,(a,v,w),total,PENALTY)
        torch.cuda.synchronize();tic=time.perf_counter();initial,gradient=objective.evaluate(objective.initial);torch.cuda.synchronize();gradient_seconds=time.perf_counter()-tic
        direction=-gradient/max(np.linalg.norm(gradient),1e-30);h=1e-6;slope=float(gradient@direction)
        finite=(objective.evaluate(objective.initial+h*direction)[0]-objective.evaluate(objective.initial-h*direction)[0])/(2*h)
        fd=abs(finite-slope)/max(1.,abs(slope))
        print(json.dumps(dict(arm=label,initial_objective=initial,fd=fd,gradient_seconds=gradient_seconds)),flush=True)
        assert fd<=1e-6 and normal<=1e-8
        point,optimization=fit(objective,seconds=120,max_iterations=1000)
        aa,vv,ww=objective.physical(point);program=cp(aa,vv,ww)
        replay=float((inner(program,program)-2*inner(target,program)+total)/total)
        x=torch.randn(8,1152,device='cuda',generator=torch.Generator(device='cuda').manual_seed(1712))
        direct=((x@program[0].T)*(x@program[1].T))@program[2].T
        grouped=torch.einsum('ng,gor,ngr->no',x@aa.T,ww,torch.einsum('nd,grd->ngr',x,vv))
        execution=float((direct-grouped).norm()/direct.norm())
        cache=Path(f'/dev/shm/bilin18_shared_reader_group_pilot_v1_{label}.pt');assert not cache.exists()
        torch.save(dict(a=aa.cpu(),v=vv.cpu(),w=ww.cpu(),output_whitener=wh.cpu(),penalty=PENALTY,
                        bias=sd['transformer.h.17.mlp.Down_bias'],scope='W is full-U isometric output, solve whitener for physical residual writes.'),cache)
        error=abs(replay-optimization['details']['residual'])
        row=dict(label=label,initialization_seconds=initialization_seconds,initial_gradient_seconds=gradient_seconds,
                 finite_difference_error=fd,amplitude_normal_error=normal,cp_replay=error,executor_replay=execution,
                 capture=1-replay,objective_gain=initial-optimization['final'],optimization=optimization,
                 cache=dict(path=str(cache),sha256=digest(cache)),matrix_floats=GROUPS*1152*(1+2*RANK))
        write(P/f'{PREFIX}_{label.upper()}.json',row);rows.append(row);functions.append(program)
    cosine=float(inner(*functions)/(inner(functions[0],functions[0])*inner(functions[1],functions[1])).sqrt())
    pred={'pred_a_instrument':all(max(row['cp_replay'],row['executor_replay'],row['amplitude_normal_error'])<=1e-8 and row['finite_difference_error']<=1e-6 and row['optimization']['maximum_increase']<=1e-10 for row in rows),
          'pred_b_objective_gain':all(row['objective_gain']>=.001 for row in rows),
          'pred_c_capture':all(row['capture']>=.1 for row in rows),
          'pred_d_cost':all(row['initial_gradient_seconds']<=10 for row in rows)}
    write(out,dict(predictions=pred,rows=rows,function_cosine=cosine,seconds=time.perf_counter()-start,
                   binding=binding,body_forwards=0,corpus_access=False,
                   scope='Native small-capacity initialization/cost pilot, not convergence/global/circuit evidence.'))
    print(json.dumps(dict(predictions=pred,function_cosine=cosine)),flush=True)


if __name__=='__main__':main()
