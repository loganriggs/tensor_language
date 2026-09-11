#!/usr/bin/env python3
# BQGATE: 0forwards0seq; full-U weights only, two matched-size group families.
"""pred_a numeric, pred_b objective gain, pred_c capture, pred_d cost.
Two starts each of64 shared-input rank8 / output-sharing LL1 rank16 groups.
120softseconds per fit. No corpus access, convergence or circuit claim.
"""
import os,sys,json,time,signal,hashlib
from pathlib import Path
HERE=Path(__file__).resolve();ROOT=HERE.parents[3];P=ROOT/'basis_aligned/polynomial_causal'
sys.path[:0]=[str(P),str(ROOT)]
import torch
import numpy as np
from native_reader_msp_generalization_v1 import CK
from shared_reader_conditional_v1 import partner_update
import shared_reader_group_objective_v1 as shared
import symmetric_ll1_objective_v1 as ll1
from shared_reader_group_fit_v1 import Objective as SharedObjective,fit
from joint_quadratic_fit_v1 import product_cross
from structured_branch_amplitudes_v1 import inner
PREFIX='MATCHED_SHARED_GROUPS_V1';GROUPS=64;PENALTY=.01


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2);f.write('\n')


def main():
    binding=json.loads((P/f'{PREFIX}_BINDING.json').read_text())
    assert all(digest(path)==sha for path,sha in binding.items())
    assert all(json.loads((P/'MATCHED_GROUP_OBJECTIVES_V1_CONTROL.json').read_text())['predictions'].values())
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps(dict(model_loaded=False,gpu_accessed=False,body_forwards=0,corpus_access=False,
                              groups=64,shared_partner_rank=8,ll1_input_rank=16,fit_seconds_per_arm=120,
                              shared_floats=1253376,ll1_floats=1254400,starts=['spectral','native'])));return
    out=P/f'{PREFIX}_RESULT.json';assert not out.exists();signal.alarm(2400)
    torch.set_default_dtype(torch.float64);torch.set_num_threads(2);torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32=False;started=time.perf_counter()
    sd=torch.load(CK,weights_only=True,map_location='cpu',mmap=True)
    l,r,d=[sd[f'transformer.h.17.mlp.{key}.weight'].double().cuda() for key in ('Left','Right','Down')]
    metric=torch.load('/dev/shm/bilin18_native_product_energy_v1.pt',weights_only=True,map_location='cpu')['unembedding_gram'].double().cuda()
    wh=torch.linalg.cholesky(metric).T;target=(l,r,wh@d)
    total=json.loads((P/'NATIVE_READER_METRIC_V1_AUDIT.json').read_text())['native_total']
    assert l.shape==r.shape==(4608,1152) and d.shape==(1152,4608)
    input_cache=torch.load('/dev/shm/bilin18_shared_input_subspace_native_v1.pt',weights_only=True,map_location='cpu')
    input_spectral=input_cache['full']['eigenvectors'][:,-64:].T.flip(0).cuda()
    output_cache=torch.load('/dev/shm/bilin18_output_function_agreement_v1.pt',weights_only=True,map_location='cpu')
    output_values,output_vectors=torch.linalg.eigh(output_cache['covariances']['native'].cuda())
    output_spectral=output_vectors[:,-64:].T.flip(0)
    energy=target[2].square().sum(0)*.5*(l.square().sum(1)*r.square().sum(1)+(l*r).sum(1).square())
    ids=torch.multinomial(energy.cpu(),64,replacement=False,generator=torch.Generator().manual_seed(1711)).cuda()
    input_native=l[ids];input_native/=input_native.norm(dim=1,keepdim=True)
    output_native=target[2][:,ids].T;output_native/=output_native.norm(dim=1,keepdim=True)
    rows=[];functions=[]
    for family,module,cls,rank,initials in [
        ('shared',shared,SharedObjective,8,[input_spectral,input_native]),
        ('ll1',ll1,ll1.Objective,16,[output_spectral,output_native])]:
        for label,initial_vectors in zip(['spectral','native'],initials):
            begin=time.perf_counter();first=[];second=[]
            for j,vector in enumerate(initial_vectors):
                if family=='shared':
                    u,v=partner_update(vector,*target,rank);first.append(v.T);second.append(u)
                else:
                    weights=vector@target[2]
                    matrix=(l.T*weights)@r;matrix=(matrix+matrix.T)/2
                    values,vectors=torch.linalg.eigh(matrix)
                    keep=values.abs().argsort(descending=True)[:rank]
                    first.append(vectors[:,keep].T);second.append(values[keep])
                if j%8==0:print(json.dumps(dict(family=family,arm=label,initialized=j+1,seconds=time.perf_counter()-begin)),flush=True)
            if family=='shared':parts=(initial_vectors,torch.stack(first),torch.stack(second))
            else:parts=(torch.stack(first),torch.stack(second),initial_vectors)
            raw=module.cp(*parts)
            gram=((raw[2].T@raw[2])*product_cross(raw[0],raw[1],raw[0],raw[1])).reshape(64,rank,64,rank).sum((1,3))
            rhs=((target[2].T@raw[2])*product_cross(target[0],target[1],raw[0],raw[1])).sum(0).reshape(64,rank).sum(1)
            matrix=gram+PENALTY*torch.diag(gram.diag());amplitudes=torch.linalg.solve(matrix,rhs)
            normal=float((matrix@amplitudes-rhs).norm()/rhs.norm())
            if family=='shared':parts=(parts[0],parts[1],parts[2]*amplitudes[:,None,None])
            else:parts=(parts[0],parts[1]*amplitudes[:,None],parts[2])
            init_seconds=time.perf_counter()-begin
            objective=cls(target,parts,total,PENALTY)
            torch.cuda.synchronize();tic=time.perf_counter();initial,gradient=objective.evaluate(objective.initial);torch.cuda.synchronize();gradient_seconds=time.perf_counter()-tic
            direction=-gradient/max(np.linalg.norm(gradient),1e-30);h=1e-6;slope=float(gradient@direction)
            fd=abs((objective.evaluate(objective.initial+h*direction)[0]-objective.evaluate(objective.initial-h*direction)[0])/(2*h)-slope)/max(1.,abs(slope))
            preflight=dict(family=family,arm=label,initial_objective=initial,fd_error=fd,normal_error=normal,gradient_seconds=gradient_seconds)
            write(P/f'{PREFIX}_{family.upper()}_{label.upper()}_PREFLIGHT.json',preflight);print(json.dumps(preflight),flush=True)
            assert fd<=1e-6 and normal<=1e-8
            point,optimization=fit(objective,seconds=120,max_iterations=1000)
            final_parts=objective.physical(point);program=module.cp(*final_parts)
            replay=float((inner(program,program)-2*inner(target,program)+total)/total)
            x=torch.randn(8,1152,device='cuda',generator=torch.Generator(device='cuda').manual_seed(1712))
            direct=((x@program[0].T)*(x@program[1].T))@program[2].T
            if family=='shared':
                a,v,w=final_parts;grouped=torch.einsum('ng,gor,ngr->no',x@a.T,w,torch.einsum('nd,grd->ngr',x,v))
            else:
                a,s,c=final_parts;grouped=torch.einsum('ngr,gr,go->no',torch.einsum('nd,grd->ngr',x,a).square(),s,c)
            execution=float((direct-grouped).norm()/direct.norm());error=abs(replay-optimization['details']['residual'])
            cache=Path(f'/dev/shm/bilin18_matched_shared_groups_v1_{family}_{label}.pt');assert not cache.exists()
            torch.save(dict(parts=tuple(v.cpu() for v in final_parts),family=family,output_whitener=wh.cpu(),penalty=PENALTY,
                            bias=sd['transformer.h.17.mlp.Down_bias'],scope='Output coefficients are full-U isometric coordinates; solve whitener for physical residual writes.'),cache)
            row=dict(family=family,label=label,initialization_seconds=init_seconds,initial_gradient_seconds=gradient_seconds,
                     finite_difference_error=fd,amplitude_normal_error=normal,cp_replay=error,executor_replay=execution,
                     capture=1-replay,objective_gain=initial-optimization['final'],optimization=optimization,
                     cache=dict(path=str(cache),sha256=digest(cache)),matrix_floats=sum(v.numel() for v in final_parts))
            write(P/f'{PREFIX}_{family.upper()}_{label.upper()}.json',row);rows.append(row);functions.append(program)
    cosine={family:float(inner(functions[i],functions[i+1])/(inner(functions[i],functions[i])*inner(functions[i+1],functions[i+1])).sqrt()) for family,i in [('shared',0),('ll1',2)]}
    predictions={}
    for family in ['shared','ll1']:
        arm=[r for r in rows if r['family']==family]
        predictions[family]={'pred_a_instrument':all(max(r['cp_replay'],r['executor_replay'],r['amplitude_normal_error'])<=1e-8 and r['finite_difference_error']<=1e-6 and r['optimization']['maximum_increase']<=1e-10 for r in arm),
                             'pred_b_objective_gain':all(r['objective_gain']>=.001 for r in arm),
                             'pred_c_capture':all(r['capture']>=.1 for r in arm),
                             'pred_d_cost':all(r['initial_gradient_seconds']<=10 for r in arm)}
    write(out,dict(predictions=predictions,rows=rows,function_cosine=cosine,seconds=time.perf_counter()-started,
                   ll1_output_rank_capture_ceiling=float(output_values[-64:].sum()/output_values.sum()),
                   shared_input_capture_ceiling=min(1.,float(2*input_cache['full']['eigenvalues'][-64:].sum()/total)),
                   binding=binding,body_forwards=0,corpus_access=False,
                   scope='Nearly matched-parameter native initialization/cost pilots, not convergence/global/circuit evidence.'))
    print(json.dumps(dict(predictions=predictions,function_cosine=cosine)),flush=True)


if __name__=='__main__':main()
