#!/usr/bin/env python3
# BQGATE:0bodyforwards;0texttokens;10starts;3refinements;600seconds.
"""pred_a GPU objective replay <=1e-8; pred_b >=2 promoted starts intrinsicgrad<=1e-6.
pred_c best squared coefficient error <=.90 best initial error at same rank32 blocks.
Null: output mixtures alone give little gain or remain locally unconverged.
No native behavior claim; mixed tensor only, twelve individual token outputs.
"""
import os,sys,json,time,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal';sys.path.insert(0,str(P))
import torch
import numpy as np
from scipy.optimize import minimize
from head17_output_block_objective_v1 import build,Objective
from sparse_path_stability_atlas_v1 import digest
STEM='HEAD17_OUTPUT_BLOCK_FIT_V1'


def main():
    binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files']
    assert all(digest(f)==h for f,h in binding.items())
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print('0forwards0tokens;10starts15iterations;top3up-to240iterations;600seconds');return
    out=P/(STEM+'_RESULT.json');artifact=P/(STEM+'_PROGRAM.pt')
    assert not out.exists() and not artifact.exists()
    signal.alarm(600);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    tic=time.perf_counter();T,ids=build('cuda');obj=Objective(T,32)
    torch.manual_seed(358);theta=(torch.randn(66,dtype=torch.float64)*.1).numpy()
    I=torch.eye(12,dtype=torch.float64,device='cuda')
    control=json.loads((P/'HEAD17_OUTPUT_BLOCK_OBJECTIVE_V1_CONTROL.json').read_text())
    replay=abs(obj.value_gradient(theta,I)[0]-control['normalized_squared_error'])
    assert replay<=1e-8
    output_eigenvectors=torch.linalg.eigh(obj.flat@obj.flat.T)[1].flip(1)
    bases=[I,output_eigenvectors]
    for seed in range(8):
        g=torch.Generator().manual_seed(360+seed)
        bases.append(torch.linalg.qr(torch.randn(12,12,generator=g,dtype=torch.float64))[0].cuda())
    initial=[obj.evaluate(q)[0] for q in bases];records=[];states=[]

    class TimeLimit(Exception):pass

    def fit(base,iterations):
        best={'value':obj.evaluate(base)[0],'Q':base.detach().clone()};history=[]
        def fun(x):
            if time.perf_counter()-tic>550:raise TimeLimit()
            value,gradient,intrinsic=obj.value_gradient(x,base)
            history.append(value)
            if value<best['value']:
                best.update(value=value,Q=obj.rotate(torch.tensor(x,device='cuda',dtype=torch.float64),base).detach())
            return value,gradient
        try:
            result=minimize(fun,np.zeros(66),jac=True,method='L-BFGS-B',
                            options=dict(maxiter=iterations,maxls=20,gtol=1e-8,ftol=1e-12))
            status=str(result.message);nit=int(result.nit)
        except TimeLimit:status='run time budget reached';nit=None
        loss,_,gn,_=obj.evaluate(best['Q'])
        return best['Q'],dict(loss=loss,intrinsic_gradient=gn,converged=gn<=1e-6,
                             evaluations=len(history),iterations=nit,optimizer_status=status,
                             last_values=history[-5:])

    for i,q in enumerate(bases):
        Q,record=fit(q,15);record.update(start=i,initial=initial[i],phase='screen')
        states.append(Q);records.append(record);print(json.dumps(record),flush=True)
    chosen=sorted(range(10),key=lambda i:records[i]['loss'])[:3];refined=[]
    for i in chosen:
        Q=states[i];cycles=[]
        for _ in range(3):
            Q,record=fit(Q,80);cycles.append(record)
            if record['converged'] or time.perf_counter()-tic>550:break
        states[i]=Q;refined.append(dict(start=i,cycles=cycles,**cycles[-1]))
        print(json.dumps(refined[-1]),flush=True)
    best_index=min(range(10),key=lambda i:obj.evaluate(states[i])[0]);Q=states[best_index]
    loss,_,gn,blocks=obj.evaluate(Q)
    mixed=(Q.T@obj.flat).reshape_as(T);u,s,vh=torch.linalg.svd(mixed,full_matrices=False)
    left=u[:,:,:32]*s[:,:32,None].transpose(1,2)*obj.scale;right=vh[:,:32,:]
    rebuilt=(Q@(left@right).flatten(1)).reshape_as(T)
    rebuild_error=abs(float((rebuilt-T).square().sum()/T.square().sum())-loss)
    assert rebuild_error<=1e-10
    torch.save(dict(output_basis=Q.cpu(),left=left.cpu(),right=right.cpu(),token_ids=torch.tensor(ids)),artifact)
    reconstructions=[(states[i]@obj.evaluate(states[i])[3].flatten(1)).flatten() for i in chosen]
    cosines=[float(torch.dot(reconstructions[i],reconstructions[j])/(reconstructions[i].norm()*reconstructions[j].norm())) for i in range(3) for j in range(i)]
    result=dict(pred_a=replay<=1e-8,pred_b=sum(x['converged'] for x in refined)>=2,
                pred_c=loss<=.9*min(initial),initial_losses=initial,screen=records,refined=refined,
                best_start=best_index,best_loss=loss,best_relative_error=loss**.5,best_intrinsic_gradient=gn,
                relative_loss_improvement=1-loss/min(initial),promoted_function_cosines=cosines,
                gpu_replay_error=replay,reconstruction_loss_error=rebuild_error,
                tensor_shape=list(T.shape),rank_per_block=32,stored_mixed_scalars=12*32*(1152+128)+144,
                seconds=time.perf_counter()-tic,artifact_sha=digest(artifact),
                scope='Ten initial output bases; top three refined with chart recentering. '
                'Orthogonal twelve-output LL1 family with fixed matrix rank32; all block readers '
                'refit by exact SVD. No global recovery guarantee or full normalized/native behavior claim.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
