#!/usr/bin/env python3
"""pred_a exact gradients; pred_b bothconverged; pred_c both5%capturegain.

BQGATE:0forwards0seq. Full-U4608product structuredfit,276480coefficients.
V2: reuse savedseed0; correctedRichardsonpreflight. Same two-start experiment.
"""
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time
RUNNER=Path(__file__).resolve();P=RUNNER.parents[3]/'basis_aligned/polynomial_causal';sys.path.insert(0,str(P))
import torch
from structured_bilinear_bank_v1 import StructuredBank
from structured_branch_amplitudes_v1 import calibrate
from chunked_bilinear_coefficient_v1 import value_gradient
CK=Path('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin')


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def diagnostics(model,loss,details):
    capture=1-details['residual']
    relative=float(torch.stack([p.grad.norm()*p.detach().norm().clamp_min(1) for p in model.parameters()]).max())/max(capture,1e-12)
    maximum=float(torch.stack([p.grad.abs().max() for p in model.parameters()]).max())
    return dict(loss=float(loss),capture=capture,relative_stationarity=relative,
                gradient_max=maximum,cancellation=details['component_energy']/max(details['fitted_energy'],1e-30),**details)


def main():
    binding=json.loads((P/'STRUCTURED_BILINEAR_NATIVE_V2_BINDING.json').read_text())
    assert all(digest(f)==h for f,h in binding.items())
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps(dict(gpu_accessed=False,body_forwards=0,corpus_access=False,
            coefficients=276480,products=4608,seeds=[0,937],initialization_seconds=60,joint_seconds=300)));return
    out=P/'STRUCTURED_BILINEAR_NATIVE_V2_RESULT.json';assert not out.exists()
    signal.alarm(1500);torch.set_default_dtype(torch.float64);torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
    sd=torch.load(CK,map_location='cpu',weights_only=True,mmap=True)
    u=sd['lm_head.weight'].double().cuda();whitener=torch.linalg.cholesky(u.T@u).T;del u
    l,r,d=[sd[f'transformer.h.17.mlp.{key}.weight'].double().cuda() for key in ('Left','Right','Down')]
    native=(l,r,whitener@d)
    total=json.loads((P/'FULLU_OUTPUT_FUNCTIONS_V1_AUDIT.json').read_text())['native_total']
    reports=[]
    for seed in (0,937):
        torch.manual_seed(seed)
        if seed==0:
            prior=json.loads((P/'STRUCTURED_BILINEAR_NATIVE_V1_PREFLIGHT_0.json').read_text())
            saved=torch.load(prior['initial_cache']['path'],map_location='cpu',weights_only=True)
            model=StructuredBank([2]*7+[3,3],branches=4).cuda();model.load_state_dict(saved['model'])
            order=saved['order'];initialization=saved['initialization'];amplitudes=saved['preflight']['amplitude_solve']
            assert sum(p.numel() for p in model.parameters())==276480
        else:
            order=torch.arange(4608) if seed==0 else torch.randperm(4608,generator=torch.Generator().manual_seed(seed))
            targets=[[l[ids],r[ids],d[:,ids]] for ids in order.reshape(4,1152)]
            model=StructuredBank([2]*7+[3,3],branches=4).cuda();model.initialize(targets)
            assert sum(p.numel() for p in model.parameters())==276480
            init_optimizer=torch.optim.Adam(model.parameters(),lr=.03)
            initial_matrix_loss=float(model.matrix_loss(targets).detach());warm_start=time.perf_counter()
            for step in range(300):
                init_optimizer.zero_grad(set_to_none=True);matrix_loss=model.matrix_loss(targets)
                matrix_loss.backward();init_optimizer.step()
                if (step+1)%25==0:print(json.dumps(dict(seed=seed,phase='factor_initialization',step=step+1,loss=float(matrix_loss.detach()),seconds=time.perf_counter()-warm_start)),flush=True)
                if time.perf_counter()-warm_start>=60:break
            initialization=dict(initial_loss=initial_matrix_loss,final_loss=float(model.matrix_loss(targets).detach()),
                                steps=step+1,seconds=time.perf_counter()-warm_start)
            del init_optimizer,targets
            amplitudes=calibrate(model,native,whitener,penalty=.01,chunk=256)
        with torch.no_grad():
            a,b,w=model.factors();white=whitener@w
            f1,g1,_=value_gradient(*native,a,b,white,total,.01,256)
            f2,g2,_=value_gradient(*native,a,b,white,total,.01,512)
            chunk_error=max(abs(float(f1-f2))/max(abs(float(f1)),1e-30),
                            *(float((x-y).norm()/y.norm().clamp_min(1e-30)) for x,y in zip(g1,g2)))
        del a,b,w,white,g1,g2
        model.zero_grad(set_to_none=True)
        base,details=model.coefficient_loss(native,whitener,total)
        params=list(model.parameters());original=[p.detach().clone() for p in params]
        grad_norm=sum(p.grad.square().sum() for p in params).sqrt()
        param_norm=sum(p.square().sum() for p in params).sqrt().detach().clamp_min(1)
        assert float(grad_norm)>1e-16
        direction=[p.grad.detach().clone()/grad_norm*param_norm for p in params]
        expected=float(grad_norm*param_norm);derivatives=[]
        for eps in (1e-5,5e-6):
            with torch.no_grad():
                for p,x,v in zip(params,original,direction):p.copy_(x+eps*v)
            plus=float(model.coefficient_loss(native,whitener,total,backward=False)[0])
            with torch.no_grad():
                for p,x,v in zip(params,original,direction):p.copy_(x-eps*v)
            minus=float(model.coefficient_loss(native,whitener,total,backward=False)[0])
            derivatives.append(dict(eps=eps,derivative=(plus-minus)/(2*eps)))
        with torch.no_grad():
            for p,x in zip(params,original):p.copy_(x)
        finite=(4*derivatives[1]['derivative']-derivatives[0]['derivative'])/3
        fd_error=abs(finite-expected)/max(abs(expected),1e-30)
        del original,direction
        preflight=dict(chunk_relative_error=chunk_error,parameter_fd_relative_error=fd_error,
                       parameter_fd_expected=expected,parameter_fd_steps=derivatives,amplitude_solve=amplitudes)
        initial_cache=Path(f'/dev/shm/bilin18_structured_bilinear_v2_s{seed}_initial.pt');assert not initial_cache.exists()
        torch.save(dict(model={k:v.cpu() for k,v in model.state_dict().items()},seed=seed,
                        order=order,initialization=initialization,preflight=preflight,binding=binding),initial_cache)
        preflight['initial_cache']=dict(path=str(initial_cache),sha256=digest(initial_cache),bytes=initial_cache.stat().st_size)
        with (P/f'STRUCTURED_BILINEAR_NATIVE_V2_PREFLIGHT_{seed}.json').open('x') as f:json.dump(preflight,f,indent=2);f.write('\n')
        print(json.dumps(dict(seed=seed,preflight=preflight,initialization=initialization)),flush=True)
        assert chunk_error<=1e-6 and fd_error<=1e-5 and amplitudes['relative_solve_error']<=1e-8
        optimizer=torch.optim.LBFGS(model.parameters(),lr=1,max_iter=1,max_eval=10,
            history_size=20,tolerance_grad=0.,tolerance_change=0.,line_search_fn='strong_wolfe')
        evaluations=0
        def closure():
            nonlocal evaluations
            model.zero_grad(set_to_none=True)
            loss,detail=model.coefficient_loss(native,whitener,total)
            assert torch.isfinite(loss) and torch.stack([torch.isfinite(p.grad).all() for p in model.parameters()]).all()
            evaluations+=1
            closure.detail=detail
            return loss
        loss=closure();history=[dict(step=0,**diagnostics(model,loss,closure.detail))]
        fit_start=time.perf_counter();converged=False;stop='step_limit'
        for step in range(1,2001):
            optimizer.step(closure)
            if step%5==0 or time.perf_counter()-fit_start>=300 or step==2000:
                loss=closure();row=dict(step=step,seconds=time.perf_counter()-fit_start,
                                       **diagnostics(model,loss,closure.detail))
                history.append(row)
                progress=abs(row['loss']-history[-5]['loss'])/max(row['capture'],1e-12) if len(history)>=5 else None
                row['five_check_relative_change']=progress
                converged=row['relative_stationarity']<=1e-4 and row['gradient_max']<=1e-7 and progress is not None and progress<=1e-5
                print(json.dumps(dict(seed=seed,phase='joint_tensor',**row)),flush=True)
                if converged:stop='converged';break
                if time.perf_counter()-fit_start>=300:stop='time_limit';break
        cache=Path(f'/dev/shm/bilin18_structured_bilinear_v2_s{seed}.pt');assert not cache.exists()
        torch.save(dict(model={k:v.cpu() for k,v in model.state_dict().items()},
                        optimizer=optimizer.state_dict(),seed=seed,order=order,
                        history=history,binding=binding),cache)
        report=dict(seed=seed,initialization=initialization,preflight=preflight,
                    final=history[-1],history=history,converged=converged,stop=stop,
                    joint_seconds=time.perf_counter()-fit_start,evaluations=evaluations,
                    cache=dict(path=str(cache),sha256=digest(cache),bytes=cache.stat().st_size,ephemeral=True))
        reports.append(report)
        with (P/f'STRUCTURED_BILINEAR_NATIVE_V2_SEED_{seed}.json').open('x') as f:json.dump(report,f,indent=2);f.write('\n')
        print(json.dumps({k:v for k,v in report.items() if k!='history'},indent=2),flush=True)
        del model,optimizer,params
    result=dict(predictions={'pred_a_instrument':True,'pred_b_both_converged':all(r['converged'] for r in reports),
                            'pred_c_both_capture_gain':all(r['final']['capture']>=1.05*.08634383041327387 for r in reports)},
        starts=reports,wall_seconds=time.perf_counter()-started,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),binding=binding,
        body_forwards=0,corpus_access=False,
        scope='Joint structured full-rank weight fit; initialization/ordering and convergence limits remain. No circuit identification or data validation.')
    with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('starts','binding')},indent=2),flush=True)


if __name__=='__main__':main()
