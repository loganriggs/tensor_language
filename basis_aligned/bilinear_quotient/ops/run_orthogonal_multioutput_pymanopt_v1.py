#!/usr/bin/env python3
"""pred_a exact latest-checkpoint replay; pred_b convergence; pred_c5%capture gain.

BQGATE:0forwards0seq. Same full-U lambda.01 overlapping16x16x4 family,120s.
"""
import os,sys,time,json,hashlib,signal
from pathlib import Path
RUNNER=Path(__file__).resolve();P=RUNNER.parents[3]/'basis_aligned/polynomial_causal';sys.path.insert(0,str(P))
import torch
from orthogonal_multioutput_pymanopt_v2 import View,evaluate,fit,manifold_for
from multioutput_quadratic_blocks_v1 import MultioutputWeightObjective
from joint_quadratic_fit_v1 import product_cross
CK=Path('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin')


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    binding=json.loads((P/'ORTHOGONAL_MULTIOUTPUT_PYMANOPT_V1_BINDING.json').read_text())
    assert all(digest(f)==h for f,h in binding.items())
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps(dict(gpu_accessed=False,body_forwards=0,corpus_access=False,seconds=120,groups=16,rank=16,outputs=4,penalty=.01)));return
    out=P/'ORTHOGONAL_MULTIOUTPUT_PYMANOPT_V1_RESULT.json';assert not out.exists();signal.alarm(900)
    torch.set_num_threads(2);torch.set_default_dtype(torch.float64);torch.backends.cuda.matmul.allow_tf32=False;start=time.perf_counter()
    sd=torch.load(CK,weights_only=True,map_location='cpu',mmap=True);u=sd['lm_head.weight'].double().cuda();metric=u.T@u;del u
    l,r,d=[sd[f'transformer.h.17.mlp.{key}.weight'].double().cuda() for key in ['Left','Right','Down']]
    total=((d.T@metric@d)*product_cross(l,r,l,r)).sum()
    objective=MultioutputWeightObjective(metric,total,l=l,r=r,d=d,penalty=.01)
    source=torch.load(P/'MULTIOUTPUT_MANIFOLD_V1_CHECKPOINT.pt',weights_only=True,map_location='cpu')
    bank=source['model']['bank'].transpose(-1,-2).contiguous().double().cuda()
    packed=source['model']['core'].reshape(64,136).T.contiguous().double().cuda();packed/=packed.norm(dim=0)
    initial,_,initial_detail=evaluate(objective,bank,packed,4)
    old=json.loads((P/'MULTIOUTPUT_MANIFOLD_V1_RESULT.json').read_text())['final']
    errors=[abs(initial-old['optimization_loss']),abs(initial_detail['residual']-old['squared_relative_error'])]
    assert max(errors)<1e-8,errors
    print(json.dumps(dict(initial_loss=initial,initial_replay_errors=errors)),flush=True)
    point,report=fit(objective,bank,packed,4,seconds=120,tolerance=1e-7)
    bank,packed=[x.cuda() for x in point]
    loss,grad,details=evaluate(objective,bank,packed,4)
    # Same tangent scaling as the older custom manifold implementation.
    gb,gc=grad;inner=bank.transpose(-1,-2)@gb
    gb=gb-bank@((inner+inner.transpose(-1,-2))/2)
    gc=gc-packed*(gc*packed).sum(0,keepdim=True)
    capture=1-details['residual'];legacy=max(float(gb.norm())*16,float(gc.norm())*8)/max(capture,1e-12)
    maximum_gradient=max(float(gb.abs().max()),float(gc.abs().max()))
    history=report['scalar_history'];last_step=report['iterations']-1
    if not history or history[-1]['iteration']!=last_step:
        history.append(dict(iteration=last_step,cost=loss,gradient_norm=report['tangent_stationarity']))
    progress=abs(loss-history[-5]['cost'])/max(capture,1e-12) if len(history)>=5 else None
    accepted=report['converged'] and legacy<=1e-4 and maximum_gradient<=1e-7 and progress is not None and progress<=1e-5
    view=View(bank,packed,4);_,_,_,writer,gram,reg=objective.terms(view);cross,_=objective.cross_gram(view)
    solve_error=float((writer@reg-cross).norm()/cross.norm())
    orth=float((bank.transpose(-1,-2)@bank-torch.eye(16,device='cuda')).abs().max())
    errors.extend([abs(loss-report['loss']),solve_error,orth,float((packed.norm(dim=0)-1).abs().max()),report['maximum_objective_increase']])
    valid=max(errors)<1e-8
    cache=Path('/dev/shm/bilin18_orthogonal_multioutput_pymanopt_v1.pt');assert not cache.exists()
    torch.save(dict(bank=bank.cpu(),packed=packed.cpu(),writer=writer.detach().cpu(),binding=binding,report=report),cache)
    result=dict(predictions={'pred_a_instrument':valid,'pred_b_converged':valid and accepted,
        'pred_c_capture_gain':valid and capture>=1.05*.08623021231576344},initial_loss=initial,
        initial_capture=1-initial_detail['residual'],final=report,captured_energy=capture,
        original_relative_stationarity=legacy,original_maximum_gradient=maximum_gradient,
        original_five_check_relative_progress=progress,all_convergence_criteria_held=accepted,
        maximum_instrument_error=max(errors),writer_condition=float(torch.linalg.cond(reg)),
        cache=dict(path=str(cache),sha256=digest(cache),bytes=cache.stat().st_size,ephemeral=True),
        wall_seconds=time.perf_counter()-start,inherited_fit_seconds=540.148+240.027,
        body_forwards=0,corpus_access=False,binding=binding,
        scope='Standard-library CG continuation of an already manifold-optimized checkpoint. Same full-U lambda.01 objective and overlapping block family, all original convergence conditions plus stricter absolute norm. No global/circuit claim.')
    with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['final','binding']},indent=2),flush=True);assert valid

if __name__=='__main__':main()
