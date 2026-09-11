#!/usr/bin/env python3
"""pred_a exact replay/Hessian; pred_b both starts converge; pred_c5%capture gain.

BQGATE:0forwards0seq. 23x4full quadratic frames, seeds0/937,300fitseconds each.
Protocol FULL_QUADRATIC_FRAME_V1_PREREGISTRATION.md. 370944learned coefficients.
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
from full_quadratic_frame_v1 import cores_for
from full_quadratic_frame_v3 import fit
from conditional_block_svd_v1 import native_targets,full_cost
from orthogonal_multioutput_pymanopt_v2 import View,evaluate
from multioutput_quadratic_blocks_v1 import MultioutputWeightObjective
from block_variable_projection_curvature_v1 import hessian_vector
CK=Path('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin')


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def project(point,vector):return vector-point@(point.transpose(-1,-2)@vector)


def polar(point):
    values,vectors=torch.linalg.eigh(point.transpose(-1,-2)@point)
    return point@((vectors*values.rsqrt().unsqueeze(-2))@vectors.transpose(-1,-2))


def geometry_check(objective,bank,packed,seed):
    torch.manual_seed(seed+1751)
    direction=project(bank,torch.randn_like(bank));direction/=direction.norm()
    _,gradient,_=evaluate(objective,bank,packed,10)
    hv=hessian_vector(objective,bank,packed,10,[direction,torch.zeros_like(packed)])[0]
    expected=project(bank,hv)-direction@(bank.transpose(-1,-2)@gradient[0])
    eps=1e-4
    plus=polar(bank+eps*direction);minus=polar(bank-eps*direction)
    gp=evaluate(objective,plus,packed,10)[1][0]
    gm=evaluate(objective,minus,packed,10)[1][0]
    finite=project(bank,(project(plus,gp)-project(minus,gm))/(2*eps))
    return float((expected-finite).norm()/finite.norm().clamp_min(1e-30))


def independent(objective,bank,packed,writer):
    targets=native_targets(objective.l,objective.r,objective.d,bank)
    groups=bank.shape[0]
    w=writer.reshape(1152,groups,10).permute(1,0,2)
    c=packed.reshape(10,groups,10).permute(1,0,2)
    return [float(x) for x in full_cost(targets,bank,w,c,.01,objective.total)]


def main():
    binding=json.loads((P/'FULL_QUADRATIC_FRAME_V1_BINDING.json').read_text())
    assert all(digest(f)==h for f,h in binding.items())
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps(dict(gpu_accessed=False,body_forwards=0,corpus_access=False,
            groups=23,rank=4,outputs=10,penalty=.01,seeds=[0,937],fit_seconds_per_start=300,
            coefficients=370944)));return
    out=P/'FULL_QUADRATIC_FRAME_V1_RESULT.json';assert not out.exists()
    signal.alarm(900);torch.set_default_dtype(torch.float64);torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;start=time.perf_counter()
    sd=torch.load(CK,map_location='cpu',weights_only=True,mmap=True)
    u=sd['lm_head.weight'].double().cuda();chol=torch.linalg.cholesky(u.T@u);del u
    l,r,d=[sd[f'transformer.h.17.mlp.{key}.weight'].double().cuda() for key in ('Left','Right','Down')]
    d=chol.T@d
    total=json.loads((P/'FULLU_OUTPUT_FUNCTIONS_V1_AUDIT.json').read_text())['native_total']
    objective=MultioutputWeightObjective(torch.eye(1152,device='cuda'),total,l=l,r=r,d=d,penalty=.01)
    sampling_energy=(d.square().sum(0)*.5*(l.square().sum(1)*r.square().sum(1)+(l*r).sum(1).square())).cpu()
    reports=[];states=[]
    for seed in (0,937):
        indices=torch.multinomial(sampling_energy,46,replacement=False,generator=torch.Generator().manual_seed(seed)).tolist()
        frames=[];ratios=[]
        for a,b in zip(indices[::2],indices[1::2]):
            q,rr=torch.linalg.qr(torch.stack([l[a],r[a],l[b],r[b]],dim=1))
            ratio=float(rr.diag().abs().min()/rr.diag().abs().max());assert ratio>1e-8
            frames.append(q);ratios.append(ratio)
        bank=torch.stack(frames);packed=cores_for(bank)
        initial,_,initial_details=evaluate(objective,bank,packed,10)
        _,_,_,writer,gram,reg=objective.terms(View(bank,packed,10))
        replay=independent(objective,bank,packed,writer)
        errors=[abs(replay[0]-initial),abs(replay[1]-initial_details['residual'])]
        hv_error=geometry_check(objective,bank,packed,seed)
        assert max(errors)<=1e-8 and hv_error<=1e-6,(errors,hv_error)
        print(json.dumps(dict(seed=seed,initial_loss=initial,initial_capture=1-initial_details['residual'],
                              native_manifold_hessian_error=hv_error)),flush=True)
        bank,report=fit(objective,bank,seconds=300,tolerance=1e-7)
        packed=cores_for(bank)
        loss,residual,energy,writer,gram,reg=objective.terms(View(bank,packed,10))
        cross,_=objective.cross_gram(View(bank,packed,10))
        replay=independent(objective,bank,packed,writer)
        errors.extend([abs(replay[0]-float(loss)),abs(replay[1]-float(residual)),
                       abs(float(loss)-report['loss']),abs(float(residual)-report['residual']),
                       float((writer@reg-cross).norm()/cross.norm()),
                       float((bank.transpose(-1,-2)@bank-torch.eye(4,device='cuda')).abs().max())])
        capture=1-float(residual)
        relative=report['gradient_norm']*(92**.5)/max(capture,1e-12)
        converged=report['gradient_norm']<=1e-7 and relative<=1e-4
        residual_writer=torch.linalg.solve_triangular(chol.T,writer,upper=True)
        cache=Path(f'/dev/shm/bilin18_full_quadratic_frame_v1_s{seed}.pt');assert not cache.exists()
        torch.save(dict(bank=bank.cpu(),packed=packed.cpu(),writer=residual_writer.cpu(),
                        report=report,binding=binding),cache)
        row=dict(seed=seed,indices=indices,minimum_initial_qr_ratio=min(ratios),
            initial_loss=initial,initial_capture=1-initial_details['residual'],
            final=report,capture=capture,component_energy=float(energy),relative_stationarity=relative,
            converged=converged,native_manifold_hessian_error=hv_error,
            maximum_instrument_error=max(errors),writer_condition=float(torch.linalg.cond(reg)),
            cache=dict(path=str(cache),sha256=digest(cache),bytes=cache.stat().st_size,ephemeral=True))
        reports.append(row);states.append((bank,writer))
        with (P/f'FULL_QUADRATIC_FRAME_V1_SEED_{seed}.json').open('x') as f:json.dump(row,f,indent=2);f.write('\n')
        print(json.dumps({k:v for k,v in row.items() if k not in ('indices','final')},indent=2),flush=True)
    both=torch.cat([s[0] for s in states]);both_cores=cores_for(both)
    _,gram=objective.cross_gram(View(both,both_cores,10));w0,w1=[s[1] for s in states]
    n0=((w0.T@w0)*gram[:230,:230]).sum();n1=((w1.T@w1)*gram[230:,230:]).sum()
    dot=((w0.T@w1)*gram[:230,230:]).sum()
    result=dict(predictions={'pred_a_instrument':all(r['maximum_instrument_error']<=1e-8 and r['native_manifold_hessian_error']<=1e-6 for r in reports),
                            'pred_b_both_converged':all(r['converged'] for r in reports),
                            'pred_c_both_capture_gain':all(r['capture']>=1.05*.08634383041327387 for r in reports)},
        starts=reports,function_cosine=float(dot/(n0*n1).sqrt()),
        coefficient_count=370944,wall_seconds=time.perf_counter()-start,binding=binding,
        body_forwards=0,corpus_access=False,
        scope='New full-quadratic4-readerblock representation and block-Frobenius penalty. Whole-function similarity may reflect common output; no circuit or global-optimum claim.')
    with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('binding','starts')},indent=2),flush=True)
    assert result['predictions']['pred_a_instrument']


if __name__=='__main__':main()
