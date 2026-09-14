"""Frozen exact scalar regression; see MASKED_DIRECTION_REFIT_V1_PREREGISTRATION."""
import hashlib
import json
import os
import signal
import time
from datetime import datetime,timezone

import torch
from retained_objective_context_v1 import Contexts, masked_balanced_direction, branch_errors, P
from sparse_interaction_executor_v1 import Executor


@torch.no_grad()
def run():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    torch.set_num_threads(2)
    signal.alarm(120)
    started=time.perf_counter()
    model=Contexts()
    package,direction_values,direction=masked_balanced_direction(model)
    original=model.decode('SPARSE_INTERACTION_EXECUTOR_V1')
    error=original-model.tensor
    def responses(seed):
        z,terms,denominator=model.sample(seed)
        e=branch_errors(error,z,terms,denominator).sum(1)
        d=branch_errors(direction,z,terms,denominator).sum(1)
        return e,d
    fits=[]
    for first_seed in (170217000,170218000):
        ee=ed=dd=0.
        for seed in range(first_seed,first_seed+128):
            e,d=responses(seed)
            ee+=float(e.square().sum());ed+=float((e*d).sum());dd+=float(d.square().sum())
        assert dd>0
        alpha=-ed/dd
        fits.append(dict(first_seed=first_seed,contexts=8192,alpha=alpha,
                         original_energy=ee/8192,optimum_energy=(ee+2*alpha*ed+alpha*alpha*dd)/8192,
                         normalized_stationarity=abs(ed+alpha*dd)/max(abs(ed),1e-30)))
    alpha=fits[0]['alpha']
    changed=dict(package,values=(package['values'].double()+alpha*direction_values).float())
    artifact=P/'MASKED_DIRECTION_REFIT_V1_PROGRAM.pt'
    assert not artifact.exists()
    torch.save(changed,artifact)
    fitted=model.decode('MASKED_DIRECTION_REFIT_V1')
    serialized_error=fitted-model.tensor
    executor=Executor(changed)
    baseline=[];candidate=[];basecontrast=[];fitcontrast=[]
    quadratic_replay=runtime_replay=0.
    for seed in range(170219000,170219128):
        z,terms,denominator=model.sample(seed)
        e=branch_errors(error,z,terms,denominator).sum(1)
        d=branch_errors(direction,z,terms,denominator).sum(1)
        exact_directional=branch_errors(error+alpha*direction,z,terms,denominator).sum(1)
        quadratic_replay=max(quadratic_replay,float((exact_directional-(e+alpha*d)).norm()/exact_directional.norm()))
        actual=branch_errors(serialized_error,z,terms,denominator).sum(1)
        if seed==170219000:
            dense=torch.einsum('oih,ni,nh->no',fitted,z,terms.sum(1))
            sparse=executor(z.float(),terms.sum(1).float()).double()
            runtime_replay=float((dense-sparse).norm()/dense.norm())
        baseline.append(e.square().sum(-1));candidate.append(actual.square().sum(-1))
        basecontrast.append((e[:,::2]-e[:,1::2]).square().sum(-1)/2)
        fitcontrast.append((actual[:,::2]-actual[:,1::2]).square().sum(-1)/2)
    baseline,candidate,basecontrast,fitcontrast=map(torch.cat,(baseline,candidate,basecontrast,fitcontrast))
    gain=baseline-candidate;se=float(gain.std()/len(gain)**.5)
    improvement=float(gain.mean()/baseline.mean())
    contrast_ratio=float(fitcontrast.mean()/basecontrast.mean())
    invariant=all(torch.equal(changed[k],package[k]) for k in ('mask','output','head'))
    price_ratio=artifact.stat().st_size/(P/'SPARSE_INTERACTION_EXECUTOR_V1_PROGRAM.pt').stat().st_size
    result=dict(utc=datetime.now(timezone.utc).isoformat(),fits=fits,validation_contexts=8192,
        validation_seeds=[170219000,170219127],original_mean_energy=float(baseline.mean()),
        candidate_mean_energy=float(candidate.mean()),relative_energy_improvement=improvement,
        paired_improvement=float(gain.mean()),paired_standard_error=se,
        contrast_energy_ratio=contrast_ratio,alpha_gap=abs(fits[0]['alpha']-fits[1]['alpha']),
        quadratic_replay=quadratic_replay,fp32_executor_replay=runtime_replay,
        mask_and_adapters_unchanged=invariant,artifact_bytes=artifact.stat().st_size,
        artifact_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),price_ratio=price_ratio,
        pred_a=max(quadratic_replay,runtime_replay)<=1e-6 and invariant and abs(price_ratio-1)<=.01,
        pred_b=improvement>=.05 and float(gain.mean())>3*se,
        pred_c=contrast_ratio<=1.01 and abs(fits[0]['alpha']-fits[1]['alpha'])<=.1,
        seconds=time.perf_counter()-started,
        scope='Exact scalar least squares in a fixed weight-derived direction, frozen support and adapters. Synthetic Gaussian raw edits, conditional normalized objective; no native text, full sparse optimization or adoption.')
    signal.alarm(0)
    return result


if __name__=='__main__':
    target=P/'MASKED_DIRECTION_REFIT_V1_RESULT.json';assert not target.exists()
    result=run()
    with target.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2))
