#!/usr/bin/env python3
"""pred_a replay/descent<=1e-8; pred_b block gap<=1e-6; pred_c 5%capture gain.

BQGATE:0forwards0seq. Weight-only fixed-frame block updates, <=100sweeps/180s.
"""
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time

RUNNER = Path(__file__).resolve()
P = RUNNER.parents[3] / 'basis_aligned/polynomial_causal'
sys.path.insert(0, str(P))
import torch
from conditional_block_svd_v1 import native_targets, full_cost, sweep, gaps
from orthogonal_multioutput_pymanopt_v2 import evaluate
from multioutput_quadratic_blocks_v1 import MultioutputWeightObjective

CK = Path('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    binding = json.loads((P/'CONDITIONAL_BLOCK_SVD_V1_BINDING.json').read_text())
    assert all(digest(p)==h for p,h in binding.items())
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps(dict(gpu_accessed=False, body_forwards=0, corpus_access=False,
                             seconds=180, groups=16, rank=16, outputs=4, penalty=.01)))
        return
    out = P/'CONDITIONAL_BLOCK_SVD_V1_RESULT.json'
    assert not out.exists()
    signal.alarm(400)
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    start = time.perf_counter()
    prior = json.loads((P/'BLOCK_TRUST_REGION_THIN_V1_RESULT.json').read_text())
    saved = torch.load(prior['cache']['path'], map_location='cpu', weights_only=True)
    sd = torch.load(CK, map_location='cpu', mmap=True, weights_only=True)
    u = sd['lm_head.weight'].double().cuda()
    chol = torch.linalg.cholesky(u.T@u)
    del u
    l,r,down = [sd[f'transformer.h.17.mlp.{key}.weight'].double().cuda()
                for key in ('Left','Right','Down')]
    down = chol.T@down
    bank = saved['bank'].double().cuda()
    original_bank = bank.clone()
    writers = (chol.T @ saved['writer'].double().cuda()).reshape(1152,16,4).permute(1,0,2).contiguous()
    cores = saved['packed'].double().cuda().reshape(136,16,4).permute(1,0,2).contiguous()
    total = json.loads((P/'FULLU_OUTPUT_FUNCTIONS_V1_AUDIT.json').read_text())['native_total']
    targets = native_targets(l,r,down,bank)
    cost,residual,energy = full_cost(targets,bank,writers,cores,.01,total)
    errors = [abs(float(cost)-prior['final']['loss']),abs(float(residual)-prior['final']['residual'])]
    assert max(errors)<1e-8, errors
    initial = float(cost)
    history = [dict(sweep=0,loss=float(cost),capture=1-float(residual))]
    maximum_increase = 0.
    fit_start = time.perf_counter()
    count = 0
    stop = 'sweep_limit'
    for count in range(1,101):
        maximum_increase = max(maximum_increase,sweep(targets,bank,writers,cores,.01)/total)
        if count%5==0 or time.perf_counter()-fit_start>=180 or count==100:
            cost,residual,energy = full_cost(targets,bank,writers,cores,.01,total)
            gap = max(gaps(targets,bank,writers,cores,.01))/total/max(1-float(residual),1e-12)
            row = dict(sweep=count,loss=float(cost),capture=1-float(residual),
                       relative_conditional_gap=gap,seconds=time.perf_counter()-fit_start)
            maximum_increase=max(maximum_increase,float(cost)-history[-1]['loss'])
            history.append(row)
            print(json.dumps(row),flush=True)
            if gap<=1e-6:
                stop='conditional_gap';break
            if time.perf_counter()-fit_start>=180:
                stop='time_limit';break
    fit_seconds = time.perf_counter()-fit_start
    packed = cores.permute(1,0,2).reshape(136,64)
    white_writer = writers.permute(1,0,2).reshape(1152,64)
    writer = torch.linalg.solve_triangular(chol.T,white_writer,upper=True)
    objective = MultioutputWeightObjective(torch.eye(1152,device='cuda'),total,l=l,r=r,d=down,penalty=.01)
    reduced,grad,details = evaluate(objective,bank,packed,4)
    gb,gc = grad
    inner=bank.transpose(-1,-2)@gb
    gb=gb-bank@((inner+inner.transpose(-1,-2))/2)
    gc=gc-packed*(gc*packed).sum(0,keepdim=True)
    errors.extend([float((bank-original_bank).abs().max()),float((packed.norm(dim=0)-1).abs().max()),maximum_increase])
    capture=1-float(residual)
    cache=Path('/dev/shm/bilin18_conditional_block_svd_v1.pt')
    assert not cache.exists()
    torch.save(dict(bank=bank.cpu(),packed=packed.cpu(),writer=writer.cpu(),binding=binding),cache)
    result=dict(predictions={'pred_a_instrument':max(errors)<=1e-8,
                            'pred_b_conditional_convergence':gap<=1e-6,
                            'pred_c_capture_gain':capture>=1.05*.08634383041327387},
                initial_loss=initial,loss=float(cost),residual=float(residual),
                captured_energy=capture,component_energy=float(energy),
                conditional_gap_relative=gap,maximum_instrument_error=max(errors),
                reduced_writer_loss=reduced,conditional_writer_extra_gain=float(cost)-reduced,
                input_tangent_norm=float(gb.norm()),core_tangent_norm=float(gc.norm()),
                original_relative_stationarity=max(float(gb.norm())*16,float(gc.norm())*8)/max(capture,1e-12),
                sweeps=count,history=history,stop=stop,fit_seconds=fit_seconds,
                wall_seconds=time.perf_counter()-start,
                cache=dict(path=str(cache),sha256=digest(cache),bytes=cache.stat().st_size,ephemeral=True),
                binding=binding,body_forwards=0,corpus_access=False,
                scope='Same fixed input frames and full-Ulambda.01family; exact one-block updates. Conditional convergence is not all-parameter convergence or circuit identification.')
    with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('binding','history')},indent=2),flush=True)
    assert result['predictions']['pred_a_instrument']


if __name__=='__main__':
    main()
