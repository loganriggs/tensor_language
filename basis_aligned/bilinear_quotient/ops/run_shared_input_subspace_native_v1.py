#!/usr/bin/env python3
"""pred_a: exact algebra1e-8; pred_b: centered rank11 bound<.15; pred_c: capture/bound>=.8."""
# BQGATE: 0forwards0seq; exact weight-only input-subspace energy and bounds.
import os, sys, json, time, signal, hashlib
from pathlib import Path
RUNNER = Path(__file__).resolve()
P = RUNNER.parents[3]/'basis_aligned/polynomial_causal'; sys.path.insert(0, str(P))
import torch
from shared_input_subspace_v1 import value_gradient
from congruence_block_operator_v1 import sandwich
CK = Path('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin')
RANKS = [1,4,8,11,16,32,64,128]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()


def main():
    binding = json.loads((P/'SHARED_INPUT_SUBSPACE_NATIVE_V1_BINDING.json').read_text())
    assert all(digest(p)==h for p,h in binding.items())
    assert json.loads((P/'SHARED_INPUT_SUBSPACE_V1_CONTROL.json').read_text())['instrument_passed']
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps(dict(gpu_accessed=False, body_forwards=0, corpus_access=False,
                              ranks=RANKS, nonlinear_optimization=False))); return
    out = P/'SHARED_INPUT_SUBSPACE_NATIVE_V1_RESULT.json'; assert not out.exists()
    cache = Path('/dev/shm/bilin18_shared_input_subspace_native_v1.pt'); assert not cache.exists()
    signal.alarm(1200); torch.set_num_threads(2); torch.backends.cuda.matmul.allow_tf32=False
    started=time.perf_counter(); sd=torch.load(CK,weights_only=True,map_location='cpu',mmap=True)
    u=sd['lm_head.weight'].double().cuda()
    l,r,down=[sd[f'transformer.h.17.mlp.{key}.weight'].double().cuda() for key in ['Left','Right','Down']]
    eye=torch.eye(1152,dtype=u.dtype,device=u.device)
    prior=json.loads((P/'SHARED_INPUT_FACTOR_NATIVE_V1_RESULT.json').read_text())
    results={}; saved={}; errors=[]; strict_checks=[]
    for name,output in [('full',u),('centered',u-u.mean(0))]:
        k=down.T@(output.T@output)@down
        second=sandwich(l,r,k,eye); second=(second+second.T)/2
        eigen,vectors=torch.linalg.eigh(second); total=float(second.trace())
        errors.append(abs(total/prior['objectives'][name]['total_energy']-1))
        errors.append(abs(2*float(eigen[-1])/total/prior['objectives'][name]['analytic_one_reader_upper_bound']-1))
        curve=[]
        for rank in RANKS:
            e=vectors[:,-rank:]; value,grad,inside=value_gradient(e,l,r,k,second)
            capture=float(value/total); ins=float(inside/total)
            upper=min(1.,2*float(eigen[-rank:].sum())/total)
            errors.extend([float((e.T@e-torch.eye(rank,device=e.device,dtype=e.dtype)).abs().max()),
                           max(0.,capture-upper),max(0.,upper/2-capture)])
            row=dict(rank=rank,capture=capture,inside=ins,mixed=capture-ins,
                     universal_rank_bound=upper,capture_over_bound=capture/upper,
                     tangent_stationarity=float((grad-e@(e.T@grad)).norm()/total),
                     dense_partner_program_numbers=rank*(1152+1152**2),
                     scalar_products=rank*1152)
            strict_checks.extend([float((e.T@e-torch.eye(rank,device=e.device,dtype=e.dtype)).abs().max())<=1e-10, capture<=upper+1e-9, capture>=upper/2-1e-9])
            curve.append(row); print(json.dumps(dict(metric=name,**row)),flush=True)
            if rank==4:
                torch.manual_seed(1337); x=torch.randn(32,1152,dtype=e.dtype,device=e.device)
                xe=(x@e)@e.T; xc=x-xe
                def f(xx):return ((xx@l.T)*(xx@r.T))@down.T
                expected=f(x)-f(xc)
                predicted=((xe@l.T)*(x@r.T)+(x@l.T)*(xe@r.T)-(xe@l.T)*(xe@r.T))@down.T
                errors.append(float((expected-predicted).norm()/expected.norm()))
                strict_checks.append(errors[-1]<=1e-10)
        cumulative=2*eigen.flip(0).cumsum(0)/total
        results[name]=dict(total_energy=total,curve=curve,
            minimum_readers_for_half_from_bound=int(torch.searchsorted(cumulative,.5))+1,
            minimum_readers_for_ninety_percent_from_bound=int(torch.searchsorted(cumulative,.9))+1)
        saved[name]=dict(second=second.cpu(),eigenvalues=eigen.cpu(),eigenvectors=vectors.cpu())
    valid=max(errors)<=1e-8 and all(strict_checks)
    c=next(row for row in results['centered']['curve'] if row['rank']==11)
    torch.save(saved,cache)
    result=dict(predictions={'pred_a_instrument':valid,
            'pred_b_small_family_ceiling':valid and c['universal_rank_bound']<.15,
            'pred_c_spectral_efficiency':valid and c['capture_over_bound']>=.8},
        metrics=results,maximum_instrument_error=max(errors),
        price=dict(native_quadratic_weight_numbers=3*1152*4608,body_forwards=0,corpus_access=False,
                   full_native_background_required=True,nonlinear_optimization=False),
        cache=dict(path=str(cache),sha256=digest(cache),bytes=cache.stat().st_size,ephemeral=True),
        wall_seconds=time.perf_counter()-started,
        scope='Weight coefficient projection and universal rank-r mixed-interaction bound. Spectral surrogate exact, extraction objective not optimized. No semantic or natural-state negative.')
    with out.open('x') as f: json.dump(result,f,indent=2); f.write('\n')
    print(json.dumps({key:value for key,value in result.items() if key!='metrics'}),flush=True)


if __name__=='__main__':main()
