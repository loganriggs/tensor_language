#!/usr/bin/env python3
"""A FP64 verification; B two near-null modes; C gap; D>=3x matvec speed."""
# BQGATE: 0forwards0seq; independent mixed search of full native congruence operator.
import os,sys,json,time,signal,hashlib,statistics
from pathlib import Path
RUNNER=Path(__file__).resolve();P=RUNNER.parents[3]/'basis_aligned/polynomial_causal';sys.path.insert(0,str(P))
import torch
from congruence_block_operator_v1 import Operator
from congruence_eigensolver_v1 import solve
from mixed_precision_congruence_v1 import MixedOperator,verify
CK=Path('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin')
CACHE=Path('/dev/shm/bilin18_native_congruence_mixed_v1_vectors.pt')


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    binding=json.loads((P/'NATIVE_CONGRUENCE_MIXED_V1_BINDING.json').read_text())
    assert all(digest(p)==h for p,h in binding.items())
    assert json.loads((P/'MIXED_PRECISION_CONGRUENCE_V1_CONTROL.json').read_text())['instrument_passed']
    prior=json.loads((P/'NATIVE_CONGRUENCE_SPECTRUM_V1_RESULT.json').read_text())
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps(dict(gpu_accessed=False,model_loaded=False,body_forwards=0,corpus_access=False,
            requested_modes=4,ncv=40,maxiter=150,seed=997,search_tolerance=1e-5,fp64_residual_bar=1e-7)))
        return
    out=P/'NATIVE_CONGRUENCE_MIXED_V1_RESULT.json';assert not out.exists() and not CACHE.exists()
    signal.alarm(900);start=time.perf_counter();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    sd=torch.load(CK,weights_only=True,map_location='cpu',mmap=True)
    u=sd['lm_head.weight'].double().cuda()
    l,r,d=[sd[f'transformer.h.17.mlp.{key}.weight'].double().cuda() for key in ['Left','Right','Down']]
    op=Operator(l,r,d.T@(u.T@u)@d);mixed=MixedOperator(op)
    trace_error=abs(float(op.total)/prior['native_total']-1)
    z=torch.randn(1152,1152,generator=torch.Generator().manual_seed(999),dtype=torch.float64).cuda()
    exact=op(z);approx=mixed(z);action_error=float((exact-approx).norm()/exact.norm())
    timings={}
    for name,operator in [('fp64',op),('fp32_search',mixed)]:
        operator(z);torch.cuda.synchronize();samples=[]
        for _ in range(5):
            torch.cuda.synchronize();tic=time.perf_counter();operator(z);torch.cuda.synchronize()
            samples.append(time.perf_counter()-tic)
        timings[name]=samples
    speed=statistics.median(timings['fp64'])/statistics.median(timings['fp32_search'])
    print(json.dumps(dict(action_relative_error=action_error,speed_ratio=speed,timings=timings)),flush=True)
    report,vectors=solve(mixed,k=4,ncv=40,maxiter=150,tol=1e-5,seed=997,
                         progress=lambda row:print(json.dumps(row),flush=True))
    checked=verify(op,report['eigenvalues'],vectors)
    order=sorted(range(len(checked)),key=lambda i:checked[i]['fp64_rayleigh'])
    checked=[checked[i] for i in order];vectors=vectors[order]
    values=[x['fp64_rayleigh'] for x in checked];bound=report['operator_norm_upper_bound']
    valid=report['converged'] and len(checked)==4 and max(x['fp64_residual_over_bound'] for x in checked)<=1e-7 and max(x['identity_overlap'] for x in checked)<=1e-8 and report['orthogonality_error']<=1e-8 and min(values,default=-1)>=-1e-8*bound and trace_error<=1e-8 and action_error<=1e-5 and bool(torch.isfinite(vectors).all())
    old_cache=Path(prior['witness_cache']['path']);assert digest(old_cache)==prior['witness_cache']['sha256']
    old=torch.load(old_cache,weights_only=True,map_location='cpu')['vectors']
    overlap=old.flatten(1)@vectors.flatten(1).T
    principal=torch.linalg.svdvals(overlap)
    oldvalues=prior['solver']['eigenvalues']
    differences=[min(abs(v-x) for x in values) for v in oldvalues] if values else []
    torch.save(dict(vectors=vectors,fp64_verification=checked,solver=report,binding=binding),CACHE)
    result=dict(predictions={'pred_a_fp64_verified_convergence':valid,
        'pred_b_two_near_null_modes':valid and values[1]<=.01*report['mean_nontrivial_eigenvalue'],
        'pred_c_gap_after_two_modes':valid and values[2]>1e-10*bound and values[1]/values[2]<=.10,
        'pred_d_matvec_speed':speed>=3},solver=report,fp64_verification=checked,
        fp64_eigenvalues=values,action_relative_error=action_error,native_trace_relative_error=trace_error,
        precision_timings=timings,matvec_speed_ratio=speed,
        prior_comparison=dict(principal_cosines=principal.tolist(),nearest_eigenvalue_absolute_differences=differences),
        witness_cache=dict(path=str(CACHE),sha256=digest(CACHE),bytes=CACHE.stat().st_size,ephemeral=True),
        price=dict(body_forwards=0,corpus_access=False,output_rank_truncation=False,input_rank_truncation=False),
        wall_seconds=time.perf_counter()-start,
        scope='Independent mixed search with original FP64 convergence verification; no native blocks or circuits identified by eigenvalues alone.')
    with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
