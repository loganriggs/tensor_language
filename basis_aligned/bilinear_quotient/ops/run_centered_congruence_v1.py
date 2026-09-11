#!/usr/bin/env python3
"""A FP64 verified convergence; B two near-null modes; C gap after two."""
# BQGATE: 0forwards0seq; entire centered quadratic family plus retained common channel.
import os,sys,json,time,signal,hashlib
from pathlib import Path
RUNNER=Path(__file__).resolve();P=RUNNER.parents[3]/'basis_aligned/polynomial_causal';sys.path.insert(0,str(P))
import torch
from congruence_block_operator_v1 import Operator
from congruence_eigensolver_v1 import solve
from mixed_precision_congruence_v1 import MixedOperator,verify
CK=Path('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin')
CACHE=Path('/dev/shm/bilin18_centered_congruence_v1_vectors.pt')


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    binding=json.loads((P/'CENTERED_CONGRUENCE_V1_BINDING.json').read_text())
    assert all(digest(p)==h for p,h in binding.items())
    prior=json.loads((P/'NATIVE_CONGRUENCE_MIXED_V1_RESULT.json').read_text())
    full_total=json.loads((P/'FULLU_OUTPUT_FUNCTIONS_V1_AUDIT.json').read_text())['native_total']
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps(dict(gpu_accessed=False,model_loaded=False,body_forwards=0,corpus_access=False,
            requested_modes=4,ncv=40,maxiter=150,seed=1009,common_channel_retained=True)))
        return
    out=P/'CENTERED_CONGRUENCE_V1_RESULT.json';assert not out.exists() and not CACHE.exists()
    signal.alarm(900);start=time.perf_counter();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    sd=torch.load(CK,weights_only=True,map_location='cpu',mmap=True)
    u=sd['lm_head.weight'].double().cuda()
    l,r,d=[sd[f'transformer.h.17.mlp.{key}.weight'].double().cuda() for key in ['Left','Right','Down']]
    assert u.shape==(50304,1152) and l.shape==r.shape==(4608,1152)
    mean=u.mean(0);metric=u.T@u-len(u)*mean[:,None]*mean[None,:]
    op=Operator(l,r,d.T@metric@d);mixed=MixedOperator(op)
    trace_error=abs(float(op.total)/92104252412.19983-1)
    coefficients=mean@d;common=(l.T*coefficients)@r;common=(common+common.T)/2
    common_energy=float(len(u)*common.square().sum())
    total_error=abs((float(op.total)+common_energy)/full_total-1)
    z=torch.randn(1152,1152,generator=torch.Generator().manual_seed(1011),dtype=torch.float64).cuda()
    exact=op(z);approx=mixed(z);action_error=float((exact-approx).norm()/exact.norm())
    report,vectors=solve(mixed,k=4,ncv=40,maxiter=150,tol=1e-5,seed=1009,
                         progress=lambda row:print(json.dumps(row),flush=True))
    checked=verify(op,report['eigenvalues'],vectors)
    order=sorted(range(len(checked)),key=lambda i:checked[i]['fp64_rayleigh'])
    checked=[checked[i] for i in order];vectors=vectors[order]
    values=[x['fp64_rayleigh'] for x in checked];bound=report['operator_norm_upper_bound']
    valid=report['converged'] and len(checked)==4 and max(x['fp64_residual_over_bound'] for x in checked)<=1e-7 and max(x['identity_overlap'] for x in checked)<=1e-8 and report['orthogonality_error']<=1e-8 and min(values,default=-1)>=-1e-8*bound and max(trace_error,total_error)<=1e-8 and action_error<=1e-5 and bool(torch.isfinite(vectors).all())
    restoration=[]
    for value,vector in zip(values,vectors):
        z=vector.cuda();residual=common@z-z.T@common
        extra=float(len(u)*residual.square().sum()/z.square().sum()/full_total)
        centered=value*float(op.total)/full_total
        restoration.append(dict(centered_cost_in_full_units=centered,common_cost_in_full_units=extra,
                                full_rayleigh=centered+extra,common_fraction_of_constraint_cost=extra/(centered+extra)))
    old_cache=Path(prior['witness_cache']['path']);assert digest(old_cache)==prior['witness_cache']['sha256']
    old=torch.load(old_cache,weights_only=True,map_location='cpu')['vectors']
    principal=torch.linalg.svdvals(old.flatten(1)@vectors.flatten(1).T)
    torch.save(dict(vectors=vectors,fp64_verification=checked,solver=report,binding=binding),CACHE)
    result=dict(predictions={'pred_a_fp64_verified_convergence':valid,
        'pred_b_two_near_null_modes':valid and values[1]<=.01*report['mean_nontrivial_eigenvalue'],
        'pred_c_gap_after_two_modes':valid and values[2]>1e-10*bound and values[1]/values[2]<=.10},
        solver=report,fp64_verification=checked,fp64_eigenvalues=values,
        native_centered_total=float(op.total),common_energy=common_energy,
        native_trace_relative_error=trace_error,common_total_closure_error=total_error,
        action_relative_error=action_error,common_restoration=restoration,
        full_vs_centered_principal_cosines=principal.tolist(),
        witness_cache=dict(path=str(CACHE),sha256=digest(CACHE),bytes=CACHE.stat().st_size,ephemeral=True),
        price=dict(body_forwards=0,corpus_access=False,output_rank_truncation=False,input_rank_truncation=False),
        wall_seconds=time.perf_counter()-start,
        scope='Centered congruence block screen with the exact common function retained separately. No original-coordinate block or circuit identification.')
    with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
