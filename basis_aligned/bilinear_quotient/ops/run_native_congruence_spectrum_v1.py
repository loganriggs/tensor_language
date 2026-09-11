#!/usr/bin/env python3
"""A full numerical/convergence check; B two near-null modes; C spectral gap."""
# BQGATE: 0forwards0seq; full-U congruence normal operator, no tensor truncation.
import os,sys,json,time,signal,hashlib
from pathlib import Path
RUNNER=Path(__file__).resolve();P=RUNNER.parents[3]/'basis_aligned/polynomial_causal'
sys.path.insert(0,str(P))
import torch
from congruence_block_operator_v1 import Operator
from congruence_eigensolver_v1 import solve
CK=Path('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin')
CACHE=Path('/dev/shm/bilin18_native_congruence_spectrum_v1_vectors.pt')


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    binding=json.loads((P/'NATIVE_CONGRUENCE_SPECTRUM_V1_BINDING.json').read_text())
    assert all(digest(p)==h for p,h in binding.items())
    assert json.loads((P/'CONGRUENCE_EIGENSOLVER_V1_CONTROL.json').read_text())['instrument_passed']
    assert json.loads((P/'CONGRUENCE_BLOCK_OPERATOR_V1_CONTROL.json').read_text())['instrument_passed']
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps(dict(model_loaded=False,gpu_accessed=False,body_forwards=0,
            corpus_access=False,matrix_dimension=1152,operator_dimension=1152**2,
            requested_modes=4,ncv=20,maxiter=100,cache=str(CACHE))))
        return
    out=P/'NATIVE_CONGRUENCE_SPECTRUM_V1_RESULT.json';assert not out.exists() and not CACHE.exists()
    signal.alarm(1200);start=time.perf_counter();torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    sd=torch.load(CK,weights_only=True,map_location='cpu',mmap=True)
    u=sd['lm_head.weight'].double().cuda()
    l,r,d=[sd[f'transformer.h.17.mlp.{key}.weight'].double().cuda() for key in ['Left','Right','Down']]
    assert l.shape==r.shape==(4608,1152) and u.shape==(50304,1152)
    metric=u.T@u;output_gram=d.T@metric@d
    op=Operator(l,r,output_gram)
    reference=json.loads((P/'FULLU_OUTPUT_FUNCTIONS_V1_AUDIT.json').read_text())['native_total']
    trace_error=abs(float(op.total)/reference-1)
    gen=torch.Generator().manual_seed(971)
    a,b=[torch.randn(1152,1152,generator=gen,dtype=torch.float64).cuda() for _ in range(2)]
    la,lb=op(a),op(b)
    adjoint_error=float(((a*lb).sum()-(la*b).sum()).abs()/(a.norm()*lb.norm()+la.norm()*b.norm()))
    identity_error=float(op(torch.eye(1152,dtype=torch.float64,device='cuda')).norm()/op.total)
    random_energy=float((a*la).sum()/op.total/a.square().sum())
    print(json.dumps(dict(native_trace_error=trace_error,adjoint_error=adjoint_error,
                         identity_error=identity_error,random_normalized_energy=random_energy)),flush=True)
    report,vectors=solve(op,k=4,ncv=20,maxiter=100,tol=1e-8,seed=961,
                         progress=lambda row:print(json.dumps(row),flush=True))
    values=report['eigenvalues'];bound=report['operator_norm_upper_bound']
    core=max(trace_error,adjoint_error,identity_error)<=1e-8 and max(adjoint_error,identity_error)<=1e-10 and random_energy>=0
    convergence=report['converged'] and len(values)==4 and max(report['residuals_over_operator_bound'],default=1)<=1e-7 and max(report['identity_overlaps'],default=1)<=1e-8 and report['orthogonality_error']<=1e-8
    valid=core and convergence and min(values,default=-1)>=-1e-8*bound and bool(torch.isfinite(vectors).all())
    near=valid and values[1]<=.01*report['mean_nontrivial_eigenvalue']
    gap=valid and values[2]>1e-10*bound and values[1]/values[2]<=.10
    # Retain exact computed vectors in RAM-backed storage for follow-up audits.
    torch.save(dict(vectors=vectors,solver=report,binding=binding,checkpoint=str(CK)),CACHE)
    result=dict(predictions={'pred_a_numerics_and_convergence':valid,'pred_b_two_near_null_modes':near,
                            'pred_c_gap_after_two_modes':gap},solver=report,
        native_total=float(op.total),native_trace_error=trace_error,adjoint_error=adjoint_error,
        identity_error=identity_error,random_normalized_energy=random_energy,
        instrument_core_passed=core,witness_cache=dict(path=str(CACHE),sha256=digest(CACHE),bytes=CACHE.stat().st_size,ephemeral=True),
        price=dict(body_forwards=0,corpus_access=False,output_rank_truncation=False,input_rank_truncation=False),
        wall_seconds=time.perf_counter()-start,
        scope='Full-U native congruence spectrum; additional near-null solutions are only block candidates. No circuit identification.')
    with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
