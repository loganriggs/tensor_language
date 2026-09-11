"""Restarted Lanczos on a trace-free congruence normal operator; no dense lift."""
import json,time
from pathlib import Path
import numpy as np
import torch
from scipy.sparse.linalg import LinearOperator,eigsh,ArpackNoConvergence


def solve(operator,k=4,ncv=20,maxiter=100,tol=1e-8,seed=961,progress=None):
    d=operator.second.shape[0]
    device=operator.second.device
    identity=torch.eye(d,dtype=torch.float64,device=device)
    total=float(operator.total)
    # ||L|| <= 4||sum Q_v²||; shift only the known scalar-null direction.
    bound=4*float(torch.linalg.eigvalsh((operator.second+operator.second.T)/2)[-1])/total
    calls=0;seconds=0.
    def matvec(array):
        nonlocal calls,seconds
        start=time.perf_counter()
        z=torch.from_numpy(np.array(array,dtype=np.float64,copy=True)).reshape(d,d).to(device)
        scalar=z.trace()/d
        z=z-scalar*identity
        action=operator(z)/total
        action=action-action.trace()/d*identity+1.1*bound*scalar*identity
        result=action.flatten().cpu().numpy()
        calls+=1;seconds+=time.perf_counter()-start
        if progress is not None and calls%25==0:
            progress(dict(matvecs=calls,matvec_seconds=seconds))
        return result
    linear=LinearOperator((d*d,d*d),matvec=matvec,dtype=np.float64)
    initial=np.random.default_rng(seed).standard_normal((d,d))
    initial-=np.trace(initial)/d*np.eye(d)
    start=time.perf_counter()
    try:
        values,vectors=eigsh(linear,k=k,which='SA',ncv=min(ncv,d*d),maxiter=maxiter,
                             tol=tol,v0=initial.ravel())
        converged=True;reason='arpack_converged'
    except ArpackNoConvergence as exc:
        values,vectors=exc.eigenvalues,exc.eigenvectors
        converged=False;reason='arpack_iteration_limit'
    order=np.argsort(values);values=values[order];vectors=vectors[:,order]
    eigen_residuals=[];traces=[]
    for i,value in enumerate(values):
        eigen_residuals.append(float(np.linalg.norm(matvec(vectors[:,i])-value*vectors[:,i])/bound))
        traces.append(float(abs(np.trace(vectors[:,i].reshape(d,d)))/np.sqrt(d)))
    orth_error=float(np.linalg.norm(vectors.T@vectors-np.eye(len(values))))
    result=dict(converged=converged,terminal_reason=reason,eigenvalues=values.tolist(),
                residuals_over_operator_bound=eigen_residuals,identity_overlaps=traces,
                orthogonality_error=orth_error,operator_norm_upper_bound=bound,
                mean_nontrivial_eigenvalue=2/(d+1),matvecs=calls,matvec_seconds=seconds,
                wall_seconds=time.perf_counter()-start,requested_modes=k,ncv=ncv,
                maxiter=maxiter,tolerance=tol,seed=seed)
    return result,torch.from_numpy(vectors.copy()).T.reshape(len(values),d,d)


def control():
    from congruence_block_operator_v1 import Operator,factor_forms
    torch.set_default_dtype(torch.float64);torch.manual_seed(963)
    d=8;blocks=[]
    for _ in range(6):
        a=torch.randn(3,3);b=torch.randn(5,5)
        blocks.append(torch.block_diag((a+a.T)/2,(b+b.T)/2))
    mixing=torch.eye(d)+.1*torch.randn(d,d)
    forms=mixing.T@torch.stack(blocks)@mixing
    l,r,w=factor_forms(forms);op=Operator(l,r,w.T@w)
    result,vectors=solve(op,k=4,ncv=32,maxiter=500,tol=1e-10)
    basis=torch.eye(d*d).reshape(d*d,d,d)
    dense=torch.stack([op(e).flatten()/op.total for e in basis],1)
    trace_error=abs(float(dense.trace())/(2*(d-1))-1)
    identity=torch.eye(d).flatten()/d**.5
    dense+=1.1*result['operator_norm_upper_bound']*identity[:,None]*identity[None,:]
    expected=torch.linalg.eigvalsh((dense+dense.T)/2)[:4]
    error=float((torch.tensor(result['eigenvalues'])-expected).abs().max()/result['operator_norm_upper_bound'])
    passed=result['converged'] and error<1e-9 and trace_error<1e-10 and max(result['residuals_over_operator_bound'])<1e-9 and max(result['identity_overlaps'])<1e-9
    receipt=dict(instrument_passed=passed,dense_eigenvalue_error_over_bound=error,
                 lowest_expected=expected.tolist(),operator_trace_relative_error=trace_error,solver=result,
                 scope='Synthetic two-block indefinite congruence problem, scalar identity removed.')
    Path(__file__).with_name('CONGRUENCE_EIGENSOLVER_V1_CONTROL.json').write_text(json.dumps(receipt,indent=2)+'\n')
    assert passed,receipt
    return receipt


if __name__=='__main__':
    print(json.dumps(control(),indent=2))
