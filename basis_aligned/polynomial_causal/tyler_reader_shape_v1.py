"""Trace-normalized angular shape for scale-invariant oblique reader recovery.

Tyler fixed point; no shrinkage or rank truncation. Existence requires enough
samples outside every proper subspace. Shape is affine equivariant up to scale.
"""
import time
import torch


def normalize_columns(y):
    norms=y.norm(dim=0,keepdim=True)
    assert bool((norms>0).all()),'Zero directions must be handled explicitly'
    return y/norms


def trace_normalize(shape):
    return shape*(shape.shape[0]/shape.trace())


@torch.no_grad()
def tyler_shape(y,max_steps=500,tolerance=1e-11,max_seconds=30):
    y=normalize_columns(y)
    d,n=y.shape
    assert n>d
    shape=trace_normalize(y@y.T/n)
    history=[];started=time.perf_counter();converged=False;stop='step_limit'
    for step in range(1,max_steps+1):
        chol=torch.linalg.cholesky(shape)
        q=torch.linalg.solve_triangular(chol,y,upper=False).square().sum(0)
        weighted=y/q.sqrt()
        updated=trace_normalize(weighted@weighted.T*(d/n))
        change=float((updated-shape).norm()/shape.norm())
        shape=updated
        history.append(dict(step=step,relative_change=change,seconds=time.perf_counter()-started))
        if change<=tolerance:
            converged=True;stop='converged';break
        if time.perf_counter()-started>=max_seconds:
            stop='time_limit';break
    eigenvalues=torch.linalg.eigvalsh(shape)
    return shape,dict(converged=converged,stop=stop,history=history,
        minimum_eigenvalue=float(eigenvalues[0]),condition=float(eigenvalues[-1]/eigenvalues[0]),
        seconds=time.perf_counter()-started)


def roots(shape):
    eigenvalues,vectors=torch.linalg.eigh(shape)
    assert bool((eigenvalues>0).all())
    square_root=(vectors*eigenvalues.sqrt())@vectors.T
    inverse_root=(vectors*eigenvalues.rsqrt())@vectors.T
    return square_root,inverse_root


def reader_maps(rotation,shape):
    """Weight coding and input-feature maps differ for an oblique dictionary."""
    square_root,inverse_root=roots(shape)
    return dict(weight_coder=rotation@inverse_root,
                input_features=rotation@square_root,
                synthesis=square_root@rotation.T)
