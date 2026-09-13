"""Weight-only writer coordinates. Numerical rank is not semantic identification."""
import torch

@torch.no_grad()
def prepare(writers,relative_tolerance=None):
    """Return unit-RMS orthogonal rows and map: amplitudes @ map @ basis.

    Factorization uses FP64. Default rank cutoff is the usual scale-aware
    floating-point cutoff; any truncation remains explicit in residual/bound.
    Keep returned map in FP64 when transforming cancelling amplitudes.
    """
    if writers.ndim!=2 or not torch.isfinite(writers).all():
        raise ValueError('Expected finite [writers, residual_width] matrix')
    w=writers.double();u,s,vh=torch.linalg.svd(w,full_matrices=False)
    tol=max(w.shape)*torch.finfo(w.dtype).eps if relative_tolerance is None else relative_tolerance
    if tol<0:raise ValueError('Negative rank tolerance')
    cutoff=s[0]*tol if s.numel() else 0.
    rank=int((s>cutoff).sum());scale=w.shape[1]**.5
    basis=vh[:rank]*scale;mapping=u[:,:rank]*s[:rank]/scale
    residual=w-mapping@basis
    return {'basis':basis,'map':mapping,'rank':rank,'singular_values':s,
            'relative_tolerance':tol,'residual':residual,
            'residual_operator_norm':torch.linalg.matrix_norm(residual,ord=2)}

def transform(amplitudes,coordinates):
    return amplitudes.double()@coordinates['map']

def edit_error_bound(amplitudes,coordinates):
    return amplitudes.double().norm(dim=-1)*coordinates['residual_operator_norm']
