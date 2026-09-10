"""Sparse quadratic-core frame updates, with exact fixed-support ascent bounds.

Q[d,r] is an orthonormal frame, not merely a subspace: rotating its columns
changes sparsity. Preserve the vertical/skew tangent component.
"""
import torch
from sparse_orthogonal_quadratic_core_v1 import orthogonal_core

def project_tangent(q,g):
    inner=q.T@g
    return g-q@((inner+inner.T)/2)

def retract(q,direction,step):
    z,r=torch.linalg.qr(q+step*direction,mode='reduced')
    signs=torch.where(r.diag()<0,-torch.ones_like(r.diag()),torch.ones_like(r.diag()))
    return z*signs

def selected_score(q,l,r,d,edges,total):
    a=l@q;b=r@q;i,j=edges
    normalizer=torch.where(i==j,torch.full_like(i,2,dtype=q.dtype),torch.full_like(i,2**.5,dtype=q.dtype))
    h=(a[:,i]*b[:,j]+a[:,j]*b[:,i])/normalizer
    writer=d@h
    return writer.square().sum()/total

def best_edges(q,l,r,d,total,count):
    with torch.no_grad():
        w,edges=orthogonal_core(l,r,d,q.T)
        energies=w.square().sum(0);selected=energies.topk(count).indices
        return edges[:,selected],energies[selected].sum()/total

def score_gradient(q,l,r,d,edges,total):
    variable=q.detach().requires_grad_(True)
    score=selected_score(variable,l,r,d,edges,total)
    gradient=torch.autograd.grad(score,variable)[0]
    return score.detach(),project_tangent(q,gradient).detach()

def armijo_step(q,gradient,direction,l,r,d,edges,total,initial_step=1.,max_backtracks=25):
    slope=float((gradient*direction).sum())
    if slope<=0:direction=gradient;slope=float(gradient.square().sum())
    with torch.no_grad():
        old=selected_score(q,l,r,d,edges,total)
        step=initial_step
        for attempt in range(max_backtracks):
            trial=retract(q,direction,step)
            lower=selected_score(trial,l,r,d,edges,total)
            if float(lower-old)>=1e-4*step*slope:
                return trial,dict(accepted=True,step=step,backtracks=attempt,old_score=float(old),new_score_lower_bound=float(lower),directional_derivative=slope)
            step/=2
    return q,dict(accepted=False,step=0.,backtracks=max_backtracks,old_score=float(old),new_score_lower_bound=float(old),directional_derivative=slope)
