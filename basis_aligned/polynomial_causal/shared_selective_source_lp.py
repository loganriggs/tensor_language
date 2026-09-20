"""Maximize worst signed first-order retention with fixed shared source weights."""
import numpy as np
from scipy.optimize import linprog

def choose(gradients,reference,control_budget=.08):
    g=np.asarray(gradients,dtype=float);reference=np.asarray(reference,dtype=float)
    if g.ndim==2:g=g[None]
    n,o,d=g.shape;target=-g[:,0]@reference;scale=np.maximum(abs(target),1e-10)
    signed=-g[:,0]*np.sign(target)[:,None]/scale[:,None]
    control=(g[:,1:]/scale[:,None,None]).reshape(-1,d)
    matrix=np.concatenate([np.c_[-signed,np.ones(n)],np.c_[control,np.zeros(len(control))],np.c_[-control,np.zeros(len(control))]])
    rhs=np.r_[np.zeros(n),np.full(2*len(control),control_budget)]
    objective=np.r_[np.zeros(d),-1.];bounds=[(-1.,1.)]*d+[(0.,2.)]
    result=linprog(objective,A_ub=matrix,b_ub=rhs,bounds=bounds,method='highs')
    if not result.success:raise RuntimeError(result.message)
    lower=np.array([b[0] for b in bounds]);upper=np.array([b[1] for b in bounds])
    dual=float(rhs@result.ineqlin.marginals+lower@result.lower.marginals+upper@result.upper.marginals)
    violation=float(max(0.,np.max(matrix@result.x-rhs),np.max(lower-result.x),np.max(result.x-upper)))
    gap=abs(float(result.fun)-dual)
    if violation>1e-7 or gap>1e-7:raise RuntimeError(f'Invalid LP certificate: {violation=} {gap=}')
    return result.x[:-1],dict(retention=float(result.x[-1]),constraint_violation=violation,duality_gap=gap,tiny_reference_rows=int(np.sum(abs(target)<1e-10)))
