"""Conditional robust coefficient selection; accepts derivatives, never native labels."""
import numpy as np
from scipy.optimize import minimize

def fit(gradient,hessian,amplitudes,atoms):
    # G[B,O,P], H[B,O,P,P], A[arms,B,P], atoms[K,P,P].
    linear=-np.einsum('bop,abp->abo',gradient,amplitudes)
    quadratic=-.5*np.einsum('abp,bopq,abq->abo',amplitudes,hessian,amplitudes)
    budgets=np.maximum(np.linalg.norm((linear+quadratic)[...,0],axis=1),1e-30)
    D=-.5*np.einsum('abp,kpq,abq->abk',amplitudes,atoms)/budgets[:,None,None]
    batch,outputs=gradient.shape[:2];rank=len(atoms)
    decomposed=[np.linalg.svd(D[:,b],full_matrices=False) for b in range(batch)]
    assert all(len(s)==rank and s[-1]>s[0]*1e-12 for _,s,_ in decomposed)
    U=np.stack([u for u,_,_ in decomposed],axis=1)
    coeff=[];checks=[]
    for o in range(outputs):
        Y=quadratic[...,o]/budgets[:,None]
        def error(v):return np.einsum('abk,bk->ab',U,v[:-1].reshape(batch,rank))-Y
        def constraint(v):return v[-1]-np.linalg.norm(error(v),axis=1)
        def jac(v):
            e=error(v);n=np.maximum(np.linalg.norm(e,axis=1),1e-30)
            return np.column_stack(((-U*(e/n[:,None])[...,None]).reshape(len(amplitudes),-1),np.ones(len(amplitudes))))
        start=np.einsum('abk,ab->bk',U,Y);v=np.r_[start.ravel(),0.];v[-1]=np.linalg.norm(error(v),axis=1).max()+1e-8
        objective_jac=np.r_[np.zeros(len(v)-1),1.]
        result=minimize(lambda v:v[-1],v,jac=lambda v:objective_jac,constraints=[dict(type='ineq',fun=constraint,jac=jac)],method='SLSQP',options=dict(maxiter=500,ftol=1e-11))
        assert result.success and constraint(result.x).min()>-1e-8, result.message
        whitened=result.x[:-1].reshape(batch,rank)
        coeff.append(np.stack([vt.T@(whitened[b]/s) for b,(_,s,vt) in enumerate(decomposed)]))
        checks.append(float(np.linalg.norm(error(result.x),axis=1).max()))
    return np.stack(coeff,axis=1),checks
