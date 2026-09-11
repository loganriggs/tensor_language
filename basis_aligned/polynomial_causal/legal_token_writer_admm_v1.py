"""Convex fixed-function native writers with concentrated centered token effects.

P has orthonormal columns; actual writer=P Z. Penalize F Z, F=P-mean(P).
Solve .5 tr(Z G Z.T)-<Z,C>+lambda||F Z||1. No arbitrary sparse output writer.
"""
import json,time
from pathlib import Path
import torch
from quadratic_token_dictionary_v1 import soft


def sylvester(rhs,values,vectors,mean,rows,rho):
    """Z G + rho (I-rows*mean mean.T) Z=rhs, using its rank-one left correction."""
    projected=rhs@vectors
    base=projected/(values+rho)[None,:]
    magnitude=mean.norm();alpha=rows*magnitude.square()
    if float(magnitude)>0:
        direction=mean/magnitude
        correction=(direction@projected)*(1/(values+rho*(1-alpha))-1/(values+rho))
        base+=direction[:,None]*correction[None,:]
    return base@vectors.T


def diagnostics(p,g,c,z,a,dual,penalty,rho):
    f=p-p.mean(0);actual=f@z
    station=z@g-c+f.T@(rho*dual)
    primal=float((actual-a).norm()/max(float(actual.norm()),float(a.norm()),1e-15))
    subgradient=float((a-soft(a+rho*dual,penalty)).norm()/a.norm().clamp_min(1e-15))
    return dict(objective=float(.5*((z@g)*z).sum()-(z*c).sum()+penalty*actual.abs().sum()),
                relative_feasibility=primal,relative_stationarity=float(station.norm()/c.norm().clamp_min(1e-15)),
                relative_l1_subgradient=subgradient,
                actual_writer_nonzeros=int((actual!=0).sum()),auxiliary_nonzeros=int((a!=0).sum()))


def solve(p,g,c,penalty,rho=1.,max_steps=2000,tolerance=1e-6,seconds=None):
    f=p-p.mean(0);mean=p.mean(0)
    values,vectors=torch.linalg.eigh((g+g.T)/2)
    assert float(values[0])>0,'Version1 requires independent fixed atoms; no silent ridge.'
    z=(c@vectors/values[None,:])@vectors.T
    a=f@z;dual=torch.zeros_like(a);start=time.perf_counter();history=[];converged=False
    for step in range(1,max_steps+1):
        rhs=c+rho*(f.T@(a-dual))
        z=sylvester(rhs,values,vectors,mean,len(p),rho)
        actual=f@z
        a=soft(actual+dual,penalty/rho)
        dual+=actual-a
        if step==1 or step%10==0 or step==max_steps:
            stats=diagnostics(p,g,c,z,a,dual,penalty,rho)
            history.append(dict(step=step,elapsed=time.perf_counter()-start,**stats))
            if max(stats['relative_feasibility'],stats['relative_stationarity'],stats['relative_l1_subgradient'])<=tolerance:
                converged=True;break
        if seconds is not None and time.perf_counter()-start>=seconds:break
    final=diagnostics(p,g,c,z,a,dual,penalty,rho)
    return z,dict(converged=converged,steps=step,seconds=time.perf_counter()-start,final=final,history=history)


def control():
    import numpy as np
    from scipy.optimize import minimize,LinearConstraint
    torch.set_num_threads(2);torch.set_default_dtype(torch.float64);torch.manual_seed(1049)
    raw=torch.randn(13,4);p=torch.linalg.qr(raw).Q
    atoms=torch.randn(3,6);g=atoms@atoms.T;c=torch.randn(4,3);penalty=.12
    rhs=torch.randn(4,3);ev,vec=torch.linalg.eigh(g);f=p-p.mean(0)
    z=sylvester(rhs,ev,vec,p.mean(0),len(p),.7)
    equation_error=float((z@g+.7*f.T@f@z-rhs).norm()/rhs.norm())
    z,report=solve(p,g,c,penalty,1.,10000,1e-9)
    count=c.numel();mapping=np.kron(f.numpy(),np.eye(3));eye=np.eye(mapping.shape[0])
    matrix=np.concatenate([np.concatenate([mapping,-eye],axis=1),np.concatenate([-mapping,-eye],axis=1)],axis=0)
    def objective(flat):
        zz=torch.from_numpy(flat[:count]).reshape(c.shape);t=flat[count:]
        value=.5*((zz@g)*zz).sum()-(zz*c).sum()+penalty*t.sum()
        gradient=np.concatenate([(zz@g-c).flatten().numpy(),np.full(len(t),penalty)])
        return float(value),gradient
    initial=np.zeros(count+len(eye))
    reference=minimize(objective,initial,jac=True,method='SLSQP',
        constraints=LinearConstraint(matrix,-np.inf,0),options={'ftol':1e-12,'maxiter':2000})
    objective_error=abs(reference.fun-report['final']['objective'])/max(1.,abs(reference.fun))
    result=dict(instrument_passed=equation_error<1e-10 and report['converged'] and reference.success and objective_error<1e-8,
                sylvester_equation_error=equation_error,independent_epigraph_qp_objective_error=objective_error,
                reference_success=bool(reference.success),admm=report,
                scope='Convex fixed-atom writer update only. Exact native output parameterization, centered L1 penalty, independent QP reference. No joint product convergence theorem.')
    Path(__file__).with_name('LEGAL_TOKEN_WRITER_ADMM_V1_CONTROL.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='admm'},indent=2));print(report['final']);assert result['instrument_passed']


if __name__=='__main__':control()
