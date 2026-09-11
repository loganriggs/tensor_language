"""Select a scalar quadratic in a fixed function span for one-product simplicity.

For orthonormal symmetric H_i and unit c, Q=sum c_i H_i has unit Frobenius norm.
Its best single real-product capture is lambda_max(Q)_+^2+lambda_min(Q)_-^2.
Optimize c with existing Pymanopt sphere CG; no global optimality claim.
"""
import json,time
from pathlib import Path
import numpy as np
import torch
import pymanopt
from pymanopt import Problem
from pymanopt.manifolds import Sphere
from pymanopt.optimizers import ConjugateGradient
from pymanopt.optimizers.line_search import BackTrackingLineSearcher


def value_gradient(c,forms):
    q=torch.einsum('a,aij->ij',c,forms)
    ev,v=torch.linalg.eigh(q)
    pos=ev[-1].clamp_min(0);neg=ev[0].clamp_max(0)
    vp,vn=v[:,-1],v[:,0]
    gp=(torch.matmul(forms,vp)*vp).sum(-1)
    gn=(torch.matmul(forms,vn)*vn).sum(-1)
    grad=2*(pos*gp+neg*gn)
    gap=torch.minimum(ev[-1]-ev[-2],ev[1]-ev[0])
    return pos.square()+neg.square(),grad,gap


def fit(forms,initial,seconds=15,tolerance=1e-7):
    manifold=Sphere(len(initial));cache={};evaluations=0
    def evaluate(c):
        nonlocal evaluations
        if 'c' not in cache or not np.array_equal(cache['c'],c):
            x=torch.as_tensor(c,dtype=forms.dtype,device=forms.device)
            value,grad,gap=value_gradient(x,forms)
            cache.update(c=c.copy(),value=float(value),grad=grad.cpu().numpy(),gap=float(gap));evaluations+=1
        return cache
    @pymanopt.function.numpy(manifold)
    def cost(c):return -evaluate(c)['value']
    @pymanopt.function.numpy(manifold)
    def gradient(c):return -evaluate(c)['grad']
    problem=Problem(manifold,cost,euclidean_gradient=gradient)
    solver=ConjugateGradient(beta_rule='PolakRibiere',min_gradient_norm=tolerance,
        min_step_size=1e-18,max_iterations=1000,max_time=seconds,
        max_cost_evaluations=10000,verbosity=0,log_verbosity=0,
        line_searcher=BackTrackingLineSearcher(max_iterations=50))
    start=time.perf_counter();initial=initial.detach().cpu().numpy();initial/=np.linalg.norm(initial)
    initial_value=-cost(initial);result=solver.run(problem,initial_point=initial)
    c=result.point;end=evaluate(c);grad=end['grad'];station=float(np.linalg.norm(grad-c*np.dot(c,grad)))
    return torch.from_numpy(c.copy()),dict(score=end['value'],initial_score=initial_value,
        tangent_stationarity=station,converged=station<=tolerance,
        extreme_eigenvalue_gap=end['gap'],iterations=int(result.iterations),
        evaluations=evaluations,seconds=time.perf_counter()-start,
        stopping_criterion=result.stopping_criterion,pymanopt_version=pymanopt.__version__)


def control():
    torch.set_num_threads(2);torch.set_default_dtype(torch.float64);torch.manual_seed(1451)
    h=torch.randn(4,7,7);h=(h+h.transpose(-1,-2))/2
    h=torch.linalg.qr(h.flatten(1).T).Q.T.reshape(4,7,7)
    c=torch.randn(4);c/=c.norm();value,grad,_=value_gradient(c,h)
    delta=1e-5;finite=[]
    for i in range(4):
        step=torch.zeros(4);step[i]=delta
        finite.append((value_gradient(c+step,h)[0]-value_gradient(c-step,h)[0])/(2*delta))
    error=float((grad-torch.stack(finite)).abs().max())
    s=(h@h).sum(0);q=torch.einsum('a,aij->ij',c,h)
    bound_error=max(0.,-float(torch.linalg.eigvalsh(s-q@q)[0]))
    # A known real product mixed with a higher-rank disjoint noise function.
    planted=torch.zeros(2,7,7);planted[0,0,0]=2**-.5;planted[0,1,1]=-2**-.5
    planted[1,2:,2:]=torch.eye(5)/5**.5
    fits=[]
    for initial in [torch.tensor([.8,.6]),torch.tensor([-.8,.6])]:
        _,row=fit(planted,initial,seconds=10);fits.append(row)
    basis=torch.linalg.qr(torch.randn(4,4)).Q
    rotated=torch.einsum('ab,bij->aij',basis,h)
    gauge_error=abs(float(value_gradient(basis@c,rotated)[0]-value))
    result=dict(instrument_passed=max(error,bound_error,gauge_error)<1e-8 and all(x['converged'] and abs(x['score']-1)<1e-10 for x in fits),
        finite_gradient_error=error,span_bound_error=bound_error,basis_rotation_error=gauge_error,planted_fits=fits,
        scope='Analytic derivative, known product recovery, and basis-invariant scalar objective. Generic native objective is nonsmooth at tied extreme eigenvalues; gaps must be reported.')
    with Path(__file__).with_name('SIMPLE_PRODUCT_IN_SPAN_V1_CONTROL.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2));assert result['instrument_passed']

if __name__=='__main__':control()
