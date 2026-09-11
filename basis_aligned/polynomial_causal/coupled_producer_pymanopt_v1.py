"""Pymanopt Grassmann adapter for the checked coupled objective; CPU matrices."""
import json,time
from pathlib import Path
import numpy as np
import torch
import pymanopt
from pymanopt import Problem
from pymanopt.manifolds import Grassmann
from pymanopt.optimizers import ConjugateGradient
from pymanopt.optimizers.line_search import BackTrackingLineSearcher
from coupled_producer_routing_objective_v1 import objective


def fit(initial,g,m,s,sharing_ceiling,influence_ceiling,weight=.5,seconds=60,tolerance=1e-7,max_iterations=5000):
    """Normalize BOTH objectives by their within-key-space maxima before mixing."""
    rank=initial.shape[1];manifold=Grassmann(*initial.shape);scaled=m/sharing_ceiling
    def evaluate(e):
        return objective(torch.from_numpy(e),g,scaled,rank,s,influence_ceiling,weight)
    @pymanopt.function.numpy(manifold)
    def cost(e):return -float(evaluate(e)[0])
    @pymanopt.function.numpy(manifold)
    def gradient(e):return -evaluate(e)[1].numpy()
    problem=Problem(manifold,cost,euclidean_gradient=gradient)
    solver=ConjugateGradient(beta_rule='PolakRibiere',min_gradient_norm=tolerance,
        min_step_size=1e-18,max_iterations=max_iterations,max_time=seconds,
        max_cost_evaluations=50000,verbosity=0,log_verbosity=0,
        line_searcher=BackTrackingLineSearcher(max_iterations=50))
    started=time.perf_counter();result=solver.run(problem,initial_point=initial.numpy())
    e=torch.from_numpy(result.point.copy());value,grad=evaluate(result.point)
    station=float((grad-e@(e.T@grad)).norm())
    report=dict(converged=station<=tolerance,score=float(value),tangent_stationarity=station,
        orthogonality_error=float((e.T@e-torch.eye(rank,dtype=e.dtype)).abs().max()),
        iterations=int(result.iterations),stopping_criterion=result.stopping_criterion,
        seconds=time.perf_counter()-started,pymanopt_version=pymanopt.__version__,
        initial_score=-cost(initial.numpy()))
    return e,report


def control():
    torch.set_num_threads(2);torch.set_default_dtype(torch.float64);torch.manual_seed(1399)
    d,r=11,3;g=torch.eye(d);a=torch.linalg.qr(torch.randn(d,r)).Q;m=a@a.T
    z=torch.randn(d,d);s=z@z.T;s/=s.trace();mu=float(torch.linalg.eigvalsh(s)[-r:].sum())
    combined=.5*m/r+.5*s/mu;eig,vec=torch.linalg.eigh(combined);optimum=float(eig[-r:].sum())
    fits=[];errors=[]
    for _ in range(2):
        initial=torch.linalg.qr(torch.randn(d,r)).Q
        e,report=fit(initial,g,m,s,1.,mu,seconds=10)
        errors.append(abs(report['score']-optimum));fits.append(report)
    result=dict(instrument_passed=all(x['converged'] for x in fits) and max(errors)<1e-10,
        maximum_known_optimum_error=max(errors),fits=fits,
        scope='Standard solver reaches known global spectral optimum on a quadratic control. Native nonlinear trace-quotient problem has no global guarantee.')
    with Path(__file__).with_name('COUPLED_PRODUCER_PYMANOPT_V1_CONTROL.json').open('x') as f:
        json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2));assert result['instrument_passed']


if __name__=='__main__':control()
