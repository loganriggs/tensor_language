"""Exact sparse degree4 coefficients on an orthonormal input basis; no dense tensor."""
import itertools
import math
import time
import numpy as np
import torch
import pymanopt
from pymanopt import Problem
from pymanopt.manifolds import Stiefel
from pymanopt.optimizers.line_search import BackTrackingLineSearcher
from orthogonal_multioutput_pymanopt_v2 import ScalarLoggedCG


def indices(rank,device=None):
    terms=list(itertools.combinations_with_replacement(range(rank),4))
    multiplicity=[24/math.prod(math.factorial(t.count(i)) for i in set(t)) for t in terms]
    return torch.tensor(terms,device=device).T,torch.tensor(multiplicity,device=device,dtype=torch.float64).sqrt()


def coefficients(weights,bank,terms,multiplicity_root,scale=1.):
    l0,r0,d0,l1,r1,d1=weights;rank=bank.shape[1]
    pair=torch.triu_indices(rank,rank,device=bank.device)
    lookup=torch.empty((rank,rank),device=bank.device,dtype=torch.long)
    lookup[pair[0],pair[1]]=torch.arange(pair.shape[1],device=bank.device)
    lookup[pair[1],pair[0]]=torch.arange(pair.shape[1],device=bank.device)
    a=bank.T@l0.T;b=bank.T@r0.T
    q=((a[pair[0]]*b[pair[1]]+b[pair[0]]*a[pair[1]])/2)@d0.T*scale
    left=q@l1.T;right=q@r1.T
    i,j,k,l=terms
    p=torch.stack([lookup[i,j],lookup[i,k],lookup[i,l]])
    s=torch.stack([lookup[k,l],lookup[j,l],lookup[j,k]])
    products=(left[p]*right[s]+right[p]*left[s]).sum(0)/6
    return ((products@d1.T)*multiplicity_root[:,None]).T


def fit_fixed(weights,bank,terms,multiplicity_root,total,scale=1.,seconds=60,tolerance=1e-7):
    manifold=Stiefel(*bank.shape);cache={};calls=[0]
    def calc(point):
        if 'x' not in cache or not np.array_equal(point,cache['x']):
            x=torch.as_tensor(point,dtype=bank.dtype,device=bank.device).requires_grad_()
            c=coefficients(weights,x,terms,multiplicity_root,scale);loss=-c.square().sum()/total
            grad=torch.autograd.grad(loss,x)[0]
            cache.update(x=point.copy(),loss=float(loss.detach()),grad=grad.detach().cpu().numpy());calls[0]+=1
        return cache
    @pymanopt.function.numpy(manifold)
    def cost(x):return calc(x)['loss']
    @pymanopt.function.numpy(manifold)
    def gradient(x):return calc(x)['grad']
    problem=Problem(manifold,cost,euclidean_gradient=gradient)
    solver=ScalarLoggedCG(beta_rule='PolakRibiere',min_gradient_norm=tolerance,min_step_size=1e-18,max_iterations=10000,
        max_time=seconds,max_cost_evaluations=100000,verbosity=0,line_searcher=BackTrackingLineSearcher(max_iterations=40))
    start=time.perf_counter();result=solver.run(problem,initial_point=bank.detach().cpu().numpy())
    data=calc(result.point);tangent=manifold.euclidean_to_riemannian_gradient(result.point,data['grad'])
    station=float(manifold.norm(result.point,tangent));capture=-data['loss']
    return torch.as_tensor(result.point.copy(),dtype=bank.dtype,device=bank.device),dict(capture=capture,tangent_norm=station,
        relative_stationarity=station/max(capture,1e-15),converged=station<=tolerance,iterations=int(result.iterations),evaluations=calls[0],
        seconds=time.perf_counter()-start,stopping_criterion=result.stopping_criterion,maximum_increase=solver.maximum_increase,history=solver.scalar_history)


def execute(bank,terms,multiplicity_root,writer,x):
    reads=x@bank
    amplitude=reads[...,terms].prod(-2)*multiplicity_root
    return amplitude@writer.T
