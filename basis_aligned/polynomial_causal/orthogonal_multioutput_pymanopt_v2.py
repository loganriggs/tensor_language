"""Standard manifold adapter for the existing overlapping block objective.

Within-block basis columns are orthonormal; different blocks may overlap.
Each packed symmetric core has unit norm. No changed objective or penalty.
"""
import json,math,time
from pathlib import Path
import numpy as np
import torch
import pymanopt
from pymanopt import Problem
from pymanopt.manifolds import Product,Stiefel,Oblique
from pymanopt.optimizers import ConjugateGradient
from pymanopt.optimizers.line_search import BackTrackingLineSearcher
from multioutput_quadratic_blocks_v1 import MultioutputWeightObjective


class ScalarLoggedCG(ConjugateGradient):
    """Keep scalar diagnostics every five steps, never retain large point arrays."""
    def _add_log_entry(self, *, iteration, point, cost, **kwargs):
        if not hasattr(self, 'scalar_history'):
            self.scalar_history=[];self.maximum_increase=0.;self.last_value=float(cost)
        self.maximum_increase=max(self.maximum_increase,float(cost)-self.last_value)
        self.last_value=float(cost)
        if (iteration-1)%5==0:
            row=dict(iteration=iteration-1,cost=float(cost),gradient_norm=float(kwargs['gradient_norm']))
            self.scalar_history.append(row)
            if (iteration-1)%500==0:print(json.dumps(row),flush=True)


class View:
    def __init__(self,bank,packed,outputs):
        self.bank=bank;self.packed=packed;self.groups=bank.shape[0];self.rank=bank.shape[-1];self.outputs=outputs
    def components(self):
        v=self.packed.T.reshape(self.groups,self.outputs,-1)
        i,j=torch.triu_indices(self.rank,self.rank,device=v.device)
        scale=torch.where(i==j,torch.ones_like(i,dtype=v.dtype),torch.full_like(i,1/math.sqrt(2),dtype=v.dtype))
        c=v.new_zeros(self.groups,self.outputs,self.rank,self.rank)
        c[...,i,j]=v*scale;c[...,j,i]=v*scale
        return self.bank.transpose(-1,-2),c


def manifold_for(bank,packed):
    groups,dim,rank=bank.shape
    return Product([Stiefel(dim,rank,k=groups),Oblique(*packed.shape)])


def evaluate(objective,bank,packed,outputs):
    bank=bank.detach().requires_grad_();packed=packed.detach().requires_grad_()
    loss,residual,energy,_,_,_=objective.terms(View(bank,packed,outputs))
    grads=torch.autograd.grad(loss,(bank,packed))
    return float(loss.detach()),[g.detach() for g in grads],dict(residual=float(residual.detach()),component_energy=float(energy.detach()))


def fit(objective,bank,packed,outputs,seconds=120,tolerance=1e-5):
    manifold=manifold_for(bank,packed);cache={};evaluations=0
    def calc(b,c):
        nonlocal evaluations
        if 'b' not in cache or not np.array_equal(cache['b'],b) or not np.array_equal(cache['c'],c):
            bt=torch.as_tensor(b,device=bank.device,dtype=bank.dtype);ct=torch.as_tensor(c,device=packed.device,dtype=packed.dtype)
            value,grad,details=evaluate(objective,bt,ct,outputs)
            cache.update(b=b.copy(),c=c.copy(),value=value,grad=[g.cpu().numpy() for g in grad],details=details);evaluations+=1
        return cache
    @pymanopt.function.numpy(manifold)
    def cost(b,c):return calc(b,c)['value']
    @pymanopt.function.numpy(manifold)
    def gradient(b,c):return calc(b,c)['grad']
    problem=Problem(manifold,cost,euclidean_gradient=gradient)
    solver=ScalarLoggedCG(beta_rule='PolakRibiere',min_gradient_norm=tolerance,min_step_size=1e-18,
        max_iterations=10000,max_time=seconds,max_cost_evaluations=100000,verbosity=0,
        line_searcher=BackTrackingLineSearcher(max_iterations=50))
    initial=[bank.detach().cpu().numpy(),packed.detach().cpu().numpy()];start=time.perf_counter()
    result=solver.run(problem,initial_point=initial);row=calc(*result.point)
    tangent=manifold.euclidean_to_riemannian_gradient(result.point,row['grad'])
    station=float(manifold.norm(result.point,tangent))
    return [torch.from_numpy(x.copy()) for x in result.point],dict(loss=row['value'],**row['details'],
        tangent_stationarity=station,converged=station<=tolerance,seconds=time.perf_counter()-start,
        iterations=int(result.iterations),evaluations=evaluations,stopping_criterion=result.stopping_criterion,
        scalar_history=solver.scalar_history,maximum_objective_increase=solver.maximum_increase)

