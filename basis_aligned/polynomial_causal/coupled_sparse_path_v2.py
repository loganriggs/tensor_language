"""Same fixed-support path objective, calculate active coefficients only.
Original full-column helper remains the support-selection authority.
"""
import time
import numpy as np
import torch
import pymanopt
from pymanopt import Problem
from pymanopt.manifolds import Stiefel
from pymanopt.optimizers.line_search import BackTrackingLineSearcher
from orthogonal_multioutput_pymanopt_v2 import ScalarLoggedCG
from sparse_path_program_v1 import edges


def selected_coefficients(left,right,writer,bank,mode,support):
    assignments=[0,1] if mode=='joint' else [0,0,1,1]
    a=torch.cat([left[port]@bank[i] for i,port in enumerate(assignments)],1)
    b=torch.cat([right[port]@bank[i] for i,port in enumerate(assignments)],1)
    pair=edges(dict(bank=bank,mode=mode,support=support))
    denominator=torch.where(pair[0]==pair[1],2.,2**.5).to(bank)
    features=(a[:,pair[0]]*b[:,pair[1]]+b[:,pair[0]]*a[:,pair[1]])/denominator
    return writer@features


def evaluate(left,right,writer,bank,mode,total,keep,support):
    assert support is not None and len(support)==keep
    c=selected_coefficients(left,right,writer,bank,mode,support)
    capture=c.square().sum()/total
    return -capture,dict(capture=capture,selected_coefficients=c,support=support)


def fit_fixed(left,right,writer,bank,mode,total,support,seconds=60,tolerance=1e-6):
    """Standard manifoldCG with a fixed selected-edge support; no sparse stationarity claim."""
    manifold=Stiefel(bank.shape[1],bank.shape[2],k=len(bank));cache={};calls=[0]
    def calc(point):
        if 'x' not in cache or not np.array_equal(point,cache['x']):
            x=torch.as_tensor(point,dtype=bank.dtype,device=bank.device).requires_grad_()
            loss,detail=evaluate(left,right,writer,x,mode,total,len(support),support)
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
        relative_stationarity=station/max(capture,1e-15),converged=station<=tolerance,
        iterations=int(result.iterations),evaluations=calls[0],seconds=time.perf_counter()-start,
        stopping_criterion=result.stopping_criterion,maximum_increase=solver.maximum_increase,history=solver.scalar_history)
