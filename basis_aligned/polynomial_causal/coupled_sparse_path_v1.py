"""Full-output sparse interaction edges with shared or independent source bases.
Writers include the full unembedding metric. Exact fixed-basis output elimination.
No text; joint uses2 bases, independent uses4, in source coordinates of equal width.
"""
import math
import time
import numpy as np
import torch
import pymanopt
from pymanopt import Problem
from pymanopt.manifolds import Stiefel
from pymanopt.optimizers.line_search import BackTrackingLineSearcher
from orthogonal_multioutput_pymanopt_v2 import ScalarLoggedCG
from sparse_orthogonal_quadratic_core_v1 import orthogonal_core


def coefficients(left,right,writer,bank,mode):
    # left/right [2,native_products,source_width], bank [2 or4,width,rank]
    rank=bank.shape[-1]
    if mode=='joint':
        a=torch.cat([left[i]@bank[i] for i in range(2)],1)
        b=torch.cat([right[i]@bank[i] for i in range(2)],1)
        return orthogonal_core(a,b,writer,torch.eye(2*rank,dtype=bank.dtype,device=bank.device))[0]
    assert mode=='independent' and len(bank)==4
    a0,b0=left[0]@bank[0],right[0]@bank[0]
    a1,b1=left[0]@bank[1],right[0]@bank[1]
    a2,b2=left[1]@bank[2],right[1]@bank[2]
    a3,b3=left[1]@bank[3],right[1]@bank[3]
    eye=torch.eye(rank,dtype=bank.dtype,device=bank.device)
    rr=orthogonal_core(a0,b0,writer,eye)[0]
    aa=orthogonal_core(a3,b3,writer,eye)[0]
    mixed=((a1[:,:,None]*b2[:,None,:]+b1[:,:,None]*a2[:,None,:])/math.sqrt(2)).flatten(1)
    return torch.cat([rr,writer@mixed,aa],1)


def evaluate(left,right,writer,bank,mode,total,keep,support=None):
    c=coefficients(left,right,writer,bank,mode);energy=c.square().sum(0)
    if support is None:support=energy.topk(keep).indices
    capture=energy[support].sum()/total
    return -capture,dict(capture=capture,coefficients=c,support=support,edge_energy=energy)


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
