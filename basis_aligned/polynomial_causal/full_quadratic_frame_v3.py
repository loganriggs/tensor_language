"""V2 exact-curvature solver with same-point memoization and scalar logging."""
import json
import time
import numpy as np
import torch
import pymanopt
from pymanopt import Problem
from pymanopt.manifolds import Grassmann
from pymanopt.optimizers import TrustRegions
from full_quadratic_frame_v1 import cores_for
from orthogonal_multioutput_pymanopt_v2 import evaluate
from block_variable_projection_curvature_v1 import hessian_vector


def fit(objective,initial,seconds=300,tolerance=1e-7):
    start=time.perf_counter();groups,dim,rank=initial.shape
    manifold=Grassmann(dim,rank,k=groups)
    cores=cores_for(initial);count=rank*(rank+1)//2
    cache={};state=dict(evaluations=0,hessian_vectors=0,accepted_points=0,history=[])
    def tensor(x):
        return torch.from_numpy(np.array(x,copy=True)).to(initial.device).reshape(groups,dim,rank)
    def value(x):
        if 'point' not in cache or not np.array_equal(cache['point'],x):
            loss,gradient,details=evaluate(objective,tensor(x),cores,count)
            cache.update(point=x.copy(),loss=loss,gradient=gradient[0].cpu().numpy().reshape(x.shape),details=details)
            state['evaluations']+=1
        return cache
    @pymanopt.function.numpy(manifold)
    def cost(x):return value(x)['loss']
    @pymanopt.function.numpy(manifold)
    def gradient(x):
        current=value(x)
        if 'accepted' not in cache or not np.array_equal(cache['accepted'],x):
            cache['accepted']=x.copy();state['accepted_points']+=1
            if (state['accepted_points']-1)%25==0:
                row=dict(accepted=state['accepted_points']-1,loss=current['loss'],
                         residual=current['details']['residual'],seconds=time.perf_counter()-start)
                state['history'].append(row);print(json.dumps(row),flush=True)
        return current['gradient'].copy()
    @pymanopt.function.numpy(manifold)
    def hessian(x,direction):
        state['hessian_vectors']+=1
        hv=hessian_vector(objective,tensor(x),cores,count,[tensor(direction),torch.zeros_like(cores)])[0]
        return hv.cpu().numpy().reshape(x.shape)
    point=initial.cpu().numpy()
    if groups==1:point=point[0]
    result=TrustRegions(max_iterations=1000,max_time=seconds,min_gradient_norm=tolerance,
        verbosity=0).run(Problem(manifold,cost,euclidean_gradient=gradient,
            euclidean_hessian=hessian),initial_point=point,maxinner=50)
    final=tensor(result.point)
    loss,grad,details=evaluate(objective,final,cores,count)
    tangent=grad[0]-final@(final.transpose(-1,-2)@grad[0])
    return final,dict(loss=loss,residual=details['residual'],gradient_norm=float(tangent.norm()),
                      iterations=result.iterations,stop=result.stopping_criterion,
                      seconds=time.perf_counter()-start,**state)
