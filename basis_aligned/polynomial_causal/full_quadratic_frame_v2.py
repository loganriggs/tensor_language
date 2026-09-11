"""Standard exact-curvature Grassmann solver after V1 CG recovery failure."""
import numpy as np
import torch
import pymanopt
from pymanopt import Problem
from pymanopt.manifolds import Grassmann
from pymanopt.optimizers import TrustRegions
from full_quadratic_frame_v1 import cores_for
from orthogonal_multioutput_pymanopt_v2 import evaluate
from block_variable_projection_curvature_v1 import hessian_vector


def fit(objective,initial,seconds=120,tolerance=1e-7):
    groups,dim,rank=initial.shape
    manifold=Grassmann(dim,rank,k=groups)
    cores=cores_for(initial);count=rank*(rank+1)//2
    def tensor(x):
        return torch.from_numpy(np.array(x,copy=True)).to(initial.device).reshape(groups,dim,rank)
    @pymanopt.function.numpy(manifold)
    def cost(x):return evaluate(objective,tensor(x),cores,count)[0]
    @pymanopt.function.numpy(manifold)
    def gradient(x):return evaluate(objective,tensor(x),cores,count)[1][0].cpu().numpy().reshape(x.shape)
    @pymanopt.function.numpy(manifold)
    def hessian(x,direction):
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
                      iterations=result.iterations,stop=result.stopping_criterion)
