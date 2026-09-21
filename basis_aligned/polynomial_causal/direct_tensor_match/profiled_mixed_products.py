"""Variable projection: solve output weights, optimize product directions.
The tiny ridge is part of the objective so detached-readout gradients obey the
envelope theorem. Inputs are normalized separately to control scaling gauges.
"""
import torch
from shared_quadratic_products import mixed_coefficient_loss

def solve_weights(teacher,left,right,ridge=1e-10):
    gram=.5*((left.T@left)*(right.T@right)+(left.T@right)*(right.T@left))
    rhs=torch.einsum('ir,oij,jr->ro',left,teacher,right)
    weights=torch.linalg.solve(gram+ridge*torch.eye(len(gram),device=gram.device,dtype=gram.dtype),rhs).T
    return weights

def profiled_loss(teacher,left,right,ridge=1e-10,detach_weights=True):
    if detach_weights:
        with torch.no_grad():weights=solve_weights(teacher,left,right,ridge)
    else:weights=solve_weights(teacher,left,right,ridge)
    loss=mixed_coefficient_loss(teacher,left,right,weights)+ridge*weights.square().sum()/teacher.square().sum()
    return loss,weights
