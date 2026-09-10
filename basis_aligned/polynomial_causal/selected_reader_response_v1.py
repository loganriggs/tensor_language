"""A conditional quadratic response state for repeated edits along one direction."""
import torch
from bilinear_scalar_consumer_v1 import context_reader


def compile_response(left,right,down,e,readers):
    k=torch.stack([context_reader(left,right,down,e,v) for v in readers])
    a=(readers@down)@((left@e)*(right@e))
    return {'K':k,'a':a}


def initialize(program,u):
    return u@program['K'].T


def step(program,state,delta):
    delta=torch.as_tensor(delta,dtype=state.dtype,device=state.device)
    if delta.ndim==state.ndim-1:delta=delta.unsqueeze(-1)
    return delta*state+delta.square()*program['a'],state+2*delta*program['a']
