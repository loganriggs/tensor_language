"""Pull streamed feature gradients back through fixed-scale quadratic normalization."""
import torch
from shared_quadratic_bank import normalize_bank
from shared_mixed_gradient import streamed_gradient


def backward(parameters,width,terms,pairs,C,teacher,transformed,location,projection,S,mu,lam,normalizer=1.,chunk=8):
    normalized=normalize_bank(*[p.reshape(width,terms,-1) for p in parameters])
    leaves=[t.detach().requires_grad_() for t in normalized]
    value=streamed_gradient(*leaves,pairs,C,teacher,transformed,location,projection,S,mu,lam,chunk)
    torch.autograd.backward(normalized,[t.grad/normalizer for t in leaves])
    return value
