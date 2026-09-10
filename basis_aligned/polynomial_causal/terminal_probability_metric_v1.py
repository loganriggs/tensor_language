"""Exact terminal score differential and local KL quadratic form."""
import torch

def terminal_scores_and_tangent(h,delta,u,eps):
    radius=(h.square().mean(-1,keepdim=True)+eps).sqrt()
    raw=(h@u.T)/radius
    radial=(h*delta).sum(-1,keepdim=True)/(h.square().sum(-1,keepdim=True)+h.shape[-1]*eps)
    raw_tangent=(delta@u.T)/radius-raw*radial
    t=torch.tanh(raw/30)
    return 30*t,(1-t.square())*raw_tangent

def kl_quadratic(probabilities,score_tangent):
    return .5*((probabilities*score_tangent.square()).sum(-1)-(probabilities*score_tangent).sum(-1).square())
