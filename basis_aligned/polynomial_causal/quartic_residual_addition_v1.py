"""Exact improvement from adding each feature to an existing coefficient span."""
import torch

def scores(k, c, retained):
    scale=k.diagonal().sqrt()
    g=k/scale[:,None]/scale[None,:];target=c/scale[:,None]
    sub=g[retained][:,retained];cross=g[:,retained]
    fit=torch.linalg.solve(sub,target[retained])
    residual=target-cross@fit
    variance=g.diagonal()-(cross*torch.linalg.solve(sub,cross.T).T).sum(-1)
    gains=residual.square().sum(-1)/variance.clamp_min(1e-30)
    gains[variance<=1e-12]=-torch.inf;gains[retained]=-torch.inf
    return gains,variance,float((fit*target[retained]).sum())
