"""Common producer coordinates minimizing normalized stacked-reader residual."""
import torch

def geometry(forms, metric):
    ev,vec=torch.linalg.eigh((metric+metric.T)/2)
    assert bool((ev>ev.max()*1e-12).all())
    root=(vec*ev.sqrt())@vec.T;inverse=(vec*ev.rsqrt())@vec.T
    maps=forms@root
    scales=maps.square().sum((-2,-1)).sqrt()
    normalized=maps/scales[:,None,None]
    gram=torch.einsum('kij,kil->jl',normalized,normalized)
    values,vectors=torch.linalg.eigh((gram+gram.T)/2)
    order=values.argsort(descending=True)
    return dict(root=root,inverse=inverse,values=values[order],vectors=vectors[:,order],
                normalized_maps=normalized,condition=float(ev.max()/ev.min()))

def interface(geometry, directions):
    return dict(read=directions.T@geometry['inverse'],write=geometry['root']@directions)

def apply(program, producer, bias):
    return (producer-bias)@program['read'].T@program['write'].T+bias

def objective(geometry, directions):
    maps=geometry['normalized_maps'];error=maps-(maps@directions)@directions.T
    return float(error.square().sum())
