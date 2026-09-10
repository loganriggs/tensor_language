"""Gauge-aware stationarity of normalized signed-square readers."""
import torch

def stationarity(model,captured,original_norms):
    a=model.a.detach();g=model.a.grad.detach();current_norms=a.norm(dim=1,keepdim=True)
    canonical_gradient=g*current_norms
    original_gradient=canonical_gradient/original_norms
    return dict(canonical_relative_stationarity=float(canonical_gradient.norm()*len(a)**.5/captured),original_gauge_relative_stationarity=float(original_gradient.norm()*original_norms.norm()/captured),canonical_gradient_max=float(canonical_gradient.abs().max()),original_gauge_gradient_max=float(original_gradient.abs().max()))
