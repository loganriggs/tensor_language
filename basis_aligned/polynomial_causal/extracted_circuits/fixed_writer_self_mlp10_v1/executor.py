"""Port-conditioned direct residual-writer square; no checkpoint dependency."""
import torch

def execute(amplitude,normalizer,writer):
    """amplitude and positive whole-MLP10 normalizer end in singleton axes."""
    if torch.any(normalizer<=0):raise ValueError('normalizer must be positive')
    return amplitude.square()/normalizer*writer.to(amplitude)

def joint_local(a,b,rho_a,rho_b,rho_parent,writer):
    """Local three-branch difference only; nonlinear suffix remains external."""
    coefficient=(a+b).square()/rho_parent-a.square()/rho_a-b.square()/rho_b
    return coefficient*writer.to(a)
