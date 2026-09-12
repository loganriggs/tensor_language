"""Conditional exact response of residual+bilinear MLP to one fixed write removal."""
import torch

def execute(z, native_biasfree_mlp, amplitude, program):
    """Return delta x8 for z -> z-a*d.

    z and native_biasfree_mlp: [...,1152], same floating dtype/device.
    amplitude: [...,1]. Background MLP output is on RMS(z), excludes Down_bias.
    Program contains direction [1152] and mixed_map [1152,1152].
    Full native background generation remains external; no layer9 re-entry here.
    """
    d=program['direction'].to(z)
    J=program['mixed_map'].to(z)
    eps=torch.finfo(torch.float32).eps
    rho=z.square().mean(-1,keepdim=True)+eps
    rhop=(z-amplitude*d).square().mean(-1,keepdim=True)+eps
    return (-amplitude*d+(rho/rhop-1)*native_biasfree_mlp
            -amplitude/rhop*((z-amplitude*d/2)@J.T))
