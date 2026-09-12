"""PSD truncation: preserve every exact null direction of the native map."""
import torch
from spectral_norm_quadratic_v1 import compile_norm


def norm_mean(x,program,k):
    z=torch.einsum('nd,hkd->nhk',x,program['readers'][:,:k])
    return (z.square()*program['values'][None,:,:k]).sum(-1)/program['rows']+torch.finfo(torch.float32).eps
