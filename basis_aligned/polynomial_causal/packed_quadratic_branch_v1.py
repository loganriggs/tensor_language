"""Execute a packed shared-parent quadratic branch program at its declared ports.

Needs normalized x16 and the actual squared MLP17 input RMS denominator.
Does not load model weights or produce those upstream states. Arithmetic uses
x.dtype; FP32 stored coefficients can be promoted by passing FP64 inputs.
"""
import torch

def execute(program, x, denominator, batch_size=8):
    dimension=program['dimension']
    assert x.ndim==2 and x.shape[1]==dimension and denominator.shape==(len(x),)
    assert bool((denominator>0).all()) and batch_size>0
    packed=program['upper_coefficients'].to(device=x.device,dtype=x.dtype)
    writers=program['writers'].to(device=x.device,dtype=x.dtype)
    assert packed.shape==(len(program['branches'])+1,dimension*(dimension+1)//2)
    assert writers.shape==(dimension,len(program['branches']))
    ij=torch.triu_indices(dimension,dimension,device=x.device);values=[]
    for start in range(0,len(x),batch_size):
        xx=x[start:start+batch_size]
        values.append((xx[:,ij[0]]*xx[:,ij[1]])@packed.T)
    q=torch.cat(values)
    den=denominator.to(device=x.device,dtype=x.dtype)
    return {branch:q[:,0,None]*q[:,j+1,None]*writers[:,j][None,:]/den[:,None]
            for j,branch in enumerate(program['branches'])}
