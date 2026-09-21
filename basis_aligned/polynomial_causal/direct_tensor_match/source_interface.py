"""Standalone continuation-write executor at native z,h input interfaces.

z: normalized MLP16 input; h: pre-normalization last-MLP input.
Parameters may be cast/moved together with input tensors. The trained model uses
float32 RMS epsilon, retained explicitly even for a float64 replay.
"""
import torch

def source_read(z,program,prefix):
    projected=z@program[prefix+'_reader']
    return (program[prefix+'_bias']+z@program[prefix+'_linear']
            +(projected.square()*program[prefix+'_eigenvalues']).sum(-1))

def residual_write(z,h,program):
    if 'shared_reader' in program:
        # Evaluate this shared intermediate once for both quadratic consumers.
        t=z@program['shared_reader']
        values=[]
        for k in ['a','b']:
            values.append(program[k+'_bias']+z@program[k+'_linear']
                          +((t@program[k+'_inner_reader']).square()*program[k+'_eigenvalues']).sum(-1))
        qa,qb=values
    else:
        qa=source_read(z,program,'a');qb=source_read(z,program,'b')
    scale=(h.square().mean(-1)+torch.finfo(torch.float32).eps).sqrt()
    scalar=((h@program['h_reader']-.5*qa)/scale-program['alpha'])*(qb/scale-program['beta'])
    return scalar[...,None]*program['residual_writer']
