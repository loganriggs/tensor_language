"""Separate input directions for each quadratic, then signed outer squares.
Actual normalizedMLP16 source and nativeMLP17 inputdenominator remain external.
"""
def run(program,x,denominator):
    import torch
    reads=torch.einsum('...d,mkdr->...mkr',x,program['input_readers'])
    inner=(reads.square()*program['inner_weights']).sum(-1)
    scalar=(inner.square()*program['outer_weights']).sum(-1)
    return (scalar@program['output_writers'].T)/denominator[...,None]
