"""Existing exact output solve + full folded shared-dictionary gradient.

Envelope gradient holds the solved writer fixed during the derivative.
Requires a sufficiently accurate solve and locally constant retained rank.
"""
import torch
from conditional_writer_spectral_v1 import solve
from folded_sparse_dictionary_v1 import decode,loss_gradient


@torch.no_grad()
def value_gradient(native,whitener,raw_basis,indices,values,row_scale,total,chunk=256):
    left,right,down=native
    readers,_,_,_=decode(raw_basis,indices,values,row_scale);a,b=readers.chunk(2)
    writer,_,_,solver=solve(down,left,right,a,b)
    loss,gradient,details=loss_gradient((left,right,whitener@down),whitener@writer,
        raw_basis,indices,values,row_scale,total,chunk)
    details['writer_solve']=solver
    return loss,gradient,writer,details
