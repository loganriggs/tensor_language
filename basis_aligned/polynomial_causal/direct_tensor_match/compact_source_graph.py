"""Execute mixed shared products and private squares without duplicating projections."""
import torch
from global_mixed_source_graph import source_reads as mixed_reads

def source_reads(z,p):
 q=mixed_reads(z,p)
 q[...,int(p['square_output'])]+=(z@p['square_reader']).square()@p['square_weights']
 return q

def component_scalars(z,h,p):
 q=source_reads(z,p);s=(h.square().mean(-1)+torch.finfo(torch.float32).eps).sqrt()[...,None]
 return ((h@p['h_readers']-.5*q[...,::2])/s-p['alpha'])*(q[...,1::2]/s-p['beta'])
