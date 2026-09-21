"""Pairwise shared graph components with arbitrary leading batch dimensions."""
import torch
from pairwise_reader_graph import source_reads

def component_scalars(z,h,program):
 reads=source_reads(z,program);scale=(h.square().mean(-1)+torch.finfo(torch.float32).eps).sqrt();values=[]
 for j in range(3):
  p=program['pairs'][str(j)];a=(h@p['h_reader']-.5*reads[...,2*j])/scale-p['alpha'];b=reads[...,2*j+1]/scale-p['beta'];values.append(a*b)
 return torch.stack(values,-1)
