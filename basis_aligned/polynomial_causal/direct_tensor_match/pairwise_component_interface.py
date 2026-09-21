"""Pairwise shared graph components with arbitrary leading batch dimensions."""
import torch
from pairwise_reader_graph import source_reads

def component_scalars(z,h,program):
 reads=source_reads(z,program);scale=(h.square().mean(-1)+torch.finfo(torch.float32).eps).sqrt();values=[]
 for j in range(3):
  p=program['pairs'][str(j)];a=(h@p['h_reader']-.5*reads[...,2*j])/scale-p['alpha'];b=reads[...,2*j+1]/scale-p['beta'];values.append(a*b)
 return torch.stack(values,-1)

def move_preserving_dtype(value,device):
 if isinstance(value,dict):return {k:move_preserving_dtype(v,device) for k,v in value.items()}
 if isinstance(value,torch.Tensor):return value.to(device=device,dtype=torch.float64 if value.is_floating_point() else value.dtype)
 return value
