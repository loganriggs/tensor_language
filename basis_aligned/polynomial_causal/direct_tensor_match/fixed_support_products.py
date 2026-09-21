"""Fit product maps inside fixed pair spans, retaining outside-span error exactly."""
import torch
from overlap_varpro import OverlapMetric
from pairwise_reader_graph import GROUPS

class FixedSupportProducts(OverlapMetric):
 def __init__(self,targets,bases,private,templates,ridge=1e-10):
  # Targets/bases/private use the chosen orthonormal metric coordinates.
  scales=targets.square().sum((-1,-2)).reshape(3,2).sum(1).sqrt()
  self.bases=bases;self.spans=[];self.shared=[];cores=[]
  for j,(a,b) in enumerate(GROUPS):
   shared=torch.cat([bases[a],bases[b]],1)
   U=torch.linalg.qr(torch.cat([shared,private[j]],1),mode='reduced').Q
   self.spans.append(U);self.shared.append(U.T@shared)
   cores.append(U.T@targets[2*j:2*j+2]@U)
  # All pair spans have equal width in this registered family.
  cores=torch.cat(cores)
  super().__init__(cores,templates,ridge)
  self.scales=scales;self.targets=cores/scales.repeat_interleave(2)[:,None,None]
  self.outside=(1-self.targets.square().sum((-1,-2)).reshape(3,2).sum(1)).mean()
 def reader(self,params,j):
  maps,private=params[2*j:2*j+2];basis=self.shared[j];t=self.templates[j];n=len(t['shared_indices'])+len(t['private_indices'])
  return basis.new_zeros(basis.shape[0],n).index_copy(1,t['shared_indices'],basis@maps).index_copy(1,t['private_indices'],private)
 def loss(self,params,detach=True,dense=False):
  value,weights,readers=super().loss(params,detach=detach,dense=dense)
  # Parent implicit formula starts from full target energy1. The dense core
  # residual needs the orthogonal, unrepresented coefficient energy restored.
  return value+(self.outside if dense else 0),weights,readers
