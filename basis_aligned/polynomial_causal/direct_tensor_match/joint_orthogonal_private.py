"""Learn pairwise shared dictionaries together with private bases.

Same executable family and price as the fixed-shared orthonormal-private fit.
The shared cores are conditionally optimized rather than fixed as directions move.
"""
import torch
from orthogonal_private_varpro import OrthogonalPrivateMetric
from pairwise_reader_graph import GROUPS

class JointOrthogonalPrivateMetric(OrthogonalPrivateMetric):
 def loss(self,params,dense=False,solve_through=False):
  self.shared=[torch.linalg.qr(torch.cat([params[a],params[b]],1),mode='reduced').Q for a,b in GROUPS]
  self.common=[P.T@T@P for P,T in zip(self.shared,self.targets)]
  return super().loss(params[3:],dense=dense,solve_through=solve_through)
