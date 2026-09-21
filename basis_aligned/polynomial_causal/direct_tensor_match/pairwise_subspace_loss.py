"""Relax pair-private supports with separately reusable pairwise dictionaries."""
import torch
from shared_subspace_loss import SharedSubspaceLoss

class PairwiseSubspaceLoss(SharedSubspaceLoss):
 def spans(self,parameters):
  # parameters: shared12, shared13, shared23, private1, private2, private3.
  groups=((0,1),(0,2),(1,2))
  return [torch.linalg.qr(torch.cat([parameters[a],parameters[b],parameters[3+j]],1),mode='reduced').Q for j,(a,b) in enumerate(groups)]

def from_common(parameters):
 """Exact span-preserving initialization from an even-width common dictionary."""
 common=parameters[0];assert common.shape[1]%2==0
 a,b=common.chunk(2,dim=1)
 return [a.clone(),b.clone(),((a+b)/2**.5).clone()]+[p.clone() for p in parameters[1:]]
