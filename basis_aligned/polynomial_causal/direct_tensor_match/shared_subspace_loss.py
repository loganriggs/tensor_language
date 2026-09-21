"""Exact dense-core elimination for tied common and pair-private input spans."""
import torch

class SharedSubspaceLoss:
 def __init__(self, targets):
  self.targets=targets.reshape(3,2,*targets.shape[-2:])
  self.norms=self.targets.square().sum((1,2,3))
 def spans(self, parameters):
  common=parameters[0]
  return [torch.linalg.qr(torch.cat([common,private],dim=1),mode='reduced').Q for private in parameters[1:]]
 def loss(self, parameters, explicit=False):
  losses=[]
  for target,norm,U in zip(self.targets,self.norms,self.spans(parameters)):
   core=U.T@target@U
   if explicit:
    losses.append((target-U@core@U.T).square().sum()/norm)
   else:
    losses.append(1-core.square().sum()/norm)
  return torch.stack(losses).mean()
