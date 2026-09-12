"""Exact scalar objective, gradient and Hessian-vector products, no dense Jacobian.
This supports a trust-region Newton comparison, not a Gauss-Newton guarantee.
The caller supplies the exact contracted variable-projection objective.
"""
import numpy as np
import torch
class Curvature:
 def __init__(self,objective,shape,device='cpu'):
  self.objective=objective;self.shape=shape;self.device=device;self.evaluations=0;self.hessian_products=0
 def tensor(self,x):return torch.as_tensor(np.array(x,copy=True),dtype=torch.float64,device=self.device).reshape(self.shape)
 def fun(self,x):
  z=self.tensor(x).requires_grad_(True);v=self.objective(z);g=torch.autograd.grad(v,z)[0];self.evaluations+=1
  return float(v.detach()),g.detach().cpu().numpy().ravel()
 def hessp(self,x,direction):
  z=self.tensor(x).requires_grad_(True);v=self.objective(z);g=torch.autograd.grad(v,z,create_graph=True)[0]
  d=self.tensor(direction);h=torch.autograd.grad((g*d).sum(),z)[0];self.hessian_products+=1
  return h.detach().cpu().numpy().ravel()
