"""Exact coordinate solves for a coefficient-regularized bilinear component loss.

The conditional component is (A0 + Da @ wa) * (B0 + Db @ wb).
Each block update is convex quadratic; joint optimization is not convex.
"""
import torch

class ComponentPairReadout:
 def __init__(self,Ga,Gb,ca,cb,energy,Da,Db,A0,B0,target,lam,ridge=1e-10):
  self.Ga,self.Gb,self.ca,self.cb=Ga,Gb,ca,cb
  self.energy=energy;self.Da,self.Db=Da,Db;self.A0,self.B0=A0,B0
  self.target=target;self.lam=lam;self.ridge=ridge
  self.variance=(target-target.mean()).square().mean()
  assert self.variance>0
 def value(self,wa,wb):
  A=self.A0+self.Da@wa;B=self.B0+self.Db@wb
  coefficient=1+(wa@(self.Ga@wa)+wb@(self.Gb@wb)-2*(wa@self.ca+wb@self.cb)+self.ridge*(wa.square().sum()+wb.square().sum()))/self.energy
  return coefficient+self.lam*(A*B-self.target).square().mean()/self.variance
 def update(self,wa,wb,side):
  if side=='a':
   factor=self.B0+self.Db@wb;D=self.Da*factor[:,None];y=self.target-self.A0*factor;G,c=self.Ga,self.ca
  elif side=='b':
   factor=self.A0+self.Da@wa;D=self.Db*factor[:,None];y=self.target-self.B0*factor;G,c=self.Gb,self.cb
  else:raise ValueError(side)
  scale=self.lam/(len(y)*self.variance)
  matrix=(G+self.ridge*torch.eye(len(G),dtype=G.dtype,device=G.device))/self.energy+scale*(D.T@D)
  rhs=c/self.energy+scale*(D.T@y)
  return torch.linalg.solve(matrix,rhs)
 def fit(self,wa,wb,cycles):
  wa=wa.clone();wb=wb.clone();history=[float(self.value(wa,wb))]
  for step in range(cycles):
   wa=self.update(wa,wb,'a');wb=self.update(wa,wb,'b')
   current=float(self.value(wa,wb));assert current<=history[-1]+1e-9*max(1,abs(history[-1]))
   history.append(current)
  return wa,wb,history
