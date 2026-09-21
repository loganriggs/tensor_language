"""Explicit private-core energy regularization for joint shared/private fitting.

Objective: mean_j [||T_j-H_j||_F^2 + rho ||C_j||_F^2], normalized
per target pair. Orthonormal P,D make the private-core penalty invariant under
rotations inside those bases. This is an amplitude penalty, not interaction
count or semantic sparsity. Conditional solve: (1+rho) C - J C J = rhs.
"""
import torch
from pairwise_reader_graph import GROUPS
from orthogonal_private_varpro import OrthogonalPrivateMetric

class RegularizedJointMetric(OrthogonalPrivateMetric):
 def __init__(self,targets,shared,rho):
  super().__init__(targets,shared)
  if rho<=0:raise ValueError('Use unregularized kernel for rho=0 controls')
  self.rho=rho
 def loss(self,params,dense=False,solve_through=False):
  values=[];states=[]
  for j,(a,b) in enumerate(GROUPS):
   P=torch.linalg.qr(torch.cat([params[a],params[b]],1),mode='reduced').Q
   D=torch.linalg.qr(params[3+j],mode='reduced').Q;T=self.targets[j]
   A=P.T@T@P;W=P.T@D;J=W.T@W;rhs=D.T@T@D-W.T@A@W
   if solve_through:
    p=D.shape[1]
    if p>32:raise ValueError('Kronecker control is toy-only')
    K=(1+self.rho)*torch.eye(p*p,dtype=D.dtype,device=D.device)-torch.kron(J.contiguous(),J.contiguous())
    C=torch.linalg.solve(K,rhs.reshape(2,-1).T).T.reshape_as(rhs)
   else:
    with torch.no_grad():
     eig,U=torch.linalg.eigh(J);den=1+self.rho-eig[:,None]*eig[None,:]
     if float(den.min())<self.rho*.99:raise ValueError('Invalid orthonormal geometry')
     C=U@((U.T@rhs@U)/den)@U.T
   shared_core=A-W@C@W.T
   if dense:residual=(T-D@C@D.T-P@shared_core@P.T).square().sum()
   else:residual=1-A.square().sum()-2*(C*rhs).sum()+C.square().sum()-(W@C@W.T).square().sum()
   # C is the exact conditional minimizer of the penalized objective. The
   # envelope gradient is valid for this sum, not the unpenalized term alone.
   values.append(residual+self.rho*C.square().sum());states.append((P,D,W,C,A))
  return torch.stack(values).mean(),states
 def reconstruction(self,params):
  _,states=self.loss(params)
  return torch.stack([(T-D@C@D.T-P@(A-W@C@W.T)@P.T).square().sum() for T,(P,D,W,C,A) in zip(self.targets,states)]).mean()
