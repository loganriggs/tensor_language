"""Equivalent shared/private quadratic family with full orthonormal private bases.

H_o = D C_o D.T + P (A_o - W C_o W.T) P.T, W=P.T D.
For fixed D, C - J C J = D.T T D - W.T A W, J=W.T W.
The envelope loss retains the derivative of the Gram term. Degenerate private
spaces overlapping the shared space are rejected rather than regularized away.
"""
import torch

class OrthogonalPrivateMetric:
 def __init__(self, targets, shared):
  self.scales=targets.square().sum((-1,-2)).reshape(3,2).sum(1).sqrt()
  self.targets=(targets/self.scales.repeat_interleave(2)[:,None,None]).reshape(3,2,*targets.shape[-2:])
  self.shared=[torch.linalg.qr(S,mode='reduced').Q for S in shared]
  self.common=[P.T@T@P for P,T in zip(self.shared,self.targets)]
 def loss(self,params,dense=False,solve_through=False):
  values=[];states=[]
  for Z,P,T,A in zip(params,self.shared,self.targets,self.common):
   D=torch.linalg.qr(Z,mode='reduced').Q;W=P.T@D;J=W.T@W
   rhs=D.T@T@D-W.T@A@W
   if solve_through:
    p=D.shape[1]
    if p>32:raise ValueError('Differentiable Kronecker control is toy-only')
    K=torch.eye(p*p,dtype=D.dtype,device=D.device)-torch.kron(J.contiguous(),J.contiguous())
    C=torch.linalg.solve(K,rhs.reshape(2,-1).T).T.reshape_as(rhs)
   else:
    with torch.no_grad():
     eig,U=torch.linalg.eigh(J);den=1-eig[:,None]*eig[None,:]
     if float(den.min())<=1e-10:raise ValueError('Degenerate shared/private overlap')
     C=U@((U.T@rhs@U)/den)@U.T
   shared_core=A-W@C@W.T
   if dense:loss=(T-D@C@D.T-P@shared_core@P.T).square().sum()
   else:loss=1-A.square().sum()-2*(C*rhs).sum()+C.square().sum()-(W@C@W.T).square().sum()
   values.append(loss);states.append((P,D,W,C,A))
  return torch.stack(values).mean(),states
