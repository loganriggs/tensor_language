"""Move private input spans; eliminate symmetric cores exactly at every step.
Shared spaces remain fixed. The envelope gradient avoids differentiating an
eigendecomposition with repeated eigenvalues in the Lyapunov solve.
"""
import torch
from common_private_als import private_forms

class FreePrivateMetric:
 def __init__(self,targets,shared):
  self.scales=targets.square().sum((-1,-2)).reshape(3,2).sum(1).sqrt()
  self.targets=(targets/self.scales.repeat_interleave(2)[:,None,None]).reshape(3,2,*targets.shape[-2:])
  self.shared=[torch.linalg.qr(S,mode='reduced').Q for S in shared]
  self.common=[P.T@T@P for P,T in zip(self.shared,self.targets)]
 def loss(self,params,dense=False,solve_through=False):
  values=[];states=[]
  for j,(P,T,A) in enumerate(zip(self.shared,self.targets,self.common)):
   Z,E=params[2*j:2*j+2];V=torch.linalg.qr(Z-P@(P.T@Z),mode='reduced').Q;B=P.T@T@V;C0=V.T@T@V
   if solve_through:
    p=E.shape[1];I=torch.eye(p,dtype=E.dtype,device=E.device);M=.5*I+E.T@E;rhs=C0+E.T@B+B.transpose(-1,-2)@E
    K=torch.kron(M.contiguous(),I)+torch.kron(I,M.contiguous());C=torch.linalg.solve(K,rhs.reshape(2,-1).T).T.reshape_as(C0)
   else:
    with torch.no_grad():C=private_forms(E,B,C0)
   if dense:
    H=P@A@P.T+V@C@V.T+P@E@C@V.T+V@C@E.T@P.T;value=(T-H).square().sum()
   else:value=1-A.square().sum()+C.square().sum()-2*(C*C0).sum()+2*(E@C).square().sum()-4*((E@C)*B).sum()
   values.append(value);states.append((P,V,E,C,A))
  return torch.stack(values).mean(),states

def parameters_from_private(shared,private):
 params=[]
 for S,D in zip(shared,private):
  P=torch.linalg.qr(S,mode='reduced').Q;V,R=torch.linalg.qr(D-P@(P.T@D),mode='reduced');E=torch.linalg.solve(R.T,(P.T@D).T).T;params.extend([V,E])
 return params
