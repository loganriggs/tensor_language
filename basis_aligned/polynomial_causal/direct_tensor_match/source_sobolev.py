"""Implicit quadratic coefficient + Gaussian source-gradient metric.
T and factors in covariance-whitened coordinates; H=inverse covariance.
Each error term normalized by its teacher energy, blend weight lam>=0.
"""
import torch
class SourceSobolev:
 def __init__(self,T,H,lam,ridge=1e-10):
  self.T=T;self.H=H;self.lam=lam;self.ridge=ridge/T.square().sum()
  self.e0=T.square().sum();HT=H@T;self.e1=(T*HT).sum()
  self.c0=1/((1+lam)*self.e0);self.c1=lam/((1+lam)*self.e1)
  self.target=self.c0*T+self.c1*(HT+HT.transpose(-1,-2))/2
 def statistics(self,L,R):
  LL=L.T@L;RR=R.T@R;LR=L.T@R;HL=self.H@L;HR=self.H@R
  K0=.5*(LL*RR+LR*LR.T)
  K1=.25*(LL*(R.T@HR)+LR*(R.T@HL)+LR.T*(L.T@HR)+RR*(L.T@HL))
  K=self.c0*K0+self.c1*K1
  rhs=torch.einsum('ir,oij,jr->ro',L,self.target,R)
  return K,rhs
 def loss(self,L,R,detach=True):
  K,rhs=self.statistics(L,R)
  W=torch.linalg.solve(K+self.ridge*torch.eye(len(K),device=K.device,dtype=K.dtype),rhs)
  if detach:W=W.detach()
  loss=1-2*(W*rhs).sum()+(W*(K@W)).sum()+self.ridge*W.square().sum()
  return loss,W.T
 def explicit(self,Qhat,W):
  diff=Qhat-self.T
  return self.c0*diff.square().sum()+self.c1*(diff*(self.H@diff)).sum()+self.ridge*W.square().sum()
