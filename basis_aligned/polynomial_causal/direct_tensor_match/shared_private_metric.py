"""Output-weighted implicit loss with shared mixed atoms and one private square branch."""
import torch

def gram(L,R,A,B):
 return .5*((L.T@A)*(R.T@B)+(L.T@B)*(R.T@A))

class SharedPrivateMetric:
 def __init__(self,T,M,output=5,ridge=1e-10):
  self.T=T;self.M=M;self.output=output;self.ridge=ridge
  self.energy=(T*torch.einsum('ab,bij->aij',M,T)).sum()
 def loss(self,L,R,V,detach=True):
  S=gram(L,R,L,R);C=gram(L,R,V,V);K=gram(V,V,V,V)
  rhs=torch.einsum('ir,oij,jr->ro',L,self.T,R);rp=torch.einsum('ir,oij,jr->ro',V,self.T,V)
  A=S+self.ridge*torch.eye(len(S),dtype=S.dtype,device=S.device)
  X=torch.linalg.solve(A,rhs);Z=torch.linalg.solve(A,C)
  v=torch.linalg.solve(K+self.ridge*torch.eye(len(K),dtype=S.dtype,device=S.device)-C.T@Z,(rp-C.T@X)@self.M[:,self.output]/self.M[self.output,self.output])
  W=X.clone();W[:,self.output]-=Z@v
  if detach:W=W.detach();v=v.detach()
  private=torch.zeros((len(v),len(self.T)),dtype=L.dtype,device=L.device);private[:,self.output]=v
  coefficients=torch.cat([W,private]);fullgram=torch.cat([torch.cat([S,C],1),torch.cat([C.T,K],1)],0);fullrhs=torch.cat([rhs,rp])
  loss=1+(-2*((fullrhs@self.M)*coefficients).sum()+((fullgram@coefficients@self.M)*coefficients).sum()+self.ridge*((coefficients@self.M)*coefficients).sum())/self.energy
  return loss,W.T,v
 def dense(self,L,R,V,W,v):
  raw=torch.einsum('ir,or,jr->oij',L,W,R);hat=(raw+raw.transpose(-1,-2))/2
  one=torch.nn.functional.one_hot(torch.tensor(self.output,device=L.device),len(self.T)).to(L)
  return hat+one[:,None,None]*((V*v)@V.T)[None]
 def explicit(self,hat,W,v):
  E=hat-self.T
  reg=(W*(self.M@W)).sum()+self.M[self.output,self.output]*v.square().sum()
  return ((E*torch.einsum('ab,bij->aij',self.M,E)).sum()+self.ridge*reg)/self.energy
