"""Residual-aware selection in an implicit quadratic coefficient Hilbert space."""
import torch

def gram(L,R):
 return .5*((L@L.T)*(R@R.T)+(L@R.T)*(R@L.T))

def greedy(K,B,width):
 # B[o,j]=<target_o, atom_j>, in the specified output weighting.
 diagonal=K.diagonal().clone();cross=B.clone();Q=[];selected=[];gains=[]
 for _ in range(width):
  valid=diagonal>1e-12
  score=cross.square().sum(0)/diagonal.clamp_min(1e-30);score[~valid]=-torch.inf
  j=int(score.argmax())
  if not valid[j]:break
  row=K[j].clone()
  if Q:
   old=torch.stack(Q);row-=old[:,j]@old
  q=row/diagonal[j].sqrt();b=cross[:,j]/diagonal[j].sqrt()
  cross-=b[:,None]*q[None,:];diagonal-=q.square();diagonal[j]=0
  Q.append(q);selected.append(j);gains.append(b.square())
 return selected,torch.stack(gains).cumsum(0)

def refit(K,B,ids):
 S=torch.tensor(ids,device=K.device);H=K[S][:,S];e,V=torch.linalg.eigh(H);cut=e.max()*1e-11;inv=torch.where(e>cut,e.reciprocal(),0.);D=(B[:,S]@V*inv)@V.T
 gain=(D*B[:,S]).sum(1)
 normal=float((D@H-B[:,S]).norm()/B[:,S].norm().clamp_min(1e-30))
 return D,gain,normal
