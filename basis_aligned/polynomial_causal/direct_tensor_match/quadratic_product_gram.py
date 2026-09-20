"""Exact symmetric coefficient Gram for products of dense quadratic forms."""
import torch

def product_gram(Q,R,S,T):
 # All matrices symmetric. Each row is the feature (xQx)(xRx).
 q=Q.flatten(1);r=R.flatten(1);s=S.flatten(1);t=T.flatten(1)
 result=(q@s.T)*(r@t.T)+(q@t.T)*(r@s.T)
 cyclic=[]
 for j in range(len(S)):
  QS=Q@S[j];RT=R@T[j];cyclic.append((QS*RT.transpose(-1,-2)).sum((-1,-2)))
 return (result+4*torch.stack(cyclic,1))/6
