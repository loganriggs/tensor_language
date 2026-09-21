"""Alternating exact solves for a shared/private quadratic block approximation."""
import torch

def private_forms(E,B,C0):
 # Symmetric C minimizes ||C-C0||^2 + 2||E C-B||^2.
 # (I/2+E.T E)C+C(I/2+E.T E)=C0+E.T B+B.T E.
 M=.5*torch.eye(E.shape[1],dtype=E.dtype,device=E.device)+E.T@E
 values,U=torch.linalg.eigh(M);rhs=C0+E.T@B+B.transpose(-1,-2)@E
 return U@((U.T@rhs@U)/(values[:,None]+values[None,:]))@U.T

def cross_map(B,C):
 return torch.linalg.lstsq(torch.cat(list(C),1).T,torch.cat(list(B),1).T,driver='gelsd').solution.T

def fit(B,C0,steps=200,tolerance=1e-10):
 C=C0.clone();E=cross_map(B,C)
 def loss(E,C):return float((C-C0).square().sum()+2*(E@C-B).square().sum())
 initial=loss(E,C);history=[initial]
 for step in range(steps):
  previous=history[-1];C=private_forms(E,B,C0);middle=loss(E,C);E=cross_map(B,C);value=loss(E,C)
  assert middle<=previous+1e-8*max(1,previous) and value<=middle+1e-8*max(1,middle)
  history.append(value)
  if previous-value<=tolerance*max(initial,1e-20):break
 return E,C,dict(initial_squared_error=initial,final_squared_error=history[-1],iterations=len(history)-1,history=history,scope='Alternating conditional global optima; no global optimum guarantee jointly inEandC.')
