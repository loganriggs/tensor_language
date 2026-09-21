"""Exact one-channel deletion cost after refitting every remaining output weight."""
import torch

def costs(C,K):
 precision=torch.cholesky_inverse(torch.linalg.cholesky(K))
 return C.square().sum(0)/precision.diag()

def controls():
 g=torch.Generator().manual_seed(937);F=torch.randn(19,7,generator=g,dtype=torch.float64);C=torch.randn(3,7,generator=g,dtype=torch.float64);K=F.T@F;pred=costs(C,K);errs=[]
 for j in range(7):
  ids=torch.tensor([i for i in range(7) if i!=j]);student=torch.linalg.lstsq(F[:,ids],F@C.T).solution;actual=(F[:,ids]@student-F@C.T).square().sum();errs.append(float(abs(actual-pred[j])/actual));assert errs[-1]<1e-12
 return errs
