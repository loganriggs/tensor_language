"""Shared-output orthogonal pursuit using exact conditional coefficient energy."""
import torch

def select(k,c,count):
 scale=k.diagonal().sqrt();g=k/scale[:,None]/scale[None,:];target=c/scale[:,None]
 active=[];history=[]
 for step in range(count):
  if active:
   a=torch.tensor(active,device=k.device);cross=g[:,a];sub=g[a][:,a]
   coeff=torch.linalg.solve(sub,target[a]);residual=target-cross@coeff
   variance=g.diagonal()-(cross*torch.linalg.solve(sub,cross.T).T).sum(-1)
  else:residual=target;variance=g.diagonal()
  score=residual.square().sum(-1)/variance.clamp_min(1e-30);score[variance<=1e-12]=-torch.inf
  if active:score[a]=-torch.inf
  chosen=int(score.argmax());assert torch.isfinite(score[chosen]);active.append(chosen)
  history.append(float(score[chosen]))
 a=torch.tensor(active,device=k.device);coeff=torch.linalg.solve(g[a][:,a],target[a])/scale[a,None]
 return a,coeff,history
