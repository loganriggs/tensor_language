"""Source-form reconstruction and channel-coefficient permutation controls."""
import torch

def forms(L,R,C):
 out=[]
 for c in C:
  raw=L.T@(c[:,None]*R);out.append((raw+raw.T)/2)
 return torch.stack(out)

def permute(C,kind,seed):
 rng=torch.Generator().manual_seed(seed)
 if kind=='common':return C[:,torch.randperm(C.shape[1],generator=rng).to(C.device)]
 if kind=='pairwise':return torch.cat([C[2*j:2*j+2,torch.randperm(C.shape[1],generator=rng).to(C.device)] for j in range(3)])
 raise ValueError(kind)

def overlap(Q,root,width,edge):
 T=Q if root is None else root@Q@root;bases=[]
 for j in range(3):
  pair=T[2*j:2*j+2];_,U=torch.linalg.eigh((pair@pair).sum(0));bases.append(U[:,-width:])
 values=[float(torch.linalg.svdvals(bases[a].T@bases[b])[:edge].square().mean()) for a,b in ((0,1),(0,2),(1,2))]
 return dict(statistic=sum(values)/3,edge_values=values)
