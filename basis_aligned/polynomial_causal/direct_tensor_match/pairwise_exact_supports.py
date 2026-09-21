"""Exact input supports and their pairwise intersections, derived only from T.
For full-rank native forms these are uninformative; this is a planted recovery
control, not a native truncation rule or approximate intersection theorem.
"""
import torch

def supports(target):
 pair=[]
 for j in range(3):
  T=target[2*j:2*j+2];e,U=torch.linalg.eigh((T@T).sum(0));pair.append(U[:,e>1e-10*e[-1]])
 common=[]
 for a,b in ((0,1),(0,2),(1,2)):
  e,U=torch.linalg.eigh(pair[a]@pair[a].T+pair[b]@pair[b].T);common.append(U[:,e>2-1e-8])
 return common+pair
