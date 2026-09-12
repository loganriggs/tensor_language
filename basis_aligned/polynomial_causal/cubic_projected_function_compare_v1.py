"""Compare complete projected coefficient functions, not isolated source vectors.
The private query/output maps are solved against the same target weights.
Head identities stay separate. No native normalization/behavior is inferred.
"""
import torch
from cubic_secant_block_v1 import gram_kernel

def solve(g,rhs):
 scale=g.diagonal().sqrt();norm=g/scale[:,None]/scale[None,:]
 l=torch.linalg.cholesky(norm)
 return torch.cholesky_solve(rhs/scale[:,None],l)/scale[:,None]

def compare(first,second,weights):
 a,ma=first;b,mb=second;n=len(ma)
 g,k=gram_kernel(torch.cat((a,b)),torch.block_diag(ma,mb),weights)
 ga,gb,gab=g[:n,:n],g[n:,n:],g[:n,n:];ka,kb,kab=k[:,:n,:n],k[:,n:,n:],k[:,:n,n:]
 ea=torch.stack([torch.trace(solve(ga,x)) for x in ka]);eb=torch.stack([torch.trace(solve(gb,x)) for x in kb])
 cross=torch.stack([(gab*solve(gb,solve(ga,x).T).T).sum() for x in kab])
 diff=ea+eb-2*cross;den=(ea+eb)/2
 return dict(first_energy=ea,second_energy=eb,cross_inner_product=cross,squared_difference=diff,cosine=cross.sum()/(ea.sum()*eb.sum()).sqrt(),symmetric_relative_error=(diff.sum().clamp_min(0)/den.sum()).sqrt(),gram_conditions=[torch.linalg.cond(ga),torch.linalg.cond(gb)])
