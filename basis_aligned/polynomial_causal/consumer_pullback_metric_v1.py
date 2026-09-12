"""Exact polynomial consumer pullback under independent Gaussian and token inputs."""
import itertools
import torch
PERMS=list(itertools.permutations(range(3)))
def query_gram(p,rotation):
 # beta_a(q) = sum_n coeff[a,n] (ell_n.q)(right_n.q) writer_n.
 ell=[];right=[];writers=[];coeff=[]
 for r in range(p['atom_write'].shape[0]):
  for i,j,k in PERMS:
   ell.append(p['q1'].double().T@rotation@p['atom_k1'][r,i].double())
   right.append(p['q2'].double().T@rotation@p['atom_k2'][r,j].double())
   writers.append(p['atom_write'][r,k].double());coeff.append(p['dual'][:,r].double())
 ell=torch.stack(ell);right=torch.stack(right);writers=torch.stack(writers);coeff=torch.stack(coeff,1)
 trace=(ell*right).sum(1)
 gaussian=trace[:,None]*trace[None,:]+(ell@ell.T)*(right@right.T)+(ell@right.T)*(right@ell.T)
 B=coeff@(gaussian*(writers@writers.T))@coeff.T
 return B,(ell,right,writers,coeff)
def fourth_moment(C,tokens):
 covariance=C@C.T;second=tokens.T@tokens/len(tokens)
 pairs=torch.einsum('ni,nj->nij',tokens,tokens).flatten(1)
 fourth=(pairs.T@pairs/len(tokens)).reshape(4,4,4,4)
 for i,j,k,l in itertools.product(range(4),repeat=4):
  fourth[i,j,k,l]+=covariance[i,j]*covariance[k,l]+covariance[i,k]*covariance[j,l]+covariance[i,l]*covariance[j,k]
  fourth[i,j,k,l]+=covariance[i,j]*second[k,l]+covariance[i,k]*second[j,l]+covariance[i,l]*second[j,k]+covariance[j,k]*second[i,l]+covariance[j,l]*second[i,k]+covariance[k,l]*second[i,j]
 return fourth
def metric_from(B,T):
 H=torch.zeros(4,4,dtype=torch.float64)
 for a in range(2):
  ia=(0,1,2+a)
  for b in range(2):
   ib=(0,1,2+b)
   for i in ia:
    for j in ib:H[i,j]+=B[a,b]*T[tuple(k for k in ia if k!=i)+tuple(k for k in ib if k!=j)]
 return (H+H.T)/2

def weighted_component(M,H):
 eig,Q=torch.linalg.eigh(H);assert float(eig.min())>0
 root=(Q*eig.sqrt()[None,:])@Q.T
 u,s,v=torch.linalg.svd(root@M,full_matrices=False)
 reader=v[0];selected=(M@reader)[:,None]*reader[None,:]
 projector=selected@torch.linalg.pinv(M)
 return selected,projector,reader,s
