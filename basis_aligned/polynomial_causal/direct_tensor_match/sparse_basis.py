"""Exact Tucker gauge search followed by hard support and coefficient refit."""
import torch
from core import normrows,packed_quadratic,metric

def dense_core(packed,r):
 i,j=torch.triu_indices(r,r,device=packed.device);g=packed.new_zeros(len(packed),r,r)
 g[:,i,j]=packed/torch.where(i==j,1.,2.);g[:,j,i]=g[:,i,j]
 return g

def gauge(state,K,J):
 P=normrows(state['leaf']);W=state['weight'];G=dense_core(state['core'],len(P))
 S=torch.matrix_exp(K);T=torch.matrix_exp(J);Si=torch.linalg.inv(S);Ti=torch.linalg.inv(T)
 leaf=S@P;writer=W@T;pn=leaf.norm(dim=-1);wn=writer.norm(dim=0)
 core=torch.einsum('ab,bij->aij',Ti,G);core=Si.T@core@Si
 core=core*wn[:,None,None]*pn[None,:,None]*pn[None,None,:]
 i,j=torch.triu_indices(len(P),len(P),device=P.device);packed=core[:,i,j]*torch.where(i==j,1.,2.)
 return leaf/pn[:,None],writer/wn[None,:],packed

def coefficients(P,W,G):
 i,j=torch.triu_indices(len(P),len(P),device=P.device)
 return W@G@packed_quadratic(P[i],P[j])

def sparse_refits(P,W,G,target,budgets=(3,4,6,8,12)):
 i,j=torch.triu_indices(len(P),len(P),device=P.device);atoms=packed_quadratic(P[i],P[j]);chol=torch.linalg.cholesky(metric(P.shape[1],2))
 design=torch.einsum('oa,pm->aopm',W,atoms).permute(0,2,1,3).reshape(G.numel(),-1) # columns represented as output/coefficient matrices
 design=(design.reshape(G.numel(),W.shape[0],-1)@chol).flatten(1).T
 rhs=(target@chol).flatten();order=G.abs().flatten().argsort(descending=True);rows=[]
 for budget in budgets:
  ids=order[:budget];fit=torch.linalg.lstsq(design[:,ids],rhs).solution;error=float((design[:,ids]@fit-rhs).norm()/rhs.norm())
  rows.append(dict(budget=budget,relative_gaussian_error=error,support=ids.tolist(),coefficients=fit.tolist(),factor_values=P.numel()+W.numel(),core_values=budget))
 return rows
