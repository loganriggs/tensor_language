"""Gauge-only shared-quadratic-bank sparsification, with exact root refits."""
import torch
from core import metric,normrows,multiply,inner
from sparse_basis import dense_core

def gauge(state,K,d):
 bank=normrows(state['bank']);S=torch.matrix_exp(K);Si=torch.linalg.inv(S);new=S@bank
 M=metric(d,2);n=((new@M)*new).sum(-1).sqrt();new=new/n[:,None]
 G=Si.T@dense_core(state['weight'],len(bank))@Si;G=G*n[None,:,None]*n[None,None,:]
 i,j=torch.triu_indices(len(bank),len(bank));root=G[:,i,j]*torch.where(i==j,1.,2.)
 return new,root

def atoms(bank,d):
 i,j=torch.triu_indices(len(bank),len(bank));return multiply(bank[i],bank[j],d,2,2)

def refits(bank,root,target,d,budgets):
 features=atoms(bank,d);chol=torch.linalg.cholesky(metric(d,4));design=(features@chol).T;rhs=(target@chol).T;order=root.norm(dim=0).argsort(descending=True);rows=[]
 for budget in budgets:
  ids=order[:budget];coef=torch.linalg.lstsq(design[:,ids],rhs).solution
  rows.append(dict(products=budget,error=float((design[:,ids]@coef-rhs).norm()/rhs.norm()),support=ids.tolist(),root=coef.T.tolist(),parameter_values=bank.numel()+target.shape[0]*budget))
 return rows
