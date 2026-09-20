"""Exact polynomial substitution x=Lz; fixes function when changing coordinates."""
import torch
from core import terms,multiply,normrows

def transform_matrix(L,degree):
 d=L.shape[0];rows=[]
 for term in terms(d,degree):
  c=L.new_ones(1)
  for n,i in enumerate(term):c=multiply(c,L[i],d,n,1)
  rows.append(c)
 return torch.stack(rows)

def transform_model_state(model,L):
 s={k:v.detach().clone() for k,v in model.state_dict().items()};kind=model.kind
 if kind=='monomial':s['weight']=s['weight']@transform_matrix(L,model.degree)
 elif kind=='cp':
  a=normrows(s['left'])@L;b=normrows(s['right'])@L
  s['weight']*=a.norm(dim=-1)*b.norm(dim=-1);s['left']=a;s['right']=b
 elif kind=='tucker':
  a=normrows(s['leaf'])@L;n=a.norm(dim=-1);i,j=torch.triu_indices(model.width,model.width,device=L.device)
  s['core']*=n[i]*n[j];s['leaf']=a
 elif kind=='tree':
  T=transform_matrix(L,2);a=normrows(s['left'])@T;b=normrows(s['right'])@T
  s['weight']*=(a.norm(dim=-1)[:,None]*b.norm(dim=-1)[None,:]).flatten();s['left']=a;s['right']=b
 elif kind=='dag':
  a=normrows(s['bank'])@transform_matrix(L,2);n=a.norm(dim=-1);i,j=torch.triu_indices(model.width,model.width,device=L.device)
  s['weight']*=n[i]*n[j];s['bank']=a
 return s
