"""Compile signed repeated input readers into one bank; no approximate equality."""
import torch

def compile_program(factors,C):
 bank=[];indices=[];signs=[];seen={}
 for f in factors:
  ix=[];sg=[]
  for row in f:
   vals=row.detach().cpu().tolist();sign=next((1. if a>0 else -1. for a in vals if a!=0),1.);key=tuple(sign*a for a in vals)
   if key not in seen:seen[key]=len(bank);bank.append(row*sign)
   ix.append(seen[key]);sg.append(sign)
  indices.append(ix);signs.append(sg)
 ix=torch.tensor(indices,device=C.device);sign=torch.tensor(signs,dtype=C.dtype,device=C.device).prod(0)
 return dict(bank=torch.stack(bank),indices=ix,coefficients=C*sign)

def evaluate(p,x):
 z=x@p['bank'].T;phi=z[:,p['indices'][0]]
 for ix in p['indices'][1:]:phi=phi*z[:,ix]
 return phi@p['coefficients'].T
