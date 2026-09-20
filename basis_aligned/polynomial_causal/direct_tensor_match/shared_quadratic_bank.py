"""Exact self Gram and native teacher cross for a shared low-rank quadratic bank."""
import torch
from quartic_cp import directional

def normalize_bank(U,V):
 E=torch.cat([U,V],1);F=torch.cat([V,U],1)/2;edge=F@E.transpose(-1,-2);norm2=(edge*edge.transpose(-1,-2)).sum((-1,-2));scale=norm2.clamp_min(1e-24).pow(.25)
 return U/scale[:,None,None],V/scale[:,None,None]

def roots(bank,device):return torch.triu_indices(bank,bank,device=device)

def bank_gram(U,V):
 E=torch.cat([U,V],1);F=torch.cat([V,U],1)/2;edge=torch.einsum('iad,jbd->ijab',F,E);quad=torch.einsum('ijab,jiba->ij',edge,edge);i,j=roots(len(U),U.device)
 first=quad[i[:,None],i[None,:]]*quad[j[:,None],j[None,:]]+quad[i[:,None],j[None,:]]*quad[j[:,None],i[None,:]]
 a=edge[i[:,None],i[None,:]];b=edge[i[None,:],j[:,None]];c=edge[j[:,None],j[None,:]];d=edge[j[None,:],i[:,None]];trace=((a@b@c)*d.transpose(-1,-2)).sum((-1,-2))
 return (first+4*trace)/6

def native_bank_cross(teacher,U,V):
 i,j=roots(len(U),U.device);k=U.shape[1];count=len(i);a=U[i,:,None,:].expand(count,k,k,-1).reshape(count*k*k,-1);b=V[i,:,None,:].expand(count,k,k,-1).reshape(count*k*k,-1);c=U[j,None,:,:].expand(count,k,k,-1).reshape(count*k*k,-1);d=V[j,None,:,:].expand(count,k,k,-1).reshape(count*k*k,-1)
 return directional(*teacher,[a,b,c,d]).reshape(count,k*k,-1).sum(1).T

def bank_entries(U,V,indices):
 a,b=roots(len(U),U.device)
 def pair(i,j):return (.5*(U[:,:,i]*V[:,:,j]+U[:,:,j]*V[:,:,i])).sum(1).T
 def outer(s,t):return .5*(s[:,a]*t[:,b]+t[:,a]*s[:,b])
 i,j,k,l=indices.T
 return (outer(pair(i,j),pair(k,l))+outer(pair(i,k),pair(j,l))+outer(pair(i,l),pair(j,k)))/3
