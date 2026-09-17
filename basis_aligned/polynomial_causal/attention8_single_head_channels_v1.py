"""Exact head8.2 channel computation; evaluates no other attention head."""
import torch
import torch.nn.functional as F

def channels(p,current8,token_ids):
 b,t,d=current8.shape;assert d==1152
 lookup={int(v):i for i,v in enumerate(p['token_ids'].tolist())}
 try:indices=[lookup[int(v)] for v in token_ids.reshape(-1).tolist()]
 except KeyError as e:raise ValueError('Token outside frozen inherited-value table') from e
 indices=torch.tensor(indices,device=current8.device).reshape(b,t)
 first=F.embedding(indices,p['first_table'][:,256:384]).to(current8.dtype).reshape(b,t,1,128)
 inv=1/(10000**(torch.arange(0,128,2,device=current8.device,dtype=torch.float32)/128));angles=torch.outer(torch.arange(t,device=current8.device,dtype=torch.float32),inv)
 co=angles.cos().bfloat16()[None,:,None];si=angles.sin().bfloat16()[None,:,None]
 def factor(name):
  x=F.rms_norm(F.linear(current8,p[name][256:384].to(current8.dtype)).reshape(b,t,1,128),(128,));a,z=x.chunk(2,-1)
  return torch.cat([a*co+z*si,-a*si+z*co],-1).to(current8.dtype)
 q,k,q2,k2=[factor(name) for name in ['q1','k1','q2','k2']]
 a=torch.einsum('bqhd,bkhd->bhqk',q,k)/128;a2=torch.einsum('bqhd,bkhd->bhqk',q2,k2)/128
 pattern=(a*a2).masked_fill(~torch.ones(t,t,device=current8.device,dtype=torch.bool).tril(),0)
 v=F.linear(current8,p['value'][256:384].to(current8.dtype)).reshape(b,t,1,128);v=(1-p['mixture'])*v+p['mixture']*first
 y=torch.einsum('bhqk,bkhd->bqhd',pattern,v).contiguous().reshape(b,t,128)
 return y.reshape(b,t,1,128)
