"""Generate key RMS; optionally estimate mixed8 RMS from the other-source context."""
import torch
import torch.nn.functional as F
from readers import execute as readers

def execute(program,head,normalized_mlp7_city,other_city_sources,lambda8,
            rotated_queries,inherited_value,city,destination,input_rms=None):
    eps=torch.finfo(torch.float32).eps
    other=other_city_sources.double()
    rho=(other.square().mean(-1,keepdim=True)+eps).sqrt() if input_rms is None else input_rms.double()
    W=torch.cat([head[k] for k in ['k1','k2','current_value']]).double()
    projected=(F.linear(other,W)+lambda8.double()*readers(program,normalized_mlp7_city).double())/rho
    k1,k2,value=projected.split(128,-1)
    inv=1/(10000**(torch.arange(0,128,2,dtype=torch.float32,device=projected.device)/128))
    angle=city*inv;co=angle.cos().bfloat16().double();si=angle.sin().bfloat16().double()
    factors=[]
    for i,key in enumerate([k1,k2]):
        key=key/(key.square().mean(-1,keepdim=True)+eps).sqrt()
        a,b=key.chunk(2,-1);rotated=torch.cat([a*co+b*si,-a*si+b*co],-1)
        factors.append((rotated_queries[:,:,i].double()*rotated[:,None]).sum(-1)/128)
    value=(1-head['mixture'].double())*value+head['mixture'].double()*inherited_value.double()
    channels=(factors[0]*factors[1])[...,None]*value[:,None]
    support=destination & (torch.arange(rotated_queries.shape[1],device=projected.device)>=city)
    return -F.linear(channels,head['output'].double())*support[None,:,None]
