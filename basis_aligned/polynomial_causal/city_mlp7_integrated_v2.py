"""City write from folded MLP7 readers, with explicit native queries/RMS context."""
import torch
import torch.nn.functional as F
from city_mlp7_readers_v1 import execute as readers

def write_from_readers(projected_mlp7_city, head, other_city_sources, lambda8,
            input_rms, key_rms, rotated_queries, inherited_value, city, destination):
    # Output of MLP7 is never reconstructed. Only its three projected readings enter.
    folded=projected_mlp7_city.double()*lambda8.double()
    W=torch.cat([head[k] for k in ['k1','k2','current_value']]).double()
    projected=(F.linear(other_city_sources.double(),W)+folded)/input_rms.double()
    k1,k2,value=projected.split(128,-1)
    t=rotated_queries.shape[1]
    inv=1/(10000**(torch.arange(0,128,2,dtype=torch.float32,device=projected_mlp7_city.device)/128))
    angle=city*inv;co=angle.cos().bfloat16().double();si=angle.sin().bfloat16().double()
    def rotate(key,den):
        a,b=(key/den).chunk(2,-1)
        return torch.cat([a*co+b*si,-a*si+b*co],-1)
    keys=[rotate(k1,key_rms[:,0]),rotate(k2,key_rms[:,1])]
    factors=[(rotated_queries[:,:,i].double()*keys[i][:,None]).sum(-1)/128 for i in range(2)]
    value=(1-head['mixture'].double())*value+head['mixture'].double()*inherited_value.double()
    channels=(factors[0]*factors[1])[...,None]*value[:,None]
    support=destination & (torch.arange(t,device=projected_mlp7_city.device)>=city)
    return -F.linear(channels,head['output'].double())*support[None,:,None]


def execute(program, head, normalized_mlp7_city, **context):
    return write_from_readers(readers(program,normalized_mlp7_city),head,**context)
