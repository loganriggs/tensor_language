"""Complete city-removal write with one supplied residual6 array; approximate RMS8."""
import torch
import torch.nn.functional as F
from attention7 import execute as attention7
from readers import execute as readers

def execute(program,residual6,token_ids,city,destination,return_queries=False):
    if residual6.shape[0]!=1:raise ValueError('Only batch size one is certified')
    generated=attention7(program['attention7'],residual6,token_ids)
    other=generated['other_sources'].double();eps=torch.finfo(torch.float32).eps
    rho=(other.square().mean(-1,keepdim=True)+eps).sqrt();head=program['head8']
    projected=(F.linear(other,head['sources'].double())+program['attention7']['lambdas8'][0].double()*readers(program['readers'],generated['normalized_mlp7']).double())/rho
    q1,k1,q2,k2,value=projected.split(128,-1);t=projected.shape[1]
    inv=1/(10000**(torch.arange(0,128,2,dtype=torch.float32,device=projected.device)/128));angles=torch.outer(torch.arange(t,dtype=torch.float32,device=projected.device),inv)
    co,si=angles.cos().bfloat16().double(),angles.sin().bfloat16().double()
    def rotate(x):
        x=x/(x.square().mean(-1,keepdim=True)+eps).sqrt();a,b=x.chunk(2,-1)
        return torch.cat([a*co+b*si,-a*si+b*co],-1)
    q1,k1,q2,k2=[rotate(x) for x in [q1,k1,q2,k2]]
    route=((q1*k1[:,city,None]).sum(-1)/128)*((q2*k2[:,city,None]).sum(-1)/128)
    inherited=generated['first_values'][:,city,256:384].double()
    value=(1-head['mixture'].double())*value[:,city]+head['mixture'].double()*inherited
    support=destination & (torch.arange(t,device=projected.device)>=city)
    delta=-F.linear(route[...,None]*value[:,None],head['output'].double())*support[None,:,None]
    return (delta,torch.stack([q1,q2],2)) if return_queries else delta
