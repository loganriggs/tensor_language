"""Residual6-to-head8 city write with exact MLP7 RMS8 closure."""
import torch
import torch.nn.functional as F
from city_attention7_full_v1 import execute as attention7
from city_mlp7_readers_v1 import execute as readers

def execute(program,residual6,token_ids,city,destination,return_rho=False):
    generated=attention7(program['attention7'],residual6,token_ids)
    norm7=generated['normalized_mlp7'];m=F.linear(norm7,program['mlp7_left'])*F.linear(norm7,program['mlp7_right']);mlp7=F.linear(m,program['mlp7_down'])+program['mlp7_bias']
    mixed8=generated['other_sources']+program['attention7']['lambdas8'][0]*mlp7;rho=(mixed8.square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps).sqrt();head=program['head8']
    # The folded readers already supply the MLP7 contribution to the five
    # attention8 fields. Add full MLP7 only to the residual used for RMS8;
    # putting it in the linear source projection too would double count it.
    projected=(F.linear(generated['other_sources'],head['sources'])+program['attention7']['lambdas8'][0]*readers(program['readers'],norm7))/rho
    q1,k1,q2,k1v,value=projected.split(128,-1);t=projected.shape[1];inv=1/(10000**(torch.arange(0,128,2,dtype=torch.float32)/128));angles=torch.outer(torch.arange(t,dtype=torch.float32),inv);co,si=angles.cos().bfloat16().double(),angles.sin().bfloat16().double()
    def rotate(x):
        x=x/(x.square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps).sqrt();a,b=x.chunk(2,-1);return torch.cat([a*co+b*si,-a*si+b*co],-1)
    q1,k1,q2,k1v=[rotate(x) for x in [q1,k1,q2,k1v]];route=((q1*k1[:,city,None]).sum(-1)/128)*((q2*k1v[:,city,None]).sum(-1)/128);inherited=generated['first_values'][:,city,256:384].double();value=(1-head['mixture'].double())*value[:,city]+head['mixture'].double()*inherited;support=destination & (torch.arange(t)>=city);delta=-F.linear(route[...,None]*value[:,None],head['output'].double())*support[None,:,None]
    return (delta,rho) if return_rho else delta
