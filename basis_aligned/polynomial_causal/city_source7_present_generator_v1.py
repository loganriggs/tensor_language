"""MLP7-present city numerator terms with full queries/normalizers held fixed."""
import torch
import torch.nn.functional as F
from city_attention7_full_v1 import execute as attention7
from city_mlp7_readers_v1 import execute as readers

def from_projected(projected,mlp_projected,inherited,head,city,destination):
    eps=torch.finfo(torch.float32).eps;t=projected.shape[1]
    q1,k1,q2,k2,value=projected.double().split(128,-1)
    _,mk1,_,mk2,mv=mlp_projected.double().split(128,-1)
    inv=1/(10000**(torch.arange(0,128,2,dtype=torch.float32,device=projected.device)/128))
    angles=torch.outer(torch.arange(t,dtype=torch.float32,device=projected.device),inv)
    co,si=angles.cos().bfloat16().double(),angles.sin().bfloat16().double()
    def rotate(x):
        a,b=x.chunk(2,-1);return torch.cat([a*co+b*si,-a*si+b*co],-1)
    q1=rotate(q1/(q1.square().mean(-1,keepdim=True)+eps).sqrt())
    q2=rotate(q2/(q2.square().mean(-1,keepdim=True)+eps).sqrt())
    den1=(k1.square().mean(-1,keepdim=True)+eps).sqrt();den2=(k2.square().mean(-1,keepdim=True)+eps).sqrt()
    keys1=[rotate(k1/den1),rotate((k1-mk1)/den1)]
    keys2=[rotate(k2/den2),rotate((k2-mk2)/den2)]
    mask=destination & (torch.arange(t,device=projected.device)>=city);writes=[]
    for a,b,v in zip(keys1,keys2,[value,value-mv],strict=True):
        route=((q1*a[:,city,None]).sum(-1)/128)*((q2*b[:,city,None]).sum(-1)/128)
        val=(1-head['mixture'].double())*v[:,city]+head['mixture'].double()*inherited.double()
        writes.append(-F.linear(route[...,None]*val[:,None],head['output'].double())*mask[None,:,None])
    return writes[0],writes[0]-writes[1],torch.stack([q1,q2],2)

def execute(program,residual6,token_ids,city,destination,return_debug=False):
    generated=attention7(program['attention7'],residual6,token_ids);other=generated['other_sources'].double()
    rho=(other.square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps).sqrt()
    contribution=program['attention7']['lambdas8'][0].double()*readers(program['readers'],generated['normalized_mlp7']).double()/rho
    projected=F.linear(other,program['head8']['sources'].double())/rho+contribution
    result=from_projected(projected,contribution,generated['first_values'][:,city,256:384],program['head8'],city,destination)
    return result if return_debug else result[1]
