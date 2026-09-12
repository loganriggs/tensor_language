"""Exact current/first scalar-value sectors, with shared complete QK1*QK2 routing."""
import torch
import torch.nn.functional as F

def head_scalar_sectors(current,token_ids,p,index):
    width=p['q1'].shape[1];length=current.shape[1]
    projected=[F.rms_norm(F.linear(current,p[name][index]),(width,),eps=torch.finfo(torch.float32).eps) for name in ('q1','k1','q2','k2')]
    inv=1/(10000**(torch.arange(0,width,2,dtype=torch.float32)/width));angle=torch.outer(torch.arange(length,dtype=torch.float32),inv)
    cos=angle.cos().bfloat16().to(current.device);sin=angle.sin().bfloat16().to(current.device)
    def rotate(x):
        a,b=x.chunk(2,-1);return torch.cat([a*cos+b*sin,-a*sin+b*cos],-1).double()
    q,k,q2,k2=[rotate(x) for x in projected]
    gamma=(q@k.transpose(-1,-2)/width)*(q2@k2.transpose(-1,-2)/width)
    gamma=gamma.masked_fill(~torch.ones(length,length,dtype=torch.bool,device=current.device).tril(),0)
    values=torch.stack([current.double()@p['current_value_readers'][index],p['first_token_values'][token_ids,index]],-1)
    return gamma@values
