"""Native-weight, city-token-conditional closed face; PyTorch only."""
import torch
import torch.nn.functional as F


def routing(p, current, city_state, city):
    b,t,d = current.shape
    inv=1/(10000**(torch.arange(0,128,2,dtype=torch.float32,device=current.device)/128))
    angles=torch.outer(torch.arange(t,dtype=torch.float32,device=current.device),inv)
    co,si=angles.cos().bfloat16(),angles.sin().bfloat16()
    def rotate(x):
        left,right=x.chunk(2,-1)
        return torch.cat((left*co+right*si,-left*si+right*co),-1).type_as(x)
    # Preserve native full-sequence normalization/rotation before selecting city.
    key_current=current.clone()
    key_current[:,city]=city_state
    factors=[]
    for qn,kn in [('q1','k1'),('q2','k2')]:
        q=rotate(F.rms_norm(F.linear(current,p[qn].to(current.dtype)),(128,)))
        k=rotate(F.rms_norm(F.linear(key_current,p[kn].to(current.dtype)),(128,)))
        factors.append(torch.einsum('bqd,bkd->bqk',q,k)[:,:,city]/128)
    route=factors[0]*factors[1]
    return route.masked_fill(torch.arange(t,device=current.device)[None,:]<city,0)


def inherited(p, token):
    ids=p['token_ids'].tolist()
    if token not in ids:
        raise ValueError('City token outside declared program vocabulary')
    return p['first_table'][ids.index(token)][None]


def execute(p, current, donor_city_state, recipient_token, donor_token, city, destination):
    a0=routing(p,current,current[:,city],city)
    a1=routing(p,current,donor_city_state,city)
    # Full sequence projection preserves the native GEMM shape for replay.
    c=F.linear(current,p['current_value'].to(current.dtype))[:,city]
    i0=inherited(p,recipient_token).to(current.dtype)
    i1=inherited(p,donor_token).to(current.dtype)
    lam=p['mixture']
    v0=(1-lam)*c+lam*i0
    v1=(1-lam)*c+lam*i1
    channels=a1[...,None]*v1[:,None]-a0[...,None]*v0[:,None]
    return F.linear(channels,p['output'].to(current.dtype))*destination[None,:,None]
