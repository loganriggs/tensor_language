"""Shared even-key routing for all 128 value channels of native head9.8.

Conditional normalized current and first-layer inputs are supplied externally.
The reflection keeps original key denominators, as in KeySpanParent.
"""
import torch
import torch.nn.functional as F


def compile_program(parent, state):
    head = slice(8*128,9*128)
    p = {k:parent[k][1].clone() for k in ['q1','q2','k1','k2','key_coordinates']}
    p.update(current_value=state['transformer.h.9.attn.c_v.weight'][head].clone(),
             first_value=state['transformer.h.0.attn.c_v.weight'][head].clone(),
             output=state['transformer.h.9.attn.c_proj.weight'][:,head].clone(),
             mixture=state['transformer.h.9.attn.lamb'].clone())
    return p


def routing(current, p):
    x = current.double(); n = current.shape[-2]
    keys = torch.cat([p['k1'],p['k2']]).double()
    adapter = (keys@keys.T)@p['key_coordinates']
    nums = [x@p[k].double().T for k in ['k1','k2']]
    shared = torch.cat(nums,-1)@p['key_coordinates']
    inv = 1/(10000**(torch.arange(0,128,2,dtype=torch.float32)/128))
    angle = torch.outer(torch.arange(n,dtype=torch.float32),inv)
    co,si = angle.cos().bfloat16(),angle.sin().bfloat16()
    def rotate(x):
        a,b=x.chunk(2,-1)
        return torch.cat((a*co+b*si,-a*si+b*co),-1)
    full,reflected=[],[]
    for j,(qn,kn) in enumerate([('q1','k1'),('q2','k2')]):
        q=rotate(F.rms_norm(F.linear(current,p[qn].to(current.dtype)),(128,),eps=torch.finfo(torch.float32).eps)).double()
        k=F.linear(current,p[kn].to(current.dtype))
        den=(k.square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps).sqrt().double()
        inside=shared@adapter[j*128:(j+1)*128].T
        full.append(q@rotate(nums[j]/den).transpose(-1,-2)/128)
        reflected.append(q@rotate((nums[j]-2*inside)/den).transpose(-1,-2)/128)
    gamma=(full[0]*full[1]+reflected[0]*reflected[1])/2
    return gamma.masked_fill(~torch.ones(n,n,dtype=torch.bool).tril(),0)


def execute(current, initial, p):
    gamma=routing(current,p)
    current_values=current.double()@p['current_value'].double().T
    first_values=initial.double()@p['first_value'].double().T
    mixture=p['mixture'].double()
    values=(1-mixture)*current_values+mixture*first_values
    channels=gamma@values
    return channels@p['output'].double().T, channels
