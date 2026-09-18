"""Native attention7 city output from residual6 prefix and token IDs; PyTorch only."""
import torch
import torch.nn.functional as F

def execute(program,residual6,token_ids,city):
    assert residual6.shape[:2]==token_ids.shape and 0<=city<token_ids.shape[1]
    residual6=residual6[:,:city+1];token_ids=token_ids[:,:city+1]
    lookup={int(t):i for i,t in enumerate(program['token_ids'].tolist())}
    try:indices=[[lookup[int(t)] for t in row] for row in token_ids.tolist()]
    except KeyError as e:raise ValueError('Token outside frozen generator vocabulary') from e
    index=torch.tensor(indices,device=residual6.device)
    initial=F.embedding(index,program['initial_table']);first=F.embedding(index,program['first_table'])
    lam=program['lambdas7'];mixed=lam[0]*residual6+lam[1]*initial;current=F.rms_norm(mixed,(1152,))
    b,t,_=current.shape
    inv=1/(10000**(torch.arange(0,128,2,dtype=torch.float32,device=current.device)/128))
    angles=torch.outer(torch.arange(t,dtype=torch.float32,device=current.device),inv)
    co,si=angles.cos().bfloat16().float()[None,:,None],angles.sin().bfloat16().float()[None,:,None]
    def factor(name,query=False):
        x=current[:,city:city+1] if query else current
        x=F.rms_norm(F.linear(x,program[name]).reshape(b,x.shape[1],9,128),(128,))
        c,s=(co[:,city:city+1],si[:,city:city+1]) if query else (co,si)
        a,z=x.chunk(2,-1);return torch.cat([a*c+z*s,-a*s+z*c],-1).to(current.dtype)
    q1,k1,q2,k2=factor('q1',True),factor('k1'),factor('q2',True),factor('k2')
    scores=torch.einsum('bqhd,bshd->bhqs',q1,k1)/128
    scores2=torch.einsum('bqhd,bshd->bhqs',q2,k2)/128
    value=(1-program['mixture'])*F.linear(current,program['value'])+program['mixture']*first
    channels=torch.einsum('bhqs,bshd->bqhd',scores*scores2,value.reshape(b,t,9,128)).reshape(b,1,1152)
    attention=F.linear(channels,program['output'])[:,0]
    post=mixed[:,city]+attention;lam8=program['lambdas8']
    return {'attention7_city':attention,'normalized_mlp7_city':F.rms_norm(post,(1152,)),
            'other_city_sources':lam8[0]*post+lam8[1]*initial[:,city]}
