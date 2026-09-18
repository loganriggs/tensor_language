"""Execute physically packed attention7 head maps and MLP7 inputs from residual6, using native factors."""
import torch
import torch.nn.functional as F

def execute(program,residual6,token_ids,heads=None):
    assert residual6.shape[:2]==token_ids.shape
    lookup={int(t):i for i,t in enumerate(program['token_ids'].tolist())}
    try:indices=[[lookup[int(t)] for t in row] for row in token_ids.tolist()]
    except KeyError as e:raise ValueError('Token outside frozen generator vocabulary') from e
    index=torch.tensor(indices,device=residual6.device);initial=F.embedding(index,program['initial_table']);first=F.embedding(index,program['first_table'])
    lam=program['lambdas7'];mixed=lam[0]*residual6+lam[1]*initial;current=F.rms_norm(mixed,(1152,));b,t,_=current.shape
    heads=tuple(program['head_ids'].tolist())
    if len(set(heads))!=len(heads) or any(h not in range(9) for h in heads):raise ValueError('Invalid attention7 heads')
    indices=torch.tensor([128*h+j for h in heads for j in range(128)],device=current.device,dtype=torch.long);n=len(heads)
    if not n:
        lam8=program['lambdas8']
        return {'normalized_mlp7':F.rms_norm(mixed,(1152,)),'other_sources':lam8[0]*mixed+lam8[1]*initial,'first_values':first}
    inv=1/(10000**(torch.arange(0,128,2,dtype=torch.float32,device=current.device)/128));angles=torch.outer(torch.arange(t,dtype=torch.float32,device=current.device),inv)
    co,si=angles.cos().bfloat16().float()[None,:,None],angles.sin().bfloat16().float()[None,:,None]
    def factor(name):
        x=F.rms_norm(F.linear(current,program[name]).reshape(b,t,n,128),(128,));a,z=x.chunk(2,-1)
        return torch.cat([a*co+z*si,-a*si+z*co],-1).to(current.dtype)
    q1,k1,q2,k2=[factor(k) for k in ['q1','k1','q2','k2']]
    pattern=(torch.einsum('bqhd,bshd->bhqs',q1,k1)/128)*(torch.einsum('bqhd,bshd->bhqs',q2,k2)/128)
    pattern=pattern.masked_fill(~torch.ones(t,t,dtype=torch.bool,device=current.device).tril(),0)
    value=(1-program['mixture'])*F.linear(current,program['value'])+program['mixture']*first[:,:,indices]
    channels=torch.einsum('bhqs,bshd->bqhd',pattern,value.reshape(b,t,n,128)).reshape(b,t,n*128)
    attention=F.linear(channels,program['output']);post=mixed+attention;lam8=program['lambdas8']
    return {'normalized_mlp7':F.rms_norm(post,(1152,)),'other_sources':lam8[0]*post+lam8[1]*initial,'first_values':first}
