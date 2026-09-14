"""Shared interaction executor. Requires torch and explicit bank/context data."""
from pathlib import Path
import torch
import torch.nn.functional as F
EPS=torch.finfo(torch.float32).eps
MAPS={'q1': 'c_q', 'k1': 'c_k', 'q2': 'c_q2', 'k2': 'c_k2', 'v': 'c_v'}

def execute_source(program,a,b):
    ref=program['numerator']
    a,b=[torch.as_tensor(x,dtype=ref.dtype,device=ref.device) for x in [a,b]]
    if a.ndim or b.ndim:raise ValueError('Two scalar strengths required')
    m=torch.stack([torch.ones_like(a),a,b,a*b,a*a,a*a*b,b*b,a*b*b,a*a*b*b])
    residual=torch.einsum('...kd,k->...d',program['linear_basis'],m[:4])
    numerator=torch.einsum('...kd,k->...d',ref,m)
    return residual+numerator/(program['denominator']@m)[...,None]+program['bias']

def attention_write(ports,output,mixture,first_values):
    q1,k1,q2,k2=[ports[k] for k in ['q1','k1','q2','k2']]
    n=q1.shape[1];device=q1.device
    inv=1/(10000**(torch.arange(0,128,2,dtype=torch.float32)/128))
    angles=torch.outer(torch.arange(n,dtype=torch.float32),inv)
    co,si=angles.cos().bfloat16().to(device)[None,:,None,:],angles.sin().bfloat16().to(device)[None,:,None,:]
    def rotate(x):
        a,b=x.chunk(2,-1);return torch.cat((a*co+b*si,-a*si+b*co),-1)
    q1,k1,q2,k2=[rotate(x) for x in [q1,k1,q2,k2]]
    pattern=(torch.einsum('bthd,bshd->bhts',q1,k1)/128)*(torch.einsum('bthd,bshd->bhts',q2,k2)/128)
    pattern=pattern.masked_fill(~torch.ones(n,n,dtype=torch.bool,device=device).tril(),0)
    lam=mixture.double();v=(1-lam)*ports['v']+lam*first_values.double().reshape_as(ports['v'])
    channels=torch.einsum('bhts,bshd->bthd',pattern,v).reshape(*first_values.shape)
    return channels@output.double().T

def load(directory=None):
    directory=Path(directory) if directory is not None else Path(__file__).parent
    return torch.load(directory/'bank.pt',map_location='cpu',weights_only=True)


def execute(bank,context,a,b):
    source=dict(context['source'],bias=bank['bias'])
    post9=execute_source(source,a,b)
    batch,tokens,width=post9.shape
    if width!=1152 or context['x0'].shape!=post9.shape:
        raise ValueError('Source and x0 must have matching B,T,1152 shapes')
    first=context['first_values']
    if first.shape not in ((batch,tokens,1152),(batch,tokens,9,128)):
        raise ValueError('First values must have B,T,1152 or B,T,9,128 shape')
    alpha,beta=bank['lambdas'].double()
    raw=alpha*post9+beta*context['x0'].double()
    normalized=F.rms_norm(raw,(1152,),eps=EPS)
    weights=bank['weights'];ports={}
    for name,native in MAPS.items():
        value=(normalized@weights[native].double().T).reshape(batch,tokens,9,128)
        ports[name]=value if name=='v' else F.rms_norm(value,(128,),eps=EPS)
    write=attention_write(ports,weights['c_proj'],weights['lamb'],first.reshape(batch,tokens,1152))
    return dict(post_mlp9=post9,attention10=write,post_attention10=raw+write)
