"""Fixed-context executable; only torch and declared program data required."""
from pathlib import Path
import torch

EPS=torch.finfo(torch.float32).eps
EXPS9=[(0,0),(1,0),(0,1),(1,1),(2,0),(2,1),(0,2),(1,2),(2,2)]
MAPS={'q1':'c_q','k1':'c_k','q2':'c_q2','k2':'c_k2','v':'c_v'}

def execute_source(program,a,b):
    ref=program['numerator']
    a,b=[torch.as_tensor(x,dtype=ref.dtype,device=ref.device) for x in [a,b]]
    if a.ndim or b.ndim:raise ValueError('Two scalar strengths required')
    m=torch.stack([torch.ones_like(a),a,b,a*b,a*a,a*a*b,b*b,a*b*b,a*a*b*b])
    residual=torch.einsum('...kd,k->...d',program['linear_basis'],m[:4])
    numerator=torch.einsum('...kd,k->...d',ref,m)
    return residual+numerator/(program['denominator']@m)[...,None]+program['bias']

def monomials(a,b,reference):
    a,b=[torch.as_tensor(x,dtype=reference.dtype,device=reference.device) for x in [a,b]]
    m9=torch.stack([a**i*b**j for i,j in EXPS9])
    m16=torch.stack([a**i*b**j for i in range(4) for j in range(4)])
    return m9,m16

def normalized_ports(program,a,b):
    m9,m16=monomials(a,b,program['q1'])
    rho=program['denominator']@m9
    if not bool((rho>0).all()):raise ValueError('Positive original RMS denominator required')
    r=torch.einsum('i,...ij,j->...',m16,program['norm_gram'],m16)+EPS*rho.square()
    if not bool((r>0).all()):raise ValueError('Nonpositive normalized-input radicand')
    ports={}
    for name in MAPS:
        raw=torch.einsum('...kd,k->...d',program[name],m16)
        raw=raw.reshape(*raw.shape[:-1],9,128)
        if name=='v':ports[name]=raw/r.sqrt()[...,None,None]
        else:ports[name]=raw/(raw.square().mean(-1,keepdim=True)+EPS*r[...,None,None]).sqrt()
    return ports

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

def execute_attention(program,a,b):
    first=program['first_values'];batch,tokens=program['q1'].shape[:2]
    if first.shape not in ((batch,tokens,1152),(batch,tokens,9,128)):
        raise ValueError('First values must be B,T,1152 or B,T,9,128')
    return attention_write(normalized_ports(program,a,b),program['output'],program['mixture'],first.reshape(batch,tokens,1152))

def load(directory=None):
    directory=Path(directory) if directory is not None else Path(__file__).parent
    return torch.load(directory/'program.pt',map_location='cpu',weights_only=True)


def execute(program,a,b):
    post9=execute_source(program['source'],a,b)
    attention10=execute_attention(program['attention'],a,b)
    alpha,beta=program['lambdas'].double()
    postattention10=alpha*post9+beta*program['x0'].double()+attention10
    return dict(post_mlp9=post9,attention10=attention10,post_attention10=postattention10)
