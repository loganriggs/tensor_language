"""Model-independent recent-path executor with explicit normalization/attention.

Ports: pre-MLP16 residual h16, normalized embedding x0, first-layer value v1.
These upstream ports are still required. No full expanded tensor is built.
"""
import torch
import torch.nn.functional as F


def load_package(directory,device='cpu'):
    """Load only the three extracted shards, verifying all manifest files."""
    import hashlib,json
    from pathlib import Path
    directory=Path(directory)
    manifest=json.loads((directory/'manifest.json').read_text())
    weights={}
    for name,metadata in manifest.items():
        path=directory/name
        if hashlib.sha256(path.read_bytes()).hexdigest()!=metadata['sha256']:
            raise ValueError(f'package hash mismatch: {name}')
        if name.endswith('.pt'):
            shard=torch.load(path,map_location=device,weights_only=True)
            if set(shard)&set(weights):raise ValueError('duplicate weight in package')
            weights.update(shard)
    return weights


def attention(x, v1, w):
    b,t,d=x.shape;heads=w['heads'];hd=d//heads
    def project(name):return (x@w[name].T).reshape(b,t,heads,hd)
    q,k,q2,k2,v=[project(name) for name in ['q','k','q2','k2','v']]
    v=(1-w['mixture'])*v+w['mixture']*v1.reshape_as(v)
    # Match native Rotary: frequencies computed on CPU, cos/sin stored bf16.
    frequency=torch.outer(torch.arange(t,dtype=torch.float32),w['inv_freq'].cpu())
    cos=frequency.cos().bfloat16().to(x.device)[None,:,None,:]
    sin=frequency.sin().bfloat16().to(x.device)[None,:,None,:]
    def rope(a):
        a=F.rms_norm(a,(hd,));left,right=a.chunk(2,-1)
        return torch.cat([left*cos+right*sin,-left*sin+right*cos],-1).to(a.dtype)
    q,k,q2,k2=map(rope,(q,k,q2,k2))
    scores=torch.einsum('bthd,bshd->bhts',q,k)/hd
    scores2=torch.einsum('bthd,bshd->bhts',q2,k2)/hd
    pattern=scores*scores2
    mask=torch.ones(t,t,dtype=torch.bool,device=x.device).tril()
    pattern=pattern.masked_fill(~mask,0)
    values=torch.einsum('bhts,bshd->bhtd',pattern,v).transpose(1,2).contiguous().reshape(b,t,d)
    return values@w['o'].T


def execute(w,h16,x0,v1,scale=1.):
    n16=F.rms_norm(h16,(h16.shape[-1],))
    if 'input_basis16' in w:
        shared=n16@w['input_basis16']
        left=shared@w['left16'].T;right=shared@w['right16'].T
    elif 'left_basis16' in w:
        left=(n16@w['left_basis16'])@w['left16'].T
        right=(n16@w['right_basis16'])@w['right16'].T
    else:
        left=n16@w['left16'].T;right=n16@w['right16'].T
    products=left*right
    write16=products@w['down16'].T
    if 'output_basis16' in w:write16=write16@w['output_basis16'].T
    write16=write16+w['bias16']
    live=w['residual17']*(h16+scale*write16)+w['embedding17']*x0
    attn=attention(F.rms_norm(live,(live.shape[-1],)),v1,w)
    h17=live+attn
    # Pull parent readers back through the preceding Down projection.
    projected_readers=w['output_basis16'].T@w['readers'] if 'output_basis16' in w else w['readers']
    folded=w['residual17']*(w['down16'].T@projected_readers)
    gated_features=scale*(products@folded)
    background=w['residual17']*h16+w['embedding17']*x0+attn+scale*w['residual17']*w['bias16']
    background_features=background@w['readers']
    denominator=h17.square().mean(-1)+torch.finfo(h17.dtype).eps
    c=w['coefficients']
    terms=torch.stack([(background_features.square()*c).sum(-1),
        (2*background_features*gated_features*c).sum(-1),
        (gated_features.square()*c).sum(-1)],-1)/denominator[...,None]
    return dict(alpha=terms.sum(-1),terms=terms,h17=h17,attention17=attn,write16=write16)


def extract(model, component):
    b16,b17=model.transformer.h[16],model.transformer.h[17]
    if model.config.gated or not b17.attn.squared_attn:
        raise ValueError('requires ungated bilinear MLP and product attention')
    w=dict(left16=b16.mlp.Left.weight,right16=b16.mlp.Right.weight,
        down16=b16.mlp.Down.weight,bias16=b16.mlp.Down_bias,
        residual17=b17.lambdas[0],embedding17=b17.lambdas[1],
        mixture=b17.attn.lamb,heads=b17.attn.n_head,inv_freq=b17.attn.rotary.inv_freq,
        readers=component['readers'],coefficients=component['coefficients'],
        residual_writer=component['residual_writer'],vocabulary_writer=component['vocabulary_writer'])
    for short,name in [('q','c_q'),('k','c_k'),('q2','c_q2'),('k2','c_k2'),('v','c_v'),('o','c_proj')]:
        w[short]=getattr(b17.attn,name).weight
    return {k:v.detach().float().clone() if torch.is_tensor(v) else v for k,v in w.items()}
