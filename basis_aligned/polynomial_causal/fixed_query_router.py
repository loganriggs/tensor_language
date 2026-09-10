"""Endpoint-only fixed-query interface and normalized folded key scorer."""
from contextlib import contextmanager
import torch


def override(output, prototypes, positions, heads, width):
    z=output.clone().view(output.shape[0],output.shape[1],-1,width)
    for i,pos in enumerate(positions):
        z[i,int(pos),list(heads)]=prototypes.to(z)
    return z.reshape_as(output)


@contextmanager
def install(attn, positions, heads=(1,4), prototypes=None, remove=False):
    handles=[]
    try:
        if prototypes is not None:
            for kind,target in enumerate((attn.c_q,attn.c_q2)):
                def hook(_m,_args,out,kind=kind):return override(out,prototypes[kind],positions,heads,attn.head_dim)
                handles.append(target.register_forward_hook(hook))
        if remove:
            def zero(_m,args):
                p=torch.zeros(len(heads),attn.head_dim,device=args[0].device,dtype=args[0].dtype)
                return (override(args[0],p,positions,heads,attn.head_dim),)+args[1:]
            handles.append(attn.c_proj.register_forward_pre_hook(zero))
        yield
    finally:
        for h in handles:h.remove()


def folded_pattern(x, query, keys, cos, sin, positions, epsilon):
    """FP64 formula with the actual rotary coefficients, no orthogonality premise.

    x[B,T,D], query[2,H], keys[2,H,D], cos/sin[T,H/2].
    Returns the selected head's endpoint pattern at every source; mask separately.
    """
    x=x.double();query=query.double();keys=keys.double();cos=cos.double();sin=sin.double()
    width=query.shape[-1];half=width//2;positions=torch.as_tensor(positions,device=x.device)
    terms=[]
    for q,w in zip(query,keys):
        q=q/(q.square().mean()+epsilon).sqrt()
        qt1=q[:half]*cos[positions]+q[half:]*sin[positions]
        qt2=-q[:half]*sin[positions]+q[half:]*cos[positions]
        # R_s^T R_t q; transpose is lawful even for rounded nonorthogonal tables.
        qs=torch.cat([cos[None]*qt1[:,None]-sin[None]*qt2[:,None],
                      sin[None]*qt1[:,None]+cos[None]*qt2[:,None]],-1)
        direction=qs@w
        numerator=(direction*x).sum(-1)
        key=x@w.T;denominator=(key.square().mean(-1)+epsilon).sqrt()
        terms.append(numerator/denominator)
    return terms[0]*terms[1]/width**2


def controls():
    from types import SimpleNamespace
    dtype=torch.float64;g=torch.Generator().manual_seed(60921)
    x=torch.randn(2,3,5,generator=g,dtype=dtype);q=torch.randn(2,4,generator=g,dtype=dtype)
    keys=torch.randn(2,4,5,generator=g,dtype=dtype)
    cos=torch.tensor([[1.,1.],[.7,.6],[.4,.3]],dtype=dtype);sin=torch.tensor([[0.,0.],[.6,.7],[.8,.9]],dtype=dtype)
    positions=[1,2];epsilon=.02
    actual=folded_pattern(x,q,keys,cos,sin,positions,epsilon)
    def rotate(a,c,s):return torch.cat([a[...,:2]*c+a[...,2:]*s,-a[...,:2]*s+a[...,2:]*c],-1)
    direct=[];unnormalized=[]
    for qi,w in zip(q,keys):
        qi=qi/(qi.square().mean()+epsilon).sqrt();raw=x@w.T
        kr=rotate(raw/(raw.square().mean(-1,keepdim=True)+epsilon).sqrt(),cos[None],sin[None])
        qr=rotate(qi.expand(2,4),cos[positions],sin[positions])
        direct.append((qr[:,None]*kr).sum(-1))
        unnormalized.append((qr[:,None]*rotate(raw,cos[None],sin[None])).sum(-1))
    expected=direct[0]*direct[1]/16;wrong=unnormalized[0]*unnormalized[1]/16
    z=torch.arange(2*3*12,dtype=dtype).reshape(2,3,12);p=torch.ones(2,4,dtype=dtype)*7
    edited=override(z,p,positions,(0,2),4);mask=torch.zeros_like(z,dtype=torch.bool)
    for i,t in enumerate(positions):mask[i,t,:4]=True;mask[i,t,8:]=True
    attn=SimpleNamespace(c_q=torch.nn.Identity(),c_q2=torch.nn.Identity(),c_proj=torch.nn.Identity(),head_dim=4)
    with install(attn,positions,(0,2),torch.stack([p,p*2])):hooked=attn.c_q(z)
    try:
        with install(attn,positions,(0,2),torch.stack([p,p*2])):raise RuntimeError('fixture')
    except RuntimeError:pass
    with install(attn,positions,(0,2),remove=True):removed=attn.c_proj(z)
    checks={'nonorthogonal_rotary_fold':bool(torch.allclose(actual,expected,atol=1e-12,rtol=1e-12)),
        'key_normalization_is_live':float((actual-wrong).abs().max())>.01,
        'selected_endpoints_only':bool(torch.equal(edited[~mask],z[~mask]) and torch.all(edited[mask]==7)),
        'native_projection_hook':bool(torch.equal(hooked,edited)),
        'selected_head_removal':bool(torch.equal(removed[~mask],z[~mask]) and torch.all(removed[mask]==0)),
        'normal_and_exception_restoration':all(not m._forward_hooks and not m._forward_pre_hooks for m in (attn.c_q,attn.c_q2,attn.c_proj))}
    return {'passed':all(checks.values()),'checks':checks,'fold_max_abs':float((actual-expected).abs().max())}
