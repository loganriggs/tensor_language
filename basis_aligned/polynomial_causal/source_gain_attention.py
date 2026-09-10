"""Query-edge source gains: exact-real quadratic numerator and explicit RMS factors."""
from contextlib import contextmanager
import torch
import torch.nn.functional as F
from projected_query_source_edits import source_coefficients


def rotate(x, cos, sin):
    a,b=x.chunk(2,dim=-1)
    return torch.cat((a*cos+b*sin,-a*sin+b*cos),dim=-1)


def prepare(sources, readers, keys, values, cos, sin, positions, eps):
    # sources B,S,D; readers I,J,H,D; keys B,I,T,J,H; values B,T,J,H.
    b,s,d=sources.shape;h=readers.shape[-2]
    p=torch.einsum('bsd,ijhd->bsijh',sources,readers)
    gram=sources@sources.transpose(-1,-2)
    norm=torch.einsum('bsijh,btijh->bijst',p,p)/h+eps*gram[:,None,None]/d
    rotated=rotate(p,cos[:,None,None,None],sin[:,None,None,None])
    factors=torch.einsum('bsijh,bitjh->bijst',rotated,keys)
    mask=torch.arange(values.shape[1],device=values.device)[None,:]<=torch.as_tensor(positions,device=values.device)[:,None]
    factors=factors*mask[:,None,None,None,:]
    return {'factors':factors,'norm':norm,'gram':gram,'values':values,'eps2':eps**2,'width':h}


def evaluate(bank,gains):
    rho=torch.einsum('bs,bst,bt->b',gains,bank['gram'],gains)
    assert bool((rho>=0).all()), 'negative source Gram norm; no clipping allowed'
    n=torch.einsum('bs,bijst,bt->bij',gains,bank['norm'],gains)+bank['eps2']
    assert bool((n>0).all())
    scores=torch.einsum('bs,bijst->bijt',gains,bank['factors'])
    pattern=scores[:,0]*scores[:,1]/(n[:,0]*n[:,1]).sqrt()[:,:,None]/bank['width']**2
    return torch.einsum('bjt,btjh->bjh',pattern,bank['values'])


def direct(sources,readers,keys,values,cos,sin,positions,gains,eps):
    u=torch.einsum('bs,bsd->bd',gains,sources)
    q=torch.einsum('bd,ijhd->bijh',F.rms_norm(u,(u.shape[-1],),eps=eps),readers)
    q=F.rms_norm(q,(q.shape[-1],),eps=eps)
    q=rotate(q,cos[:,None,None],sin[:,None,None])
    scores=torch.einsum('bijh,bitjh->bijt',q,keys)
    mask=torch.arange(values.shape[1],device=values.device)[None,:]<=torch.as_tensor(positions,device=values.device)[:,None]
    pattern=scores[:,0]*scores[:,1]/q.shape[-1]**2*mask[:,None]
    return torch.einsum('bjt,btjh->bjh',pattern,values)


@contextmanager
def install_reads(attn,read,positions,heads=(1,4)):
    def hook(_m,args):
        x=args[0].clone();view=x.view(x.shape[0],x.shape[1],attn.n_head,attn.head_dim)
        for i,pos in enumerate(positions):view[i,int(pos),list(heads)]=read[i].to(view)
        return (x,)+args[1:]
    h=attn.c_proj.register_forward_pre_hook(hook)
    try:yield
    finally:h.remove()


@contextmanager
def capture(model,positions,layer=9,heads=(1,4)):
    """Observe native sources and selected attention inputs; manual backend compatible."""
    handles=[];raw={};record={};attn=model.transformer.h[layer].attn
    original=attn.squared_attention;had='squared_attention' in attn.__dict__
    endpoint=lambda x:torch.stack([x[i,int(pos)] for i,pos in enumerate(positions)]).detach().clone()
    def save(key,tuple_output=False):
        def hook(_m,_a,out):raw[key]=endpoint(out[0] if tuple_output else out)
        return hook
    try:
        handles.append(model.transformer.wte.register_forward_hook(save('embedding')))
        for j in range(layer):
            handles.append(model.transformer.h[j].attn.register_forward_hook(save(('a',j),True)))
            handles.append(model.transformer.h[j].mlp.register_forward_hook(save(('m',j))))
        handles.append(attn.register_forward_pre_hook(lambda _m,args:record.update(normalized_input=endpoint(args[0]))))
        def observe(q,k,v,q2,k2):
            y=original(q,k,v,q2,k2)
            record['keys']=torch.stack((k[:,:,list(heads)],k2[:,:,list(heads)]),dim=1).detach().double()
            record['values']=v[:,:,list(heads)].detach().double()
            record['native_read']=torch.stack([y[i,list(heads),int(pos)] for i,pos in enumerate(positions)]).detach().double()
            return y
        attn.squared_attention=observe
        yield record
        lam=[b.lambdas.detach().double().cpu().tolist() for b in model.transformer.h[:layer+1]]
        alpha,coeff=source_coefficients(lam)
        e=F.rms_norm(raw['embedding'],(raw['embedding'].shape[-1],)).double()
        record['sources']=torch.stack([alpha*e]+[coeff[j]*raw[k,j].double() for j in range(layer) for k in ('a','m')],dim=1)
        record['readers']=torch.stack([torch.stack([p.weight[h*attn.head_dim:(h+1)*attn.head_dim].detach().double() for h in heads]) for p in (attn.c_q,attn.c_q2)])
        record['cos']=attn.rotary.cos_cached[0,:,0,:][list(positions)].double() if attn.rotary.cos_cached.ndim==4 else attn.rotary.cos_cached[list(positions)].double()
        record['sin']=attn.rotary.sin_cached[0,:,0,:][list(positions)].double() if attn.rotary.sin_cached.ndim==4 else attn.rotary.sin_cached[list(positions)].double()
    finally:
        for h in handles:h.remove()
        if had:attn.squared_attention=original
        else:del attn.squared_attention


def controls():
    torch.set_num_threads(2);g=torch.Generator().manual_seed(60924)
    rand=lambda *s:torch.randn(*s,generator=g,dtype=torch.float64)
    sources=rand(2,5,12);readers=.1*rand(2,2,4,12);keys=rand(2,2,7,2,4);values=rand(2,7,2,4)
    # Deliberately nonorthogonal rounded table, unequal query positions, nonzero eps.
    cos=rand(2,2).bfloat16().double();sin=rand(2,2).bfloat16().double();positions=[3,6];eps=.03
    bank=prepare(sources,readers,keys,values,cos,sin,positions,eps)
    masks=torch.cat((torch.ones(1,5),torch.zeros(1,5),torch.eye(5),1-torch.eye(5),torch.tensor([[1.,-.5,.3,0,1]]))).double()
    errors=[]
    for mask in masks:
        z=mask.expand(2,-1)
        errors.append(float((evaluate(bank,z)-direct(sources,readers,keys,values,cos,sin,positions,z,eps)).abs().max()))
    bad=dict(bank);bad['norm']=torch.diag_embed(torch.diagonal(bank['norm'],dim1=-2,dim2=-1))
    coupling=float((evaluate(bank,torch.ones(2,5,dtype=torch.float64))-evaluate(bad,torch.ones(2,5,dtype=torch.float64))).norm())
    checks={'all_gain_oracles':max(errors)<1e-11,'zero_gain_read':bool((evaluate(bank,torch.zeros(2,5,dtype=torch.float64))==0).all()),'cross_source_denominator_live':coupling>.01,'rounded_rotation_nonorthogonal':float((cos.square()+sin.square()-1).abs().max())>.1}
    return {'passed':all(checks.values()),'checks':checks,'max_abs_error':max(errors),'discard_cross_terms_error':coupling,'model_forwards':0}
