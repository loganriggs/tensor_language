"""Exact two-QK attention edge, bilinear in nonlinear query/key-value features.

Inputs are two affine token states at a causally ordered query/key pair.
Input/head RMS, rounded RoPE and the cached first value remain explicit.
Only the mixed interaction is localized to this edge for two distinct edited
positions. The caller must check the query occurs later than the key.
"""


def compile_core(weights,q0,qd,k0,kd,readers,first_value,mixture,heads,
                 qcos,qsin,kcos,ksin,input_eps,head_eps):
    import torch
    b,d=q0.shape;hd=d//heads
    def input_norm(x,v):
        return torch.stack([x.square().mean(-1)+input_eps,2*(x*v).mean(-1),v.square().mean(-1)],dim=-1)
    qs,ks=input_norm(q0,qd),input_norm(k0,kd)
    def branch(weight,x,v,s,cos,sin):
        base=(x@weight.T).reshape(b,heads,hd);linear=(v@weight.T).reshape(b,heads,hd)
        norm=torch.stack([base.square().mean(-1),2*(base*linear).mean(-1),linear.square().mean(-1)],dim=-1)+head_eps*s[:,None,:]
        def rotate(z):
            left,right=z.chunk(2,dim=-1)
            return torch.cat([left*cos[:,None,:]+right*sin[:,None,:],-left*sin[:,None,:]+right*cos[:,None,:]],dim=-1)
        return torch.stack([rotate(base),rotate(linear)],dim=2),norm
    scores=[];qnorm=[];knorm=[]
    for qname,kname in [('q','k'),('q2','k2')]:
        q,qn=branch(weights[qname],q0,qd,qs,qcos,qsin)
        k,kn=branch(weights[kname],k0,kd,ks,kcos,ksin)
        scores.append(torch.einsum('bhid,bhjd->bhij',q,k)/hd);qnorm.append(qn);knorm.append(kn)
    product=torch.zeros(b,heads,3,3,dtype=q0.dtype,device=q0.device)
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for l in range(2):product[:,:,i+k,j+l]+=scores[0][:,:,i,j]*scores[1][:,:,k,l]
    output=torch.einsum('bod,dhk->bohk',readers,weights['o'].reshape(d,heads,hd))
    vb=(k0@weights['v'].T).reshape(b,heads,hd);vl=(kd@weights['v'].T).reshape(b,heads,hd)
    return dict(product=product,qnorm=torch.stack(qnorm,dim=2),knorm=torch.stack(knorm,dim=2),
                input_norm=ks,value_base=(1-mixture)*torch.einsum('bohd,bhd->boh',output,vb),
                value_linear=(1-mixture)*torch.einsum('bohd,bhd->boh',output,vl),
                cached_value=mixture*torch.einsum('bohd,bhd->boh',output,first_value.reshape(b,heads,hd)))


def features(core,u,v):
    import torch
    anchor=core['product'];b=len(anchor)
    u=torch.as_tensor(u,dtype=anchor.dtype,device=anchor.device).reshape(-1).expand(b)
    v=torch.as_tensor(v,dtype=anchor.dtype,device=anchor.device).reshape(-1).expand(b)
    def normalization(coefficients,t):
        values=coefficients[...,0]+t[:,None,None]*coefficients[...,1]+t[:,None,None].square()*coefficients[...,2]
        return values.prod(-1).sqrt()
    ub=torch.stack([torch.ones_like(u),u,u*u],dim=-1);vb=torch.stack([torch.ones_like(v),v,v*v],dim=-1)
    query=ub[:,None,:]/normalization(core['qnorm'],u)[...,None]
    norm=core['input_norm'];vs=(norm[:,0]+v*norm[:,1]+v*v*norm[:,2]).sqrt()
    value=(core['value_base']+v[:,None,None]*core['value_linear'])/vs[:,None,None]+core['cached_value']
    key_value=value[...,None]*vb[:,None,None,:]/normalization(core['knorm'],v)[:,None,:,None]
    return query,key_value


def execute(core,u,v):
    import torch
    query,key_value=features(core,u,v)
    return torch.einsum('bhij,bhi,bohj->bo',core['product'],query,key_value)


def mixed(core,u,v):
    import torch
    query,key_value=features(core,u,v);q0,k0=features(core,0.,0.)
    return torch.einsum('bhij,bhi,bohj->bo',core['product'],query-q0,key_value-k0)
