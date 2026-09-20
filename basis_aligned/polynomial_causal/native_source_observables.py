"""Explicit native suffix in float64, preserving float32 RMS epsilon and rounded RoPE."""
def source_observables(model,raw,x0,first,directions,positions,read,pairs,amplitudes):
    import torch
    import torch.nn.functional as F
    eps=torch.finfo(torch.float32).eps
    raw64=raw.double();x064=x0.double();first64=first.double();directions64=directions.double()
    batch=torch.arange(len(raw),device=raw.device);pos=positions
    def norm(x):return F.rms_norm(x,(x.shape[-1],),eps=eps)
    edit=torch.zeros_like(raw64);edit[batch,pos]=torch.einsum('bi,bid->bd',amplitudes.double(),directions64)
    x=raw64+edit
    for l in range(11,18):
        b=model.transformer.h[l]
        if l>11:x=b.lambdas[0].double()*x+b.lambdas[1].double()*x064
        att=b.attn;z=norm(x);bs,t,d=z.shape;heads=att.n_head;hd=att.head_dim
        projections=[F.linear(z,getattr(att,name).weight.double()).reshape(bs,t,heads,hd) for name in ['c_q','c_k','c_q2','c_k2','c_v']]
        q,k,q2,k2,value=projections
        cos,sin=att.rotary(torch.zeros(bs,t,heads,hd,device=x.device,dtype=torch.float32));cos=cos.double();sin=sin.double()
        def rotate(y):
            y=norm(y);y1,y2=y[...,:hd//2],y[...,hd//2:]
            return torch.cat([y1*cos+y2*sin,-y1*sin+y2*cos],dim=-1)
        q,k,q2,k2=[rotate(y) for y in [q,k,q2,k2]]
        scores=torch.einsum('bthd,bshd->bhts',q,k)/hd;scores2=torch.einsum('bthd,bshd->bhts',q2,k2)/hd
        mask=torch.ones(t,t,device=x.device,dtype=torch.bool).tril();pattern=(scores*scores2).masked_fill(~mask,0.)
        value=(1-att.lamb.double())*value+att.lamb.double()*first64.reshape_as(value)
        at=torch.einsum('bhts,bshd->bthd',pattern,value).reshape(bs,t,d)
        x=x+F.linear(at,att.c_proj.weight.double());z=norm(x)
        x=x+F.linear(F.linear(z,b.mlp.Left.weight.double())*F.linear(z,b.mlp.Right.weight.double()),b.mlp.Down.weight.double(),b.mlp.Down_bias.double())
    logits=30*torch.tanh((norm(x[batch,read])[:,None,None,:]*model.lm_head.weight[pairs].double()).sum(-1)/30)
    return logits[:,:,0]-logits[:,:,1]
