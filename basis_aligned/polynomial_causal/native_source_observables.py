"""Explicit native float64 suffix, preserving float32 RMS epsilon and rounded RoPE."""
def norm64(x):
    import torch
    import torch.nn.functional as F
    return F.rms_norm(x,(x.shape[-1],),eps=torch.finfo(torch.float32).eps)

def attention_write64(block,x,first):
    import torch
    import torch.nn.functional as F
    att=block.attn;z=norm64(x);bs,t,d=z.shape;heads=att.n_head;hd=att.head_dim
    projections=[F.linear(z,getattr(att,name).weight.double()).reshape(bs,t,heads,hd) for name in ['c_q','c_k','c_q2','c_k2','c_v']]
    q,k,q2,k2,value=projections
    cos,sin=att.rotary(torch.zeros(bs,t,heads,hd,device=x.device,dtype=torch.float32));cos=cos.double();sin=sin.double()
    def rotate(y):
        y=norm64(y);y1,y2=y[...,:hd//2],y[...,hd//2:]
        return torch.cat([y1*cos+y2*sin,-y1*sin+y2*cos],dim=-1)
    q,k,q2,k2=[rotate(y) for y in [q,k,q2,k2]]
    scores=torch.einsum('bthd,bshd->bhts',q,k)/hd;scores2=torch.einsum('bthd,bshd->bhts',q2,k2)/hd
    mask=torch.ones(t,t,device=x.device,dtype=torch.bool).tril();pattern=(scores*scores2).masked_fill(~mask,0.)
    value=(1-att.lamb.double())*value+att.lamb.double()*first.double().reshape_as(value)
    at=torch.einsum('bhts,bshd->bthd',pattern,value).reshape(bs,t,d)
    return F.linear(at,att.c_proj.weight.double())

def mlp_write64(block,x):
    import torch.nn.functional as F
    z=norm64(x)
    return F.linear(F.linear(z,block.mlp.Left.weight.double())*F.linear(z,block.mlp.Right.weight.double()),block.mlp.Down.weight.double(),block.mlp.Down_bias.double())

def source_input(raw,directions,positions,amplitudes):
    import torch
    batch=torch.arange(len(raw),device=raw.device);edit=torch.zeros_like(raw,dtype=torch.float64)
    edit[batch,positions]=torch.einsum('bi,bid->bd',amplitudes.double(),directions.double())
    return raw.double()+edit

def first_source_mlp_input(model,raw,first,directions,positions,amplitudes):
    x=source_input(raw,directions,positions,amplitudes)
    return x+attention_write64(model.transformer.h[11],x,first)

def source_observables(model,raw,x0,first,directions,positions,read,pairs,amplitudes,capture=None):
    import torch
    x=source_input(raw,directions,positions,amplitudes);batch=torch.arange(len(raw),device=raw.device)
    if capture is not None:capture.update(mlp_inputs={},mlp_outputs={})
    for l in range(11,18):
        b=model.transformer.h[l]
        if l>11:x=b.lambdas[0].double()*x+b.lambdas[1].double()*x0.double()
        x=x+attention_write64(b,x,first)
        if capture is not None:capture['mlp_inputs'][l]=x
        x=x+mlp_write64(b,x)
        if capture is not None:capture['mlp_outputs'][l]=x
    logits=30*torch.tanh((norm64(x[batch,read])[:,None,None,:]*model.lm_head.weight[pairs].double()).sum(-1)/30)
    return logits[:,:,0]-logits[:,:,1]

def source_observables32(model,raw,x0,first,directions,positions,read,pairs,amplitudes,capture=None,start_layer=11):
    """Native float32 endpoint for the same nominated-source interface."""
    import torch
    import torch.nn.functional as F
    batch=torch.arange(len(raw),device=raw.device);x=raw.clone()
    x[batch,positions]=(raw[batch,positions].double()+torch.einsum('bi,bid->bd',amplitudes.double(),directions.double())).float()
    if not 11<=start_layer<=18:raise ValueError('Invalid suffix start')
    if capture is not None and capture.get('all_blocks'):
        capture['initial_state']=x.detach().clone();capture['post_blocks']={}
    for layer in range(start_layer,18):
        block=model.transformer.h[layer]
        if layer>11:x=block.lambdas[0]*x+block.lambdas[1]*x0
        attention,_=block.attn(F.rms_norm(x,(x.shape[-1],)),first);x=x+attention
        x=x+block.mlp(F.rms_norm(x,(x.shape[-1],)))
        if capture is not None and capture.get('all_blocks'):capture['post_blocks'][layer]=x.detach().clone()
    if capture is not None:capture['final_state']=x[batch,read].detach().clone()
    logits=30*torch.tanh(model.lm_head(F.rms_norm(x[batch,read],(x.shape[-1],)))/30)
    values=logits.gather(1,pairs.reshape(len(x),-1)).reshape(len(x),-1,2)
    return (values[:,:,0]-values[:,:,1]).double()
