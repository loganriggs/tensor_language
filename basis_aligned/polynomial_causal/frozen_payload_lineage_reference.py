"""Exact linear payload transport conditioned on native nonlinear routing gates."""
import torch


def prepare(model,tokens):
    e=model.embed(tokens);x=e;gates=[]
    for layer in model.layers[:3]:
        assert layer.residual=='lerp' and layer.scale==.5 and layer.v.bias is None and layer.o.bias is None
        eps=torch.finfo(x.dtype).eps if layer.norm.eps is None else layer.norm.eps
        gates.append({'pattern':layer.pattern(x),'gain':torch.rsqrt(x.square().mean(-1,keepdim=True)+eps)})
        x=layer(x)
    return {'embedding':e,'gates':gates,'native_prefix':x}


def propagate(model,z,cache,diagonal=False):
    for layer,gate in zip(model.layers[:3],cache['gates']):
        v=layer.v(z*gate['gain']).reshape(len(z),z.shape[1],layer.n_head,layer.d_head)
        if diagonal:
            p=gate['pattern'].diagonal(dim1=-2,dim2=-1).transpose(1,2)
            summed=p.unsqueeze(-1)*v
        else:summed=torch.einsum('bhts,bshd->bthd',gate['pattern'],v)
        z=.5*z+.5*layer.o(summed.flatten(-2))
    return z


def decompose(model,tokens,cache=None):
    if cache is None:cache=prepare(model,tokens)
    e=cache['embedding'];key=torch.zeros_like(e);value=torch.zeros_like(e);query=torch.zeros_like(e)
    key[:,:48:2]=e[:,:48:2];value[:,1:48:2]=e[:,1:48:2];query[:,48:]=e[:,48:]
    local=propagate(model,value,cache,True);all_value=propagate(model,value,cache)
    return {'key_roots':propagate(model,key,cache),'local_value_roots':local,
            'transferred_value_roots':all_value-local,'query_roots':propagate(model,query,cache)}


def controls():
    from deep_model import DeepModel
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(27908);model=DeepModel(8,16,4,['attn']*4,240,norm='rms').double().eval()
        tok=torch.randint(8,(2,9));delta=torch.randn(2,9,16,dtype=torch.float64)*.1
    checks={}
    with torch.inference_mode():
        cache=prepare(model,tok);e=cache['embedding'];dense=propagate(model,e,cache)
        checks['native_payload_closure']=bool(torch.allclose(dense,cache['native_prefix'],atol=1e-9,rtol=1e-10))
        pieces=[];local=propagate(model,e,cache,True);diagonal=True;causal=True
        for s in range(tok.shape[1]):
            root=torch.zeros_like(e);root[:,s]=e[:,s];output=propagate(model,root,cache);pieces.append(output)
            diagonal &= bool(torch.allclose(output[:,s],local[:,s],atol=1e-9,rtol=1e-10))
            causal &= not bool(output[:,:s].ne(0).any())
        checks['sum_all_roots']=bool(torch.allclose(sum(pieces),dense,atol=1e-9,rtol=1e-10))
        checks['causal_diagonal_recurrence']=diagonal;checks['future_to_past_zero']=causal
        changed=e+delta
        for layer in model.layers[:3]:changed=layer(changed)
        frozen=propagate(model,e+delta,cache)
        checks['frozen_payload_is_not_input_intervention']=float((changed-frozen).abs().max())>1e-12
    return {'passed':all(checks.values()),'checks':checks}
