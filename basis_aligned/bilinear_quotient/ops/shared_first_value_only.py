"""Remove chosen contextual value producers; retain shared first-value paths."""
# BQGATE: LIBRARY
from contextlib import contextmanager


@contextmanager
def remove_context_values(model, selection):
    handles=[];width=model.config.n_embd//model.config.n_head
    try:
        for layer,heads in selection.items():
            if layer==0 or not heads or len(set(heads))!=len(heads):raise ValueError('invalid contextual-value selection')
            if any(h<0 or h>=model.config.n_head for h in heads):raise ValueError('head out of range')
            def patch(_module,_args,out,heads=heads):
                changed=out.clone()
                for h in heads:changed[...,h*width:(h+1)*width]=0
                return changed
            handles.append(model.transformer.h[layer].attn.c_v.register_forward_hook(patch))
        yield
    finally:
        for handle in handles:handle.remove()


def controls():
    import torch
    import torch.nn.functional as F
    from jacclust.tt_model import GPT, GPTConfig
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(5909)
        model=GPT(GPTConfig(vocab_size=8,n_layer=3,n_head=2,n_embd=16,bilinear=True,squared_attn=True,bilinear_attn=True)).double().eval()
        state=torch.randn(2,7,16,dtype=torch.float64);first=torch.randn(2,7,2,8,dtype=torch.float64)
    checks={}
    with torch.inference_mode():
        for layer,lamb in ((1,-.7),(2,1.3)):
            attention=model.transformer.h[layer].attn;attention.lamb.fill_(lamb)
            observed={};original=attention.squared_attention
            def capture(q,k,v,q2,k2):
                observed['actual']=original(q,k,v,q2,k2)
                observed['shared']=original(q,k,attention.lamb*first,q2,k2)
                return observed['actual']
            attention.squared_attention=capture
            try:
                attention(F.rms_norm(state,(16,)),first);native=observed['actual'].clone()
                with remove_context_values(model,{1:(0,),2:(0,)}):attention(F.rms_norm(state,(16,)),first)
                actual=observed['actual'];shared=observed['shared']
                checks[str(layer)+'_selected_shared_read']=torch.allclose(actual[:,0],shared[:,0],atol=1e-9,rtol=1e-10)
                checks[str(layer)+'_unselected_unchanged']=torch.equal(actual[:,1],native[:,1])
                checks[str(layer)+'_live_removal']=float((actual[:,0]-native[:,0]).abs().max())>1e-5
                attention(F.rms_norm(state,(16,)),first)
                checks[str(layer)+'_hooks_restored']=torch.equal(observed['actual'],native)
            finally:attention.squared_attention=original
        try:
            with remove_context_values(model,{0:(0,)}):pass
            checks['first_producer_guard']=False
        except ValueError:checks['first_producer_guard']=True
    checks={k:bool(v) for k,v in checks.items()}
    return {'passed':all(checks.values()),'checks':checks}
