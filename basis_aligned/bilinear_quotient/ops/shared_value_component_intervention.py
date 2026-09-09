"""Consumer-local value component deletions, with a live shared producer."""
# BQGATE: LIBRARY
from contextlib import contextmanager


@contextmanager
def remove_components(model, selection, *, contextual=False, shared=False):
    import torch
    handles=[];methods=[];raw={};width=model.config.n_embd//model.config.n_head
    def capture(layer):
        def save(_module,_args,out):raw[layer]=out
        return save
    try:
        if 0 in selection:raise ValueError('shared producer must remain intact')
        handles.append(model.transformer.h[0].attn.c_v.register_forward_hook(capture(0)))
        for layer,heads in selection.items():
            if not heads or len(set(heads))!=len(heads) or any(h<0 or h>=model.config.n_head for h in heads):
                raise ValueError('invalid head selection')
            attn=model.transformer.h[layer].attn;original=attn.squared_attention
            had_instance_method='squared_attention' in attn.__dict__
            methods.append((attn,original,had_instance_method))
            handles.append(attn.c_v.register_forward_hook(capture(layer)))
            def changed(q,k,v,q2,k2,attn=attn,original=original,layer=layer,heads=heads):
                if not contextual and not shared:return original(q,k,v,q2,k2)
                value=v.clone()
                if contextual and shared:replacement=torch.zeros_like(v)
                elif contextual:replacement=attn.lamb*raw[0].view_as(v)
                else:replacement=(1-attn.lamb)*raw[layer].view_as(v)
                for h in heads:value[:,:,h]=replacement[:,:,h]
                return original(q,k,value,q2,k2)
            attn.squared_attention=changed
        yield
    finally:
        for handle in handles:handle.remove()
        for attn,original,had_instance_method in methods:
            if had_instance_method:attn.squared_attention=original
            else:del attn.squared_attention


def controls():
    import torch
    import torch.nn.functional as F
    from jacclust.tt_model import GPT,GPTConfig
    from shared_first_value_only import remove_context_values
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(60909)
        model=GPT(GPTConfig(vocab_size=8,n_layer=3,n_head=2,n_embd=16,bilinear=True,squared_attn=True,bilinear_attn=True)).double().eval()
        for block in model.transformer.h:block.attn.c_proj.weight.data.normal_(std=.1)
        model.lm_head.weight.data.normal_(std=.1)
        model.transformer.h[1].attn.lamb.data.fill_(-.7);model.transformer.h[2].attn.lamb.data.fill_(1.3)
        tokens=torch.randint(8,(2,7))
    selection={1:(0,),2:(1,)}
    def forward():
        x=F.rms_norm(model.transformer.wte(tokens),(16,));x0=x;first=None
        for block in model.transformer.h:x,first=block(x,first,x0)
        return model.lm_head(F.rms_norm(x,(16,)))
    with torch.inference_mode():
        native=forward();arms={}
        for contextual,shared in ((False,False),(True,False),(False,True),(True,True)):
            with remove_components(model,selection,contextual=contextual,shared=shared):arms[contextual,shared]=forward()
        with remove_context_values(model,selection):old=forward()
        handles=[]
        for layer,heads in selection.items():
            def zero(_module,args,heads=heads):
                x=args[0].clone()
                for h in heads:x[...,h*8:(h+1)*8]=0
                return (x,)+args[1:]
            handles.append(model.transformer.h[layer].attn.c_proj.register_forward_pre_hook(zero))
        try:zero_heads=forward()
        finally:
            for handle in handles:handle.remove()
        checks={'no_cut_identity':torch.equal(native,arms[False,False]),
                'parent_context_cut':torch.allclose(old,arms[True,False],atol=1e-9,rtol=1e-10),
                'direct_head_zero':torch.allclose(zero_heads,arms[True,True],atol=1e-9,rtol=1e-10),
                'all_deletions_live':all(float((a-native).abs().max())>1e-5 for k,a in arms.items() if k!=(False,False)),
                'restored_forward':torch.equal(native,forward()),
                'restored_method_descriptors':all('squared_attention' not in block.attn.__dict__ for block in model.transformer.h)}
        try:
            with remove_components(model,selection,shared=True):raise RuntimeError('planted exception')
        except RuntimeError:pass
        checks['exception_restoration']=torch.equal(native,forward()) and all('squared_attention' not in b.attn.__dict__ for b in model.transformer.h)
    checks={k:bool(v) for k,v in checks.items()}
    return {'passed':all(checks.values()),'checks':checks}
