"""Observe native shared/contextual reads without changing the returned forward."""
# BQGATE: LIBRARY
from contextlib import contextmanager


@contextmanager
def capture(model, selection):
    import torch
    handles=[];methods=[];raw={};observed={}
    def save(layer):
        def hook(_module,_args,out):raw[layer]=out
        return hook
    try:
        if 0 in selection:raise ValueError('shared producer must remain intact')
        handles.append(model.transformer.h[0].attn.c_v.register_forward_hook(save(0)))
        for layer,heads in selection.items():
            attn=model.transformer.h[layer].attn
            if not heads or len(set(heads))!=len(heads) or any(h<0 or h>=attn.n_head for h in heads):raise ValueError('invalid heads')
            original=attn.squared_attention;methods.append((attn,original,'squared_attention' in attn.__dict__))
            handles.append(attn.c_v.register_forward_hook(save(layer)))
            def observe(q,k,v,q2,k2,attn=attn,original=original,layer=layer,heads=heads):
                native=original(q,k,v,q2,k2)
                shared=attn.lamb*raw[0].view_as(v);context=(1-attn.lamb)*raw[layer].view_as(v)
                shared_read=original(q,k,shared,q2,k2);context_read=original(q,k,context,q2,k2)
                width=q.shape[-1];length=q.shape[1]
                pattern=(torch.einsum('bqhd,bkhd->bhqk',q,k)/width)*(torch.einsum('bqhd,bkhd->bhqk',q2,k2)/width)
                pattern.masked_fill_(~torch.ones(length,length,device=q.device,dtype=torch.bool).tril(),0.)
                if layer in observed:raise ValueError('capture is one forward only')
                observed[layer]={h:{'pattern':pattern[:,h], 'value':shared[:,:,h],
                    'weight':attn.c_proj.weight[:,h*width:(h+1)*width],
                    'shared_read':shared_read[:,h], 'native_read':native[:,h],
                    'context_read':context_read[:,h]} for h in heads}
                return native
            attn.squared_attention=observe
        yield observed
    finally:
        for handle in handles:handle.remove()
        for attn,original,had_instance in methods:
            if had_instance:attn.squared_attention=original
            else:del attn.squared_attention


def controls():
    import torch
    import torch.nn.functional as F
    from jacclust.tt_model import GPT,GPTConfig
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(60910)
        model=GPT(GPTConfig(vocab_size=8,n_layer=3,n_head=2,n_embd=16,bilinear=True,squared_attn=True,bilinear_attn=True)).double().eval()
        for block in model.transformer.h:block.attn.c_proj.weight.data.normal_(std=.1)
        model.lm_head.weight.data.normal_(std=.1)
        model.transformer.h[1].attn.lamb.data.fill_(-.7);model.transformer.h[2].attn.lamb.data.fill_(1.3)
        tokens=torch.randint(8,(4,7))
    def forward():
        x=F.rms_norm(model.transformer.wte(tokens),(16,));x0=x;first=None
        for block in model.transformer.h:x,first=block(x,first,x0)
        return model.lm_head(F.rms_norm(x,(16,)))
    with torch.inference_mode():
        native=forward()
        with capture(model,{1:(0,),2:(1,)}) as reads:actual=forward()
        rs=[r for heads in reads.values() for r in heads.values()]
        checks={'forward_unchanged':torch.equal(native,actual),'counts':len(rs)==2,
            'pattern_oracle':all(torch.allclose(r['pattern']@r['value'],r['shared_read'],atol=1e-9,rtol=1e-10) for r in rs),
            'value_partition':all(torch.allclose(r['shared_read']+r['context_read'],r['native_read'],atol=1e-9,rtol=1e-10) for r in rs),
            'shared_live':all(float(r['shared_read'].norm())>1e-4 for r in rs),
            'restored':torch.equal(native,forward()) and all('squared_attention' not in b.attn.__dict__ for b in model.transformer.h)}
    return {'passed':all(checks.values()),'checks':checks}
