"""Receiver-only value-response derivatives through the unchanged native backend."""
import torch
import value_lineage_capture as C


def field(backend,batch,source,source_layer=4,target_layer=9,heads=(1,4)):
    values=[];attn=backend.model.transformer.h[target_layer].attn
    # No detach: forward-mode derivatives must survive this observational hook.
    hook=attn.c_v.register_forward_hook(lambda _m,_args,out:values.append(out))
    try:
        with C.replace_mlp(backend.model.transformer.h[source_layer].mlp,source,[len(r) for r in batch.token_rows]):
            backend.native(batch,capture=False)
    finally:hook.remove()
    assert len(values)==1
    out=values[0]
    return out.view(*out.shape[:2],attn.n_head,attn.head_dim)[:,:,list(heads)]


def response(backend,batch,source,delta,source_layer=4,target_layer=9,heads=(1,4)):
    f=lambda x:field(backend,batch,x,source_layer,target_layer,heads)
    return torch.func.jvp(f,(source,),(delta,))


def controls():
    from jacclust.tt_model import GPT,GPTConfig
    from circuit_fast_screen_producer import Bilin18TorchBackend,ModelBatch
    import torch.nn.functional as F
    torch.set_num_threads(2)
    class TinyBackend(Bilin18TorchBackend):
        # Only dimension validation changes; native/_forward are inherited intact.
        def __init__(self,model):self.model=model;self.torch=torch;self.F=F;self.device='cpu'
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(60929)
        model=GPT(GPTConfig(vocab_size=16,n_layer=4,n_head=2,n_embd=16,bilinear=True,squared_attn=True,bilinear_attn=True)).double().eval()
        for b in model.transformer.h:
            b.attn.c_proj.weight.data.normal_(std=.05);b.mlp.Down.weight.data.normal_(std=.05);b.lambdas.data.copy_(torch.tensor([.7,.2]))
        batch=ModelBatch(('a','b'),'base',((1,2,3),(4,5,6,7,8)),(1,1),(2,2),(2,4))
        backend=TinyBackend(model)
        with C.capture(model,source=1,target=3) as r:backend.native(batch,capture=False)
        x=r['mlp_output'].clone();delta=torch.randn_like(x)*.1
    with torch.no_grad():
        primal,jvp=response(backend,batch,x,delta,1,3,(1,))
        h=1e-5;plus=field(backend,batch,x+h*delta,1,3,(1,));minus=field(backend,batch,x-h*delta,1,3,(1,));numerical=(plus-minus)/(2*h)
        _,zero=response(backend,batch,x,torch.zeros_like(delta),1,3,(1,))
    native=r['value'].view(2,5,2,8)[:,:,[1]]
    error=float((jvp-numerical).abs().max());rel=float((jvp-numerical).norm()/jvp.norm())
    checks={'primal_native_replay':torch.equal(primal,native),'central_difference':error<1e-8 and rel<1e-6,'nonzero_derivative':float(jvp.norm())>.01,'zero_direction_derivative':bool((zero==0).all()),'hooks_restored':not any(m._forward_hooks or m._forward_pre_hooks for m in model.modules())}
    return {'passed':all(checks.values()),'checks':checks,'central_difference_max_error':error,'central_difference_relative_error':rel,'tiny_model_forwards':5,'tiny_jvp_forwards':2}
