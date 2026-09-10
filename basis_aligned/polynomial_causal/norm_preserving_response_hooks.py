"""Use current native normalization while simplifying only product-factor responses."""
from contextlib import contextmanager
import torch
import polynomial_factor_response as F


@contextmanager
def capture(model,layers=range(5,9)):
    records={layer:{} for layer in layers};handles=[];methods=[]
    def save(layer,name):
        def hook(_m,_args,out):records[layer][name]=out.detach().clone()
        return hook
    try:
        for layer in layers:
            block=model.transformer.h[layer];attn=block.attn;original=attn.squared_attention
            methods.append((attn,original,'squared_attention' in attn.__dict__))
            def observe(q,k,v,q2,k2,layer=layer,original=original):
                out=original(q,k,v,q2,k2)
                records[layer]['factors']=tuple(x.detach().clone() for x in (q,k,v,q2,k2));records[layer]['read']=out.detach().clone()
                return out
            attn.squared_attention=observe
            handles.extend((block.mlp.Left.register_forward_hook(save(layer,'left')),block.mlp.Right.register_forward_hook(save(layer,'right')),block.mlp.register_forward_hook(save(layer,'mlp'))))
        yield records
    finally:
        for h in handles:h.remove()
        for attn,original,had in methods:
            if had:attn.squared_attention=original
            else:del attn.squared_attention


@contextmanager
def install(model,base,layers=range(5,9),audit=None):
    handles=[];methods=[];current={layer:{} for layer in layers}
    def save(layer,name):
        def hook(_m,_args,out):current[layer][name]=out
        return hook
    try:
        for layer in layers:
            block=model.transformer.h[layer];attn=block.attn;original=attn.squared_attention
            methods.append((attn,original,'squared_attention' in attn.__dict__))
            reference=tuple(x.double() for x in base[layer]['factors'])
            def replace_read(q,k,v,q2,k2,layer=layer,reference=reference):
                # These arguments already contain current residual/head RMS and RoPE.
                change=F.attention_response(reference,tuple(x.double() for x in (q,k,v,q2,k2)))
                if audit is not None:audit.append(('attention',layer))
                return (base[layer]['read'].double()+change).to(q)
            attn.squared_attention=replace_read
            handles.extend((block.mlp.Left.register_forward_hook(save(layer,'left')),block.mlp.Right.register_forward_hook(save(layer,'right'))))
            weight=block.mlp.Down.weight.detach().double()
            def replace_mlp(_module,_args,out,layer=layer,weight=weight):
                change=F.mlp_response(base[layer]['left'].double(),base[layer]['right'].double(),current[layer]['left'].double(),current[layer]['right'].double(),weight)
                if audit is not None:audit.append(('mlp',layer))
                return (base[layer]['mlp'].double()+change).to(out)
            handles.append(block.mlp.register_forward_hook(replace_mlp))
        yield
    finally:
        for h in handles:h.remove()
        for attn,original,had in methods:
            if had:attn.squared_attention=original
            else:del attn.squared_attention


def controls():
    import value_lineage_capture as C
    from jacclust.tt_model import GPT,GPTConfig
    import torch.nn.functional as NF
    torch.set_num_threads(2)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(60931)
        model=GPT(GPTConfig(vocab_size=16,n_layer=4,n_head=2,n_embd=16,bilinear=True,squared_attn=True,bilinear_attn=True)).double().eval()
        for b in model.transformer.h:
            b.attn.c_proj.weight.data.normal_(std=.1);b.mlp.Down.weight.data.normal_(std=.1);b.lambdas.data.copy_(torch.tensor([.7,.2]))
        tokens=torch.randint(16,(2,5));lengths=[5,5]
    def run():
        x=NF.rms_norm(model.transformer.wte(tokens),(16,));e=x;v=None
        for b in model.transformer.h:x,v=b(x,v,e)
        return x,v
    with torch.inference_mode():
        with capture(model,layers=(1,2)) as base:
            with C.capture(model,source=0,target=3) as source:native,payload=run()
        with install(model,base,layers=(1,2),audit=(coverage:=[])):identity,identity_payload=run()
        with C.replace_mlp(model.transformer.h[0].mlp,torch.zeros_like(source['mlp_output']),lengths):
            full,_=run()
            with install(model,base,layers=(1,2)):candidate,candidate_payload=run()
        error=max(float((F.attention(*(v.double() for v in r['factors']))-r['read']).abs().max()) for r in base.values())
        checks={'identity_exact':torch.equal(native,identity),'first_value_payload_preserved':torch.equal(payload,identity_payload) and torch.equal(payload,candidate_payload),'factor_read_oracle':error<1e-11,'all_current_factor_callbacks':sorted(coverage)==[('attention',1),('attention',2),('mlp',1),('mlp',2)],'candidate_live':float((candidate-native).norm())>.01,'omitted_terms_live':float((candidate-full).norm())>.01,'hooks_and_methods_restored':not any(m._forward_hooks or m._forward_pre_hooks for m in model.modules()) and all('squared_attention' not in b.attn.__dict__ for b in model.transformer.h)}
    return {'passed':all(checks.values()),'checks':checks,'factor_read_error':error,'tiny_model_forwards':4,'candidate_full_difference':float((candidate-full).norm())}
