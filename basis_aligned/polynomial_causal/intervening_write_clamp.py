"""Capture/freeze actual intermediate module outputs during a source intervention."""
from contextlib import contextmanager
import torch


@contextmanager
def capture(model,layers=range(5,9)):
    handles=[];values={}
    def save(key,is_attention):
        def hook(_m,_args,out):values[key]=(out[0] if is_attention else out).detach().clone()
        return hook
    try:
        for layer in layers:
            b=model.transformer.h[layer]
            handles.append(b.attn.register_forward_hook(save(('attention',layer),True)))
            handles.append(b.mlp.register_forward_hook(save(('mlp',layer),False)))
        yield values
    finally:
        for h in handles:h.remove()


def controls():
    import torch.nn.functional as F
    import value_lineage_capture as C
    from jacclust.tt_model import GPT,GPTConfig
    torch.set_num_threads(2)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(60928)
        model=GPT(GPTConfig(vocab_size=16,n_layer=4,n_head=2,n_embd=16,bilinear=True,squared_attn=True,bilinear_attn=True)).double().eval()
        for b in model.transformer.h:
            b.attn.c_proj.weight.data.normal_(std=.05);b.mlp.Down.weight.data.normal_(std=.05);b.lambdas.data.copy_(torch.tensor([.7,.2]))
        tokens=torch.randint(16,(2,5));lengths=[3,5]
    def run():
        x=F.rms_norm(model.transformer.wte(tokens),(16,));e=x;v=None
        for b in model.transformer.h:x,v=b(x,v,e)
        return x,v
    with torch.inference_mode():
        with capture(model,layers=(2,)) as values:
            with C.capture(model,source=1,target=3) as original:native,payload=run()
        with clamp(model,values,lengths,('attention','mlp'),layers=(2,)):identity,identity_payload=run()
        with C.replace_mlp(model.transformer.h[1].mlp,torch.zeros_like(original['mlp_output']),lengths):
            with clamp(model,values,lengths,('attention','mlp'),layers=(2,)):
                with C.capture(model,source=1,target=3) as perturbed:changed,changed_payload=run()
        gamma=float(model.transformer.h[2].lambdas[0]*model.transformer.h[3].lambdas[0])
        errors=[float((perturbed['residual'][i,:n]-original['residual'][i,:n]+gamma*original['mlp_output'][i,:n]).abs().max()) for i,n in enumerate(lengths)]
        checks={'identity_clamps':torch.equal(native,identity),'first_value_payload_preserved':torch.equal(payload,identity_payload) and torch.equal(payload,changed_payload),'both_frozen_gives_direct_lineage':max(errors)<1e-11,'source_still_live':float((changed-native).norm())>.01,'hooks_restored':not any(m._forward_hooks or m._forward_pre_hooks for m in model.modules())}
    return {'passed':all(checks.values()),'checks':checks,'direct_lineage_max_error':max(errors),'tiny_model_forwards':3}


@contextmanager
def clamp(model,values,lengths,kinds,layers=range(5,9)):
    handles=[]
    def replace(key,is_attention):
        def hook(_m,_args,out):
            tensor=out[0] if is_attention else out;changed=tensor.clone()
            for i,n in enumerate(lengths):changed[i,:n]=values[key][i,:n].to(changed)
            # Shared first-value payload remains the actual receiving payload.
            return (changed,)+out[1:] if is_attention else changed
        return hook
    try:
        for layer in layers:
            b=model.transformer.h[layer]
            if 'attention' in kinds:handles.append(b.attn.register_forward_hook(replace(('attention',layer),True)))
            if 'mlp' in kinds:handles.append(b.mlp.register_forward_hook(replace(('mlp',layer),False)))
        yield
    finally:
        for h in handles:h.remove()
