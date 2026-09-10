"""Read-only residual recurrence capture and scoped whole-source/value edge swaps."""
from contextlib import contextmanager
import torch
import torch.nn.functional as F


@contextmanager
def capture(model,source=4,target=9):
    handles=[];state={};record={};d=model.config.n_embd
    def embedding(_m,_args,out):state['embedding']=F.rms_norm(out,(d,)).detach();state['residual']=state['embedding']
    def attention(block):
        def hook(_m,_args,out):state['residual']=block.lambdas[0]*state['residual']+block.lambdas[1]*state['embedding']+out[0].detach()
        return hook
    def mlp(layer):
        def hook(_m,args,out):
            state['residual']=state['residual']+out.detach()
            if layer==source:record.update(mlp_input=args[0].detach().clone(),mlp_output=out.detach().clone())
        return hook
    def target_input(_m,args):
        block=model.transformer.h[target]
        record['residual']=block.lambdas[0]*state['residual']+block.lambdas[1]*state['embedding']
        record['normalized_input']=args[0].detach().clone()
        record['recurrence_bitwise']=torch.equal(F.rms_norm(record['residual'],(d,)),args[0])
    try:
        handles.append(model.transformer.wte.register_forward_hook(embedding))
        for layer,block in enumerate(model.transformer.h[:target]):
            handles.append(block.attn.register_forward_hook(attention(block)))
            handles.append(block.mlp.register_forward_hook(mlp(layer)))
        handles.append(model.transformer.h[target].attn.register_forward_pre_hook(target_input))
        handles.append(model.transformer.h[target].attn.c_v.register_forward_hook(lambda _m,_a,out:record.update(value=out.detach().clone())))
        yield record
    finally:
        for h in handles:h.remove()


@contextmanager
def replace_mlp(module,replacement,lengths):
    def hook(_m,_args,out):
        changed=out.clone()
        for i,n in enumerate(lengths):changed[i,:n]=replacement[i,:n].to(changed)
        return changed
    h=module.register_forward_hook(hook)
    try:yield
    finally:h.remove()


@contextmanager
def replace_value(attn,replacement,lengths,heads=(1,4),audit=None):
    def hook(_m,_args,out):
        changed=out.clone();view=changed.view(*out.shape[:2],attn.n_head,attn.head_dim)
        mask=torch.zeros_like(view,dtype=torch.bool)
        for i,n in enumerate(lengths):
            view[i,:n,list(heads)]=replacement[i,:n].to(view);mask[i,:n,list(heads)]=True
        if audit is not None:audit.append(torch.equal(changed[~mask.reshape_as(out)],out[~mask.reshape_as(out)]))
        return changed
    h=attn.c_v.register_forward_hook(hook)
    try:yield
    finally:h.remove()


def controls():
    from jacclust.tt_model import GPT,GPTConfig
    torch.set_num_threads(2)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(60926)
        model=GPT(GPTConfig(vocab_size=16,n_layer=4,n_head=2,n_embd=16,bilinear=True,squared_attn=True,bilinear_attn=True)).double().eval()
        for b in model.transformer.h:
            b.attn.c_proj.weight.data.normal_(std=.05);b.mlp.Down.weight.data.normal_(std=.05);b.lambdas.data.copy_(torch.tensor([.7,.2]))
        tokens=torch.randint(16,(2,5));lengths=[3,5]
    def run():
        x=F.rms_norm(model.transformer.wte(tokens),(16,));e=x;v=None
        for b in model.transformer.h:x,v=b(x,v,e)
        return x
    with torch.inference_mode():
        native=run()
        with capture(model,source=1,target=3) as r:observed=run()
        value=r['value'].view(2,5,2,8)[:,:,[1]]
        with replace_value(model.transformer.h[3].attn,value,lengths,heads=(1,),audit=(mask:=[])):identity=run()
        with replace_mlp(model.transformer.h[1].mlp,torch.zeros_like(r['mlp_output']),lengths):
            with capture(model,source=1,target=3) as edited:changed=run()
        checks={'capture_unchanged':torch.equal(native,observed),'recurrence_native_and_edited':r['recurrence_bitwise'] and edited['recurrence_bitwise'],'identity_value':torch.equal(native,identity),'unselected_value_unchanged':all(mask),'source_edit_live':float((changed-native).norm())>.01,'hooks_restored':not any(m._forward_hooks or m._forward_pre_hooks for m in model.modules())}
    return {'passed':all(checks.values()),'checks':checks,'tiny_model_forwards':4}
