"""Token-derived first-attention writes and cue messages; no native prefix calls.

Preserve deployed operation order, including first-value mixing and BF16 RoPE.
The retained native suffix/background and arbitrary weight contents stay charged.
"""
from contextlib import contextmanager
import torch
import torch.nn.functional as F
import token_local_attention_support as T


def factors(model,tokens):
    block=model.transformer.h[0];attn=block.attn;d=model.config.n_embd
    e=F.rms_norm(F.embedding(tokens,model.transformer.wte.weight),(d,))
    n=F.rms_norm(block.lambdas[0]*e+block.lambdas[1]*e,(d,))
    project=lambda m:F.linear(n,m.weight.to(n.dtype)).view(*n.shape[:2],attn.n_head,attn.head_dim)
    q,k,q2,k2,v=(project(m) for m in (attn.c_q,attn.c_k,attn.c_q2,attn.c_k2,attn.c_v))
    # Native Rotary keeps inv_freq as a plain CPU attribute, not a moved buffer.
    inv=attn.rotary.inv_freq
    time=torch.arange(tokens.shape[1],device=n.device).type_as(inv)
    freq=torch.outer(time,inv).to(n.device)
    cos=freq.cos().bfloat16()[None,:,None,:];sin=freq.sin().bfloat16()[None,:,None,:]
    def rotate(x):
        x=F.rms_norm(x,(x.shape[-1],));h=x.shape[-1]//2;a,b=x[...,:h],x[...,h:]
        return torch.cat((a*cos+b*sin,a*(-sin)+b*cos),-1).type_as(x)
    return rotate(q),rotate(k),(1-attn.lamb)*v+attn.lamb*v,rotate(q2),rotate(k2)


def read(factors,edges=None):
    q,k,v,q2,k2=factors;t=q.shape[1];d=q.shape[-1]
    dot=lambda a,b:torch.einsum('bthd,bshd->bhts',a,b)
    pattern=(dot(q,k)/d)*(dot(q2,k2)/d)
    mask=torch.ones(t,t,device=q.device,dtype=torch.bool).tril()
    if edges is not None:mask=mask & edges
    pattern=pattern.masked_fill(~mask.unsqueeze(-3),0.)
    return torch.einsum('bhts,bshd->bhtd',pattern,v)


def project(model,head_read):
    x=head_read.transpose(1,2).contiguous().flatten(2)
    return F.linear(x,model.transformer.h[0].attn.c_proj.weight.to(x.dtype))


def write(model,fs,edges=None):return project(model,read(fs,edges))


def cue_delta(model,base_factors,donor_factors,changed):
    edges=T.single_edit_edges(changed)
    return project(model,read(donor_factors,edges)-read(base_factors,edges))


@contextmanager
def capture(attn):
    record={}
    hook=attn.register_forward_hook(lambda _m,_a,y:record.update(write=y[0].detach().clone(),first_value=y[1].detach().clone()))
    try:yield record
    finally:hook.remove()


@contextmanager
def replace_write(attn,replacement,lengths,audit=None):
    def hook(_m,_a,y):
        changed=y[0].clone()
        for i,n in enumerate(lengths):changed[i,:n]=replacement[i,:n].to(changed)
        result=(changed,y[1])+tuple(y[2:])
        if audit is not None:audit.append(result[1] is y[1])
        return result
    handle=attn.register_forward_hook(hook)
    try:yield
    finally:handle.remove()


def controls():
    from jacclust.tt_model import GPT,GPTConfig
    from circuit_fast_screen_producer import Bilin18TorchBackend,ModelBatch
    import norm_preserving_response_hooks as H
    torch.set_num_threads(2)
    class TinyBackend(Bilin18TorchBackend):
        def __init__(self,model):self.model=model;self.torch=torch;self.F=F;self.device='cpu'
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(910413)
        model=GPT(GPTConfig(vocab_size=16,n_layer=2,n_head=2,n_embd=16,bilinear=True,squared_attn=True,bilinear_attn=True)).double().eval()
        for b in model.transformer.h:
            b.attn.c_proj.weight.data.normal_(std=.1);b.mlp.Down.weight.data.normal_(std=.05)
    base=ModelBatch(('a','b'),'base',((1,3,4,5),(7,2,8,9,10)),(1,1),(2,2),(3,4))
    donor=ModelBatch(('a','b'),'donor',((6,3,4,5),(7,11,8,9,10)),(1,1),(2,2),(3,4))
    backend=TinyBackend(model);records=[];fs=[];inputs=[];errors={};error=lambda a,b:float((a-b).abs().max())
    with torch.no_grad():
        for i,batch in enumerate((base,donor)):
            with H.capture(model,layers=(0,)) as native:
                with capture(model.transformer.h[0].attn) as record:backend.native(batch,capture=False)
            tokens,lengths=backend._tensor_batch(batch);f=factors(model,tokens)
            errors[f'factors{i}']=max(error(a,b) for a,b in zip(f,native[0]['factors']))
            errors[f'write{i}']=error(write(model,f),record['write'])
            inputs.append(tokens);fs.append(f);records.append(record)
        changed=inputs[0]!=inputs[1];delta=cue_delta(model,*fs,changed)
        errors['cue_delta']=error(delta,records[1]['write']-records[0]['write'])
        errors['identity_delta']=float(cue_delta(model,fs[0],fs[0],torch.zeros_like(changed)).abs().max())
        checks=[]
        with replace_write(model.transformer.h[0].attn,records[0]['write']+delta,[4,5],checks):
            with capture(model.transformer.h[0].attn) as patched:backend.native(base,capture=False)
        errors['installed_write']=max(error(patched['write'][i,:n],records[1]['write'][i,:n]) for i,n in enumerate((4,5)))
        first_value=torch.equal(patched['first_value'],records[0]['first_value']) and checks==[True]
    restored=not any(m._forward_hooks or m._forward_pre_hooks for m in model.modules()) and 'squared_attention' not in model.transformer.h[0].attn.__dict__
    assert first_value and restored and float(delta.norm())>1e-8 and all(v<1e-10 for v in errors.values())
    return {'passed':True,'errors':errors,'receiving_first_value_preserved':first_value,'hooks_restored':restored,
            'delta_norm':float(delta.norm()),'tiny_forwards':3,'tiny_sequence_instances':6,'tiny_token_factor_productions':2,
            'trained_model_loaded':False,'gpu_accessed':False,
            'scope':'Independent token-weight first-attention program and scoped write edit; no trained cue-carrier or smaller-model claim.'}


if __name__=='__main__':
    import json
    print(json.dumps(controls(),indent=2))
