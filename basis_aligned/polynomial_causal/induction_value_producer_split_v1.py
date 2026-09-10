"""Live equality-message removals split by local versus shared-first value producer."""
from contextlib import contextmanager
import torch

SITES={5:(5,),7:(3,),8:(3,4)}


def component(q,k,q2,k2,value,positions,mask):
    batch=torch.arange(len(q),device=q.device)
    p=(torch.einsum('bhd,bthd->bht',q[batch,positions].double(),k.double())/q.shape[-1])
    p*=torch.einsum('bhd,bthd->bht',q2[batch,positions].double(),k2.double())/q.shape[-1]
    return torch.einsum('bht,bthd->bhd',p*mask[:,None,:],value.double())


@contextmanager
def remove(model,positions,equality_mask,neutral_mask,mode,audit,sites=SITES):
    if mode not in ('zero','shared','local','full','joint','neutral'):raise ValueError(mode)
    handles=[];originals=[];first={}
    try:
        handles.append(model.transformer.h[0].attn.register_forward_hook(
            lambda _m,_a,o:first.update(value=o[1].detach())))
        for layer,heads in sites.items():
            attn=model.transformer.h[layer].attn;raw={}
            handles.append(attn.c_v.register_forward_hook(lambda _m,_a,o,_r=raw:_r.update(value=o.detach())))
            assert 'squared_attention' not in attn.__dict__
            original=attn.squared_attention;originals.append(attn)
            def wrapped(q,k,v,q2,k2,*,_f=original,_a=attn,_r=raw,_heads=heads,_layer=layer):
                base=_f(q,k,v,q2,k2)
                pos=positions.to(q.device);mask=(neutral_mask if mode=='neutral' else equality_mask).to(q.device)
                local=_r['value'].view_as(v).double()*(1-_a.lamb.double())
                shared=first['value'].view_as(v).double()*_a.lamb.double()
                closure=float((local+shared-v.double()).norm()/v.double().norm().clamp_min(1e-30))
                if mode=='zero':delta=torch.zeros_like(base[:,:,0]).double()
                elif mode=='joint':delta=component(q,k,q2,k2,shared,pos,mask)+component(q,k,q2,k2,local,pos,mask)
                else:delta=component(q,k,q2,k2,{'shared':shared,'local':local,'full':v,'neutral':v}[mode],pos,mask)
                changed=base.clone();batch=torch.arange(len(q),device=q.device)
                for head in _heads:
                    changed[batch,head,pos]=(base[batch,head,pos].double()-delta[:,head]).to(base)
                keep=torch.ones(base.shape[:-1],dtype=torch.bool,device=q.device)
                for head in _heads:keep[batch,head,pos]=False
                audit.append(dict(layer=_layer,value_mix_relative=closure,
                    untouched_outputs_bitwise=torch.equal(changed[keep],base[keep]),
                    finite=bool(changed.isfinite().all()),delta_norm=float(delta[:,list(_heads)].norm())))
                return changed
            attn.squared_attention=wrapped
        yield
    finally:
        for h in handles:h.remove()
        for a in originals:del a.squared_attention


def controls():
    from types import SimpleNamespace
    from jacclust.tt_model import CausalBilinearSelfAttention
    torch.manual_seed(9101010)
    class Fixture(torch.nn.Module):
        def __init__(self):
            super().__init__()
            cfg=SimpleNamespace(n_embd=8,n_head=2,squared_attn=True,bilinear_attn=True)
            self.transformer=torch.nn.Module()
            self.transformer.h=torch.nn.ModuleList()
            for i in range(3):
                block=torch.nn.Module();block.attn=CausalBilinearSelfAttention(cfg).double()
                torch.nn.init.normal_(block.attn.c_proj.weight,std=.2)
                block.attn.lamb.data.fill_([.5,-.25,.75][i])
                self.transformer.h.append(block)
        def forward(self,x):
            first=None
            for b in self.transformer.h:
                out,first=b.attn(x,first);x=x+out
            return x,first
    model=Fixture();x=torch.randn(2,6,8,dtype=torch.float64)
    positions=torch.tensor([4,5]);mask=torch.zeros(2,6,dtype=torch.bool);mask[:,2]=True
    neutral=mask.roll(1,1);outputs={};audits=[]
    with torch.inference_mode():
        native,first=model(x)
        for mode in ('zero','shared','local','full','joint','neutral'):
            with remove(model,positions,mask,neutral,mode,audits,sites={1:(0,),2:(1,)}):
                outputs[mode],f=model(x)
                assert torch.equal(first,f)
    # Independent explicit summation of two score factors and each selected source value.
    q=torch.randn(2,6,2,4,dtype=torch.float64);k=torch.randn_like(q)
    q2=torch.randn_like(q);k2=torch.randn_like(q);v=torch.randn_like(q)
    actual=component(q,k,q2,k2,v,positions,mask);expected=torch.zeros_like(actual)
    for b in range(2):
        for h in range(2):
            for t in range(6):
                if mask[b,t]:expected[b,h]+=q[b,positions[b],h].dot(k[b,t,h])*q2[b,positions[b],h].dot(k2[b,t,h])*v[b,t,h]/16
    checks=dict(zero_bitwise=torch.equal(outputs['zero'],native),
        joint_full=float((outputs['joint']-outputs['full']).abs().max())<1e-10,
        independent_source_sum=float((actual-expected).abs().max())<1e-12,
        both_producers_live=all(float((outputs[m]-native).norm())>1e-6 for m in ('shared','local')),
        nonmatching_control_live=float((outputs['neutral']-native).norm())>1e-6,
        untouched=all(a['untouched_outputs_bitwise'] for a in audits),
        restored=not any(m._forward_hooks for m in model.modules()) and all('squared_attention' not in b.attn.__dict__ for b in model.transformer.h))
    assert all(checks.values()),checks
    return dict(passed=True,checks=checks,maximum_mix_relative=max(a['value_mix_relative'] for a in audits),
                maximum_joint_full=float((outputs['joint']-outputs['full']).abs().max()))
