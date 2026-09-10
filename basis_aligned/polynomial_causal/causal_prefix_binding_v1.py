"""Early-memory/late-query binding, with live conditional query features."""
from contextlib import contextmanager
import itertools
import numpy as np
import torch


def groups_and_signs(corners,early,late):
    corners=np.asarray(corners);groups={};lookup={tuple(row):i for i,row in enumerate(corners)};flip=[]
    for i,row in enumerate(corners):
        key=tuple(v for j,v in enumerate(row) if j not in (early,late))
        groups.setdefault(key,[]).append(i)
        opposite=row.copy();opposite[late]*=-1;flip.append(lookup[tuple(opposite)])
    groups=np.array(list(groups.values()));assert groups.shape[1]==4
    e=corners[groups,early];l=corners[groups,late]
    assert all(len(set(zip(a,b)))==4 for a,b in zip(e,l))
    return groups,e,l,np.array(flip)


def partition(q,k,v,q2,k2,corners,early,late):
    """Queries are suffix-only; K1/K2/V are prefix-only, after native norms/phase."""
    groups,e,l,flip=groups_and_signs(corners,early,late)
    ix=torch.tensor(groups,device=q.device);flip=torch.tensor(flip,device=q.device)
    assert all(torch.equal(x,x[flip]) for x in (k,k2,v)), 'Late factor entered prefix memory'
    e=torch.tensor(e,device=q.device,dtype=torch.float64);l=torch.tensor(l,device=q.device,dtype=torch.float64)
    qa,qb=q[ix].double(),q2[ix].double();ka,kb,payload=k[ix].double(),k2[ix].double(),v[ix].double()
    d=q.shape[-1]
    scores=(torch.einsum('giqhd,gjshd->gijhqs',qa,ka)/d)*(torch.einsum('giqhd,gjshd->gijhqs',qb,kb)/d)
    cross=torch.einsum('gijhqs,gjshv->gijhqv',scores,payload)
    new=torch.einsum('gi,gj,gijhqv->ghqv',l,e,cross)/16
    inherited=torch.einsum('gi,gijhqv->ghqv',e*l,cross)/16
    diagonal=cross[:,torch.arange(4,device=q.device),torch.arange(4,device=q.device)]
    total=torch.einsum('gi,gihqv->ghqv',e*l,diagonal)/4
    result={}
    for name,coefficient in [('new',new),('inherited',inherited),('full',total)]:
        component=(e*l)[:,:,None,None,None]*coefficient[:,None]
        output=torch.empty((len(q),)+coefficient.shape[1:],device=q.device,dtype=torch.float64)
        output[ix.flatten()]=component.flatten(0,1);result[name]=output
    return result


@contextmanager
def remove(model,native_memory,corners,early,late,mode,audit):
    assert mode in ('zero','new','inherited','full','joint');originals=[]
    try:
        for layer,memory in native_memory.items():
            attn=model.transformer.h[layer].attn;assert 'squared_attention' not in attn.__dict__
            original=attn.squared_attention;originals.append(attn)
            def wrapped(q,k,v,q2,k2,*,_f=original,_i=layer,_memory=memory):
                suffix=_memory['suffix']
                assert all(torch.equal(now[:,:-suffix],_memory[name]) for now,name in [(k,'k'),(k2,'k2'),(v,'v')])
                parts=partition(q[:,-suffix:],k[:,:-suffix],v[:,:-suffix],q2[:,-suffix:],k2[:,:-suffix],corners,early,late)
                difference=parts['new']+parts['inherited']-parts['full']
                closure=float(difference.norm()/parts['full'].norm().clamp_min(1e-30))
                assert closure<=1e-8 and float(difference.abs().max())<=1e-8
                delta=parts['new']+parts['inherited'] if mode=='joint' else (torch.zeros_like(parts['full']) if mode=='zero' else parts[mode])
                base=_f(q,k,v,q2,k2);changed=base.clone()
                changed[:,:,-suffix:]=(base[:,:,-suffix:].double()-delta).to(base)
                audit.append({'layer':_i,'closure_relative':closure,'prefix_factors_bitwise':True,
                    'late_factor_absent_bitwise':True,'prefix_output_bitwise':torch.equal(base[:,:,:-suffix],changed[:,:,:-suffix]),
                    'finite':bool(changed.isfinite().all()),'removed_norm':float(delta.norm())})
                return changed
            attn.squared_attention=wrapped
        yield
    finally:
        for attn in originals:del attn.squared_attention


def controls():
    corners=np.array(list(itertools.product((-1,1),repeat=2)));torch.manual_seed(9100940);worst=0.
    for early,late in [(0,1),(1,0)]:
        e=torch.tensor(corners[:,early],dtype=torch.float64)[:,None,None,None]
        l=torch.tensor(corners[:,late],dtype=torch.float64)[:,None,None,None]
        def rand(shape):return torch.randn(*shape,dtype=torch.float64)
        for _ in range(16):
            q=rand((4,2,2,3));q2=rand((4,2,2,3))
            k=rand((1,5,2,3))+e*rand((1,5,2,3));k2=rand((1,5,2,3))+e*rand((1,5,2,3));v=rand((1,5,2,3))+e*rand((1,5,2,3))
            parts=partition(q,k,v,q2,k2,corners,early,late)
            worst=max(worst,float((parts['new']+parts['inherited']-parts['full']).abs().max()))
        qa=rand((1,2,2,3));qb=rand((1,2,2,3)).expand(4,-1,-1,-1)
        ka=rand((1,5,2,3));kb=rand((1,5,2,3)).expand(4,-1,-1,-1);vv=rand((1,5,2,3)).expand(4,-1,-1,-1)
        new=partition(l*qa,e*ka,vv,qb,kb,corners,early,late)
        inherited=partition(e*l*qa,ka.expand(4,-1,-1,-1),vv,qb,kb,corners,early,late)
        assert new['new'].norm()>.01 and inherited['inherited'].norm()>.01
        assert new['inherited'].abs().max()<1e-12 and inherited['new'].abs().max()<1e-12
        bad=ka.expand(4,-1,-1,-1).clone();bad[0]+=1;rejected=False
        try:partition(l*qa,bad,vv,qb,kb,corners,early,late)
        except AssertionError:rejected=True
        assert rejected
    assert worst<1e-11
    return {'passed':True,'random_fixtures':32,'maximum_partition_error':worst,
            'both_causal_orders_checked':True,'new_only_and_inherited_only_identified':True,
            'late_memory_leak_rejected':True,
            'scope':'Exact conditional interaction algebra for query lift and late-invariant prefix memory. Counterfactual features retained; not independent semantic input generation.'}


def native_class_controls():
    from types import SimpleNamespace
    from jacclust.tt_model import CausalBilinearSelfAttention
    import prefix_memory_port_v1 as M
    class Fixture(torch.nn.Module):
        def __init__(self):
            super().__init__();self.transformer=torch.nn.Module();block=torch.nn.Module()
            block.attn=CausalBilinearSelfAttention(SimpleNamespace(n_head=2,n_embd=8,squared_attn=True,bilinear_attn=True))
            self.transformer.h=torch.nn.ModuleList([block])
            with torch.no_grad():block.attn.c_proj.weight.normal_(std=.2)
        def forward(self,x,first):return self.transformer.h[0].attn(x,first)
    torch.manual_seed(9100942);model=Fixture();corners=np.array(list(itertools.product((-1,1),repeat=2)))
    e=torch.tensor(corners[:,0],dtype=torch.float32)[:,None,None]
    x=torch.randn(4,6,8);x[:,:3]=torch.randn(1,3,8)+e*torch.randn(1,3,8)
    first=torch.randn(4,6,2,4);first[:,:3]=torch.randn(1,3,2,4)+e[...,None]*torch.randn(1,3,2,4)
    with torch.inference_mode():
        with M.capture(model,(0,)) as memory:base,_=model(x,first)
        outputs={};audits={}
        for mode in ('zero','full','new','inherited','joint'):
            audits[mode]=[]
            with remove(model,memory,corners,0,1,mode,audits[mode]):out,returned=model(x,first)
            assert torch.equal(returned,first) and torch.equal(out[:,:3],base[:,:3]);outputs[mode]=out
        assert torch.equal(outputs['zero'],base)
        joint_error=float((outputs['full']-outputs['joint']).abs().max());assert joint_error<1e-6
        edited=x.clone();edited[:,-3:]+=.2*torch.randn_like(edited[:,-3:]);changed_audit=[]
        with remove(model,memory,corners,0,1,'new',changed_audit):model(edited,first)
        changed=abs(changed_audit[0]['removed_norm']-audits['new'][0]['removed_norm']);assert changed>1e-6
    assert not any(m._forward_hooks or m._forward_pre_hooks for m in model.modules())
    assert 'squared_attention' not in model.transformer.h[0].attn.__dict__
    return {'passed':True,'zero_bitwise':True,'full_joint_max_abs':joint_error,
            'live_query_changes_component_norm':changed,'first_value_and_prefix_preserved':True,'cleanup':True}


if __name__=='__main__':
    import json,sys
    from pathlib import Path
    result={'algebra':controls(),'native_class':native_class_controls()}
    with Path(sys.argv[1]).open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))
