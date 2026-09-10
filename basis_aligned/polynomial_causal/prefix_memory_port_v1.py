"""Factored prefix-memory interchange for native two-QK product attention.

The port replaces only earlier-token contributions to the last `suffix` queries.
Recipient within-suffix attention remains live. Queries are recomputed at donor
absolute phases so that their information-stage distance to donor history agrees.
No dense k1 tensor k2 tensor v accumulator is allocated.
"""
from contextlib import contextmanager
import torch
import torch.nn.functional as F
from jacclust.tt_model import apply_rotary_emb


def contract(q,k,v,q2,k2):
    d=q.shape[-1]
    p=(torch.einsum('bqhd,bshd->bhqs',q,k)/d)*(torch.einsum('bqhd,bshd->bhqs',q2,k2)/d)
    return torch.einsum('bhqs,bshd->bhqd',p,v)


@contextmanager
def capture(model,layers=range(9,18),suffix=3):
    records={};originals=[]
    try:
        for layer in layers:
            attn=model.transformer.h[layer].attn
            assert 'squared_attention' not in attn.__dict__
            original=attn.squared_attention;originals.append((attn,original))
            def wrapped(q,k,v,q2,k2,*,_a=attn,_f=original,_i=layer):
                assert q.shape[1]>suffix
                records[_i]={'k':k[:,:-suffix].detach().clone(),
                    'k2':k2[:,:-suffix].detach().clone(),'v':v[:,:-suffix].detach().clone(),
                    'cos':_a.rotary.cos_cached[-suffix:][None,:,None,:].detach().clone(),
                    'sin':_a.rotary.sin_cached[-suffix:][None,:,None,:].detach().clone(),
                    'length':q.shape[1],'suffix':suffix}
                return _f(q,k,v,q2,k2)
            attn.squared_attention=wrapped
        yield records
    finally:
        for attn,_ in originals:del attn.squared_attention


def slice_batch(records,start,end):
    return {i:{k:(v[start:end] if k in ('k','k2','v') else v) for k,v in r.items()}
            for i,r in records.items()}


def join_batches(chunks):
    return {i:{k:(torch.cat([r[i][k] for r in chunks]) if k in ('k','k2','v') else v)
               for k,v in chunks[0][i].items()} for i in chunks[0]}


@contextmanager
def replace(model,donor,recipient,audit):
    originals=[];handles=[]
    try:
        assert donor.keys()==recipient.keys()
        for layer in donor:
            attn=model.transformer.h[layer].attn;raw={}
            assert 'squared_attention' not in attn.__dict__
            original=attn.squared_attention;originals.append((attn,original))
            handles.append(attn.c_q.register_forward_hook(lambda _m,_a,o,_r=raw:_r.update(q=o.detach())))
            handles.append(attn.c_q2.register_forward_hook(lambda _m,_a,o,_r=raw:_r.update(q2=o.detach())))
            def wrapped(q,k,v,q2,k2,*,_f=original,_i=layer,_raw=raw):
                own=recipient[_i];other=donor[_i];suffix=own['suffix']
                assert suffix==other['suffix'] and q.shape[1]==own['length']
                assert all(torch.equal(now[:,:-suffix],own[name]) for now,name in [(k,'k'),(k2,'k2'),(v,'v')])
                def rematerialize(name,reference):
                    unrotated=_raw[name].view_as(reference)[:,-suffix:]
                    normalized=F.rms_norm(unrotated,(unrotated.shape[-1],))
                    return apply_rotary_emb(normalized,other['cos'],other['sin'])
                donor_q=rematerialize('q',q);donor_q2=rematerialize('q2',q2)
                base=_f(q,k,v,q2,k2)
                own_read=contract(q[:,-suffix:],k[:,:-suffix],v[:,:-suffix],q2[:,-suffix:],k2[:,:-suffix])
                donor_read=contract(donor_q,other['k'],other['v'],donor_q2,other['k2'])
                changed=base.clone()
                changed[:,:,-suffix:]=(base[:,:,-suffix:].double()-own_read.double()+donor_read.double()).to(base)
                audit.append({'layer':_i,'prefix_factors_bitwise':True,
                    'prefix_output_bitwise':torch.equal(changed[:,:,:-suffix],base[:,:,:-suffix]),
                    'finite':bool(changed.isfinite().all()),
                    'memory_read_change_norm':float((donor_read.double()-own_read.double()).norm()),
                    'query_phase_mode':'live raw Q RMS, donor native rounded phases'})
                return changed
            attn.squared_attention=wrapped
        yield
    finally:
        for h in handles:h.remove()
        for attn,_ in originals:del attn.squared_attention


def controls():
    from types import SimpleNamespace
    from jacclust.tt_model import CausalBilinearSelfAttention
    torch.manual_seed(9100918)
    class Fixture(torch.nn.Module):
        def __init__(self):
            super().__init__();self.transformer=torch.nn.Module()
            block=torch.nn.Module()
            block.attn=CausalBilinearSelfAttention(SimpleNamespace(n_head=2,n_embd=8,squared_attn=True,bilinear_attn=True))
            self.transformer.h=torch.nn.ModuleList([block])
            with torch.no_grad():block.attn.c_proj.weight.normal_(std=.2)
        def forward(self,x,v1=None):return self.transformer.h[0].attn(x,v1)
    model=Fixture();attn=model.transformer.h[0].attn
    x=torch.randn(2,6,8);y=torch.randn(2,7,8);edited=x.clone();edited[:,-3:]+=torch.randn(2,3,8)*.25
    with torch.inference_mode():
        with capture(model,(0,)) as a:base,first=model(x)
        with capture(model,(0,)) as b:model(y)
        with replace(model,a,a,[]) :identity,_=model(x)
        identity_error=float((identity-base).abs().max());assert identity_error<1e-6
        audit=[]
        with replace(model,b,a,audit):changed,changed_first=model(x)
        with capture(model,(0,)) as edited_memory:edited_base,_=model(edited)
        with replace(model,b,a,audit):joint,joint_first=model(edited)
        assert torch.equal(first,changed_first)
        assert torch.equal(joint_first[:,:-3],first[:,:-3])
        assert torch.equal(changed[:,:-3],base[:,:-3])
        assert all(torch.equal(edited_memory[0][k],a[0][k]) for k in ('k','k2','v'))
        difference=float(((joint-edited_base)-(changed-base)).norm())
        assert float((changed-base).norm())>.01 and difference>1e-4
        # Independent loop oracle for the actual different-length memory patch.
        pre={}
        h=attn.c_proj.register_forward_pre_hook(lambda _m,args:pre.update(head=args[0].clone()))
        try:model(x)
        finally:h.remove()
        raw_q=F.rms_norm(attn.c_q(x).view(2,6,2,4),(4,))[:,-3:]
        raw_q2=F.rms_norm(attn.c_q2(x).view(2,6,2,4),(4,))[:,-3:]
        def loop_read(memory):
            qa=apply_rotary_emb(raw_q,memory['cos'],memory['sin']).double()
            qb=apply_rotary_emb(raw_q2,memory['cos'],memory['sin']).double()
            output=torch.zeros(2,3,2,4,dtype=torch.float64)
            for batch in range(2):
                for query in range(3):
                    for head in range(2):
                        for source in range(memory['k'].shape[1]):
                            left=torch.dot(qa[batch,query,head],memory['k'][batch,source,head].double())/4
                            right=torch.dot(qb[batch,query,head],memory['k2'][batch,source,head].double())/4
                            output[batch,query,head]+=left*right*memory['v'][batch,source,head].double()
            return output.flatten(-2)
        oracle_head=pre['head'].double();oracle_head[:,-3:]+=loop_read(b[0])-loop_read(a[0])
        oracle=F.linear(oracle_head.float(),attn.c_proj.weight)
        oracle_error=float((oracle-changed).abs().max());assert oracle_error<1e-5
        # Make the shared-first-value mixture nontrivial as in later layers.
        external=torch.randn(2,6,2,4);donor_external=torch.randn(2,7,2,4)
        with capture(model,(0,)) as am:model(x,external)
        with capture(model,(0,)) as bm:model(y,donor_external)
        expected_mixed=((1-attn.lamb)*attn.c_v(x).view(2,6,2,4)+attn.lamb*external)[:,:-3]
        assert torch.equal(am[0]['v'],expected_mixed)
        with replace(model,bm,am,[]):_,returned_external=model(edited,external)
        assert torch.equal(returned_external,external)
        # Independent direct-vs-accumulator contraction on a small exact shape.
        q=torch.randn(2,3,2,4,dtype=torch.float64);q2=torch.randn_like(q)
        k=torch.randn(2,5,2,4,dtype=torch.float64);k2=torch.randn_like(k);v=torch.randn_like(k)
        memory=torch.einsum('bsha,bshc,bshv->bhacv',k,k2,v)
        recurrent=torch.einsum('bqha,bqhc,bhacv->bhqv',q,q2,memory)/16
        contraction_error=float((recurrent-contract(q,k,v,q2,k2)).abs().max());assert contraction_error<1e-12
        # A changed prefix is outside the declared source intervention domain.
        invalid=x.clone();invalid[:,0]+=1;rejected=False
        try:
            with replace(model,b,a,[]):model(invalid)
        except AssertionError:rejected=True
        assert rejected
    assert 'squared_attention' not in attn.__dict__
    assert not any(m._forward_hooks or m._forward_pre_hooks for m in model.modules())
    assert all(r['prefix_output_bitwise'] and r['finite'] for r in audit)
    return {'passed':True,'native_class_identity_max_abs':identity_error,
            'different_length_memory_effect_norm':float((changed-base).norm()),
            'live_source_changes_memory_effect_norm':difference,
            'direct_accumulator_max_abs':contraction_error,'changed_prefix_rejected':rejected,
            'independent_prefix_loop_max_abs':oracle_error,
            'external_first_value_mixing_and_preservation':True,
            'hooks_and_methods_restored':True,
            'scope':'CPU native attention-class controls, including real QK RMS, rounded RoPE, mixed values and output projection. No trained-model context-transfer claim.'}


if __name__=='__main__':
    import json,sys
    from pathlib import Path
    result=controls()
    with Path(sys.argv[1]).open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))
