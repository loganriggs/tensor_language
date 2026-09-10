"""Read alternate final-query phases without changing any native forward output."""
from contextlib import contextmanager
from types import MethodType
import torch
from rotary_phase_transport import transport


def final_read(q,k,v,q2,k2,positions):
    idx=torch.arange(q.shape[0],device=q.device)
    pos=torch.as_tensor(positions,device=q.device)
    qf,qf2=q[idx,pos],q2[idx,pos]
    return read_queries(qf,k,v,qf2,k2,pos)


def read_queries(q,k,v,q2,k2,positions):
    d=q.shape[-1]
    score=torch.einsum('bhd,bshd->bhs',q,k)/d
    score2=torch.einsum('bhd,bshd->bhs',q2,k2)/d
    pattern=score*score2
    causal=torch.arange(k.shape[1],device=k.device)[None,:]<=positions[:,None]
    pattern=pattern.masked_fill(~causal[:,None,:],0.)
    return torch.einsum('bhs,bshd->bhd',pattern,v)


@contextmanager
def capture(model,batch,virtual_positions,layers=None):
    """Outputs are native/virtual dictionaries keyed like the producer head cache."""
    native,virtual={},{}
    saved=[]
    for layer,block in enumerate(model.transformer.h):
        if layers is not None and layer not in layers:continue
        attn=block.attn;original=attn.squared_attention
        existed='squared_attention' in attn.__dict__
        prior=attn.__dict__.get('squared_attention')
        def wrapped(self,q,k,v,q2,k2,_original=original,_layer=layer):
            # Return the exact original tensor; no edited states feed later layers.
            result=_original(q,k,v,q2,k2)
            idx=torch.arange(q.shape[0],device=q.device)
            pos=torch.as_tensor(batch.semantic_positions,device=q.device)
            new=torch.as_tensor(virtual_positions,device=q.device)
            if bool((new<0).any()) or bool((new>=q.shape[1]).any()):
                raise ValueError('Virtual phase position outside cached table')
            c,s=self.rotary.cos_cached,self.rotary.sin_cached
            oldc,olds=c[pos,None,:],s[pos,None,:]
            newc,news=c[new,None,:],s[new,None,:]
            qf,qf2=q[idx,pos],q2[idx,pos]
            qn=transport(qf,oldc,olds,newc,news)
            qn2=transport(qf2,oldc,olds,newc,news)
            # Keep the registered identity exactly identical; changed rows use inversion.
            same=(new==pos)[:,None,None]
            qn=torch.where(same,qf,qn);qn2=torch.where(same,qf2,qn2)
            alt=read_queries(qn,k,v,qn2,k2,pos)
            alt=torch.where(same,result[idx,:,pos,:],alt)
            for i,rid in enumerate(batch.row_ids):
                for h in range(q.shape[2]):
                    key=(rid,f'attn:{_layer:02d}:head:{h:02d}')
                    native[key]=result[i,h,pos[i]].detach().clone()
                    virtual[key]=alt[i,h].detach().clone()
            return result
        attn.squared_attention=MethodType(wrapped,attn)
        saved.append((attn,existed,prior))
    try:yield native,virtual
    finally:
        for attn,existed,prior in reversed(saved):
            if existed:attn.squared_attention=prior
            else:del attn.squared_attention


def controls():
    from jacclust.tt_model import GPT,GPTConfig,apply_rotary_emb
    from circuit_fast_screen_producer import Bilin18TorchBackend,ModelBatch
    import torch.nn.functional as F
    class Tiny(Bilin18TorchBackend):
        def __init__(self,m):self.model=m;self.torch=torch;self.F=F;self.device='cpu'
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(910440)
        model=GPT(GPTConfig(vocab_size=16,n_layer=2,n_head=2,n_embd=16,bilinear=True,bilinear_attn=True,squared_attn=True)).double().eval()
        for block in model.transformer.h:block.attn.c_proj.weight.data.normal_(std=.1)
    batch=ModelBatch(('a','b'),'base',((1,2,3,4,5),(1,2,3,4)),(1,1),(2,2),(4,3))
    backend=Tiny(model)
    with torch.no_grad():
        ordinary=backend.native(batch,capture=True)
        with capture(model,batch,[1,2]) as (native,virtual):observed=backend.native(batch,capture=True)
        with capture(model,batch,list(batch.semantic_positions)) as (identity,identity_alt):backend.native(batch,capture=True)
    error=lambda a,b:float((a-b).abs().max())
    cache_error=max(error(native[k],ordinary.captured[k]) for k in native)
    id_error=max(error(identity[k],identity_alt[k]) for k in identity)
    assert ordinary.answer_foil==observed.answer_foil
    assert cache_error<1e-10 and id_error<1e-10
    live=max(error(native[k],virtual[k]) for k in native)
    assert live>1e-8
    assert all('squared_attention' not in b.attn.__dict__ for b in model.transformer.h)
    # Direct rerotation oracle from pre-RoPE normalized projections, separate from inverse math.
    attn=model.transformer.h[0].attn
    with torch.no_grad():
        x=torch.randn(2,5,16,dtype=torch.float64)
        proj=lambda m:F.rms_norm(m(x).view(2,5,2,8),(8,))
        q,k,q2,k2=(proj(m) for m in (attn.c_q,attn.c_k,attn.c_q2,attn.c_k2))
        v=attn.c_v(x).view(2,5,2,8);c,s=attn.rotary(q)
        ks,k2s=apply_rotary_emb(k,c,s),apply_rotary_emb(k2,c,s)
        newc=c.expand(2,-1,-1,-1).clone();news=s.expand(2,-1,-1,-1).clone()
        for i,(old,new) in enumerate(zip((4,3),(1,2))):newc[i,old]=c[0,new];news[i,old]=s[0,new]
        expected=attn.squared_attention(apply_rotary_emb(q,newc,news),ks,v,apply_rotary_emb(q2,newc,news),k2s)
        with capture(model,batch,[1,2],layers=(0,)) as (_,alternate):
            attn.squared_attention(apply_rotary_emb(q,c,s),ks,v,apply_rotary_emb(q2,c,s),k2s)
        oracle=max(error(alternate[(rid,f'attn:00:head:{h:02d}')],expected[i,h,pos])
                   for i,(rid,pos) in enumerate(zip(batch.row_ids,batch.semantic_positions)) for h in range(2))
    assert oracle<1e-10
    return dict(passed=True,native_cache_error=cache_error,identity_read_error=id_error,
                rerotation_oracle_error=oracle,live_phase_read_max_change=live,
                native_outputs_unchanged=True,methods_restored=True,tiny_forwards=3,
                trained_forwards=0,gpu_accessed=False)


if __name__=='__main__':
    import json
    print(json.dumps(controls(),indent=2))
