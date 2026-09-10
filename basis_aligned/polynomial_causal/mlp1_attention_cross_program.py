"""Exact shared-denominator bilinear components and read-only native streams."""
from contextlib import contextmanager
import torch
import torch.nn.functional as F


def components(pre,attention,left,right,reader_down,epsilon):
    squared_scale=(pre+attention).square().mean(-1,keepdim=True)+epsilon
    lu,ru=pre@left.T,pre@right.T
    la,ra=attention@left.T,attention@right.T
    return {'pre':(lu*ru)@reader_down.T/squared_scale,
            'cross':(lu*ra+la*ru)@reader_down.T/squared_scale,
            'attention':(la*ra)@reader_down.T/squared_scale}


@contextmanager
def capture_native_streams(model):
    frames=[];current={};handles=[]
    def embedding(_m,_args,out):
        current.clear();current['embedding']=F.rms_norm(out.detach(),(out.shape[-1],)).clone()
    def attention0(_m,_args,out):current['attention0']=out[0].detach().clone()
    def mlp0(_m,_args,out):current['mlp0']=out.detach().clone()
    def attention1(_m,_args,out):current['attention1']=out[0].detach().clone()
    def mlp1_input(_m,args):
        e=current['embedding'];l0=model.transformer.h[0].lambdas;l1=model.transformer.h[1].lambdas
        r0=(l0[0]*e+l0[1]*e)+current['attention0'];r0=r0+current['mlp0']
        pre=l1[0]*r0+l1[1]*e
        current['frame']={'pre':pre,'attention':current['attention1'],'normalized_input':args[0].detach().clone()}
    def mlp1_output(_m,_args,out):
        frame=current.pop('frame');frame['output']=out.detach().clone();frames.append(frame)
    for module,fn,pre in [(model.transformer.wte,embedding,False),(model.transformer.h[0].attn,attention0,False),
                           (model.transformer.h[0].mlp,mlp0,False),(model.transformer.h[1].attn,attention1,False),
                           (model.transformer.h[1].mlp,mlp1_input,True),(model.transformer.h[1].mlp,mlp1_output,False)]:
        handles.append(module.register_forward_pre_hook(fn) if pre else module.register_forward_hook(fn))
    try:yield frames
    finally:
        for handle in handles:handle.remove()


def controls():
    from types import SimpleNamespace
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(60917);dtype=torch.float64;d=16;h=24;k=8
        pre=torch.randn(3,5,d,dtype=dtype);attention=torch.randn_like(pre)
        left=torch.randn(h,d,dtype=dtype);right=torch.randn_like(left);down=torch.randn(k,h,dtype=dtype);bias=torch.randn(k,dtype=dtype)
    eps=torch.finfo(torch.float32).eps
    parts=components(pre,attention,left,right,down,eps)
    n=(pre+attention)/((pre+attention).square().mean(-1,keepdim=True)+eps).sqrt()
    direct=((n@left.T)*(n@right.T))@down.T+bias
    error=float((sum(parts.values())+bias-direct).abs().max())
    separately=components(pre,torch.zeros_like(pre),left,right,down,eps)['pre']+components(torch.zeros_like(pre),attention,left,right,down,eps)['attention']+parts['cross']
    # Native-style two-block execution checks stream capture including lambda reentry.
    class Attention(torch.nn.Module):
        def forward(self,x,v=None):return x*.2,(x if v is None else v)
    class MLP(torch.nn.Module):
        def forward(self,x):return x.square()*.1+.03
    blocks=[SimpleNamespace(attn=Attention(),mlp=MLP(),lambdas=torch.tensor([.8,.3],dtype=dtype)),
            SimpleNamespace(attn=Attention(),mlp=MLP(),lambdas=torch.tensor([.7,.2],dtype=dtype))]
    model=SimpleNamespace(transformer=SimpleNamespace(wte=torch.nn.Identity(),h=blocks))
    with capture_native_streams(model) as frames:
        e=F.rms_norm(model.transformer.wte(pre),(d,));x=e;v=None;expected_pre=None
        for layer,block in enumerate(blocks):
            live=block.lambdas[0]*x+block.lambdas[1]*e
            if layer==1:expected_pre=live.clone()
            a,v=block.attn(F.rms_norm(live,(d,)),v);x=live+a;x=x+block.mlp(F.rms_norm(x,(d,)))
    restored=all(not getattr(m,'_forward_hooks') and not getattr(m,'_forward_pre_hooks') for m in [model.transformer.wte]+[m for b in blocks for m in (b.attn,b.mlp)])
    frame=frames[0]
    checks={'component_sum':error<1e-9,'cross_is_live':float(parts['cross'].abs().max())>1,
            'independent_normalization_rejected':float((separately+bias-direct).abs().max())>1,
            'source_exchange_symmetry':torch.allclose(parts['cross'],components(attention,pre,left,right,down,eps)['cross'],atol=1e-12,rtol=1e-12),
            'native_stream_reconstruction':torch.equal(frame['pre'],expected_pre),
            'native_normalization_replay':torch.equal(F.rms_norm(frame['pre']+frame['attention'],(d,)),frame['normalized_input']),
            'capture_hooks_restored':restored}
    checks={key:bool(v) for key,v in checks.items()}
    return {'passed':all(checks.values()),'checks':checks,'max_component_output_error':error,
            'scope':'Shared full denominator; no independent-source knockout or normalization-path claim.'}
