"""Direct normalized-head reference and zero-axis tripwires for the mixed core."""
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from attention_mixed_edge_core import compile_core,execute,mixed

torch.set_num_threads(2);torch.manual_seed(914)
b,d,heads=3,8,2;hd=d//heads;eps=torch.finfo(torch.float32).eps
weights={k:torch.randn(d,d,dtype=torch.float64)/d**.5 for k in ['q','k','q2','k2','v','o']}
q0,qd,k0,kd=[torch.randn(b,d,dtype=torch.float64) for _ in range(4)]
readers=torch.randn(b,4,d,dtype=torch.float64);first=torch.randn(b,d,dtype=torch.float64);mixture=.4
angles=[torch.randn(b,hd//2,dtype=torch.float64) for _ in range(2)]
qcos,qsin,kcos,ksin=[f(x).bfloat16().double() for x in angles for f in [torch.cos,torch.sin]]
def compile(qdirection,kdirection):
    return compile_core(weights,q0,qdirection,k0,kdirection,readers,first,mixture,heads,qcos,qsin,kcos,ksin,eps,eps)
core=compile(qd,kd)
def direct(u,v):
    q=F.rms_norm(q0+u*qd,(d,),eps=eps);k=F.rms_norm(k0+v*kd,(d,),eps=eps)
    def branch(x,name,cos,sin):
        z=F.rms_norm(F.linear(x,weights[name]).reshape(b,heads,hd),(hd,),eps=eps)
        left,right=z.chunk(2,dim=-1)
        return torch.cat([left*cos[:,None,:]+right*sin[:,None,:],-left*sin[:,None,:]+right*cos[:,None,:]],dim=-1)
    pattern=((branch(q,'q',qcos,qsin)*branch(k,'k',kcos,ksin)).sum(-1)/hd)*((branch(q,'q2',qcos,qsin)*branch(k,'k2',kcos,ksin)).sum(-1)/hd)
    value=(1-mixture)*F.linear(k,weights['v']).reshape(b,heads,hd)+mixture*first.reshape(b,heads,hd)
    write=F.linear((pattern[...,None]*value).reshape(b,d),weights['o'])
    return torch.einsum('bod,bd->bo',readers,write)
errors=[];mixed_errors=[]
for u in [-1.,0.,.5,1.]:
    for v in [-1.,0.,.5,1.]:
        errors.append(float((execute(core,u,v)-direct(u,v)).abs().max()))
        expected=direct(u,v)-direct(u,0)-direct(0,v)+direct(0,0)
        mixed_errors.append(float((mixed(core,u,v)-expected).abs().max()))
dead=max(float(mixed(compile(torch.zeros_like(qd),kd),1.,1.).abs().max()),float(mixed(compile(qd,torch.zeros_like(kd)),1.,1.).abs().max()))
assert max(errors)<1e-11 and max(mixed_errors)<1e-11 and dead<1e-12
out=dict(max_edge_replay=max(errors),max_mixed_replay=max(mixed_errors),dead_axis_replay=dead,
         coefficients_per_context=sum(v.numel() for v in core.values())//b,
         scope='Planted normalized attention edge; native block-11 mixed localization still requires a native check.')
(Path(__file__).parent/'ATTENTION_MIXED_EDGE_CONTROL.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
