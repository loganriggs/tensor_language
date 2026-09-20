"""Independent dense two-QK reference and derivative tests for source core."""
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from attention_source_core import compile_core,execute

def main():
    torch.manual_seed(20260920);torch.set_num_threads(2);b,t,d,p,o,heads=2,4,12,5,3,3;hd=d//heads;dtype=torch.float64;eps=torch.finfo(torch.float32).eps
    weights={k:torch.randn(d,d,dtype=dtype)/d**.5 for k in ['q','k','q2','k2','v','o']};h=torch.randn(b,t,d,dtype=dtype);K=torch.randn(b,t,d,p,dtype=dtype)*.2;readers=torch.randn(b,t,o,d,dtype=dtype);first=torch.randn(b,t,heads,hd,dtype=torch.float32);mixture=.37
    angle=torch.randn(t,hd//2);cos=angle.cos().bfloat16().double();sin=angle.sin().bfloat16().double()
    def dense(a):
        x=h+torch.einsum('btdp,bp->btd',K,a);x=F.rms_norm(x,(d,),eps=eps)
        def branch(name):
            z=(x@weights[name].T).reshape(b,t,heads,hd);z=F.rms_norm(z,(hd,),eps=eps);u,v=z.chunk(2,dim=-1);c=cos[None,:,None,:];s=sin[None,:,None,:];return torch.cat([u*c+v*s,-u*s+v*c],dim=-1)
        q,k,q2,k2=[branch(n) for n in ['q','k','q2','k2']];scores=torch.einsum('bthd,bshd->bhts',q,k)/hd;scores2=torch.einsum('bthd,bshd->bhts',q2,k2)/hd;pattern=(scores*scores2).masked_fill(~torch.ones(t,t,dtype=torch.bool).tril(),0.)
        v=(1-mixture)*(x@weights['v'].T).reshape(b,t,heads,hd)+mixture*first.double();y=torch.einsum('bhts,bshd->bthd',pattern,v).reshape(b,t,d)@weights['o'].T
        return torch.einsum('btod,btd->bo',readers,y)
    core=compile_core(weights,h,K,readers,first,mixture,heads,cos,sin,eps,eps);cases=[torch.zeros(b,p,dtype=dtype),torch.ones(b,p,dtype=dtype),-torch.ones(b,p,dtype=dtype),torch.randn(b,p,dtype=dtype)*2]
    replay=max(float((execute(core,a)-dense(a)).abs().max()) for a in cases)
    def derivatives(fn):
        a=torch.zeros(b,p,dtype=dtype,requires_grad=True);y=fn(a);gs=[];hs=[]
        for j in range(o):
            g=torch.autograd.grad(y[:,j].sum(),a,create_graph=True,retain_graph=True)[0];gs.append(g.detach());hs.append(torch.stack([torch.autograd.grad(g[:,i].sum(),a,retain_graph=True)[0] for i in range(p)],dim=1))
        return torch.stack(gs,dim=1),torch.stack(hs,dim=1)
    g,hess=derivatives(dense);gc,hc=derivatives(lambda a:execute(core,a));ge=float((g-gc).abs().max());he=float((hess-hc).abs().max())
    wrong=dict(core,cached_value=torch.zeros_like(core['cached_value']));tripwire=float((execute(wrong,cases[0])-dense(cases[0])).abs().max())
    def count(v):return v.numel() if isinstance(v,torch.Tensor) else sum(count(x) for x in v.values()) if isinstance(v,dict) else 0
    assert replay<1e-10 and ge<1e-10 and he<1e-10 and tripwire>1e-3
    result=dict(replay_max_abs=replay,gradient_max_abs=ge,hessian_max_abs=he,omitted_cached_value_error=tripwire,compiled_values=count(core),native_weight_values=count(weights),scope='CPU toy with position/context-dependent source directions and readers. Literal compiled cost can exceed weights; no native-context validation or speed claim yet.')
    Path(__file__).with_name('ATTENTION_SOURCE_CORE_V1_CPU_RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
if __name__=='__main__':main()
