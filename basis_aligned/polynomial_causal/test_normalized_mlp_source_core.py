"""Independent exact replay, derivatives, explicit bias and bilinear gauge controls."""
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from normalized_mlp_source_core import compile_core,evaluate,derivatives_at_zero

def main():
    torch.set_num_threads(2);torch.manual_seed(20260920);dtype=torch.float64
    b,d,m,p,o=2,11,17,5,4;eps=torch.finfo(torch.float32).eps
    left=torch.randn(m,d,dtype=dtype);right=torch.randn_like(left);down=torch.randn(d,m,dtype=dtype);bias=torch.randn(d,dtype=dtype)
    background=torch.randn(b,d,dtype=dtype);directions=torch.randn(b,d,p,dtype=dtype);readers=torch.randn(b,o,d,dtype=dtype)
    def native(a):
        x=background+torch.einsum('bdp,bp->bd',directions,a);x=F.rms_norm(x,(d,),eps=eps)
        y=((x@left.T)*(x@right.T))@down.T+bias
        return torch.einsum('bod,bd->bo',readers,y)
    core=compile_core(left,right,down,bias,background,directions,readers,eps)
    cases=[torch.zeros(b,p,dtype=dtype),torch.ones(b,p,dtype=dtype),-torch.ones(b,p,dtype=dtype),torch.randn(b,p,dtype=dtype)*2]
    replay=max(float((evaluate(core,a)-native(a)).abs().max()) for a in cases)
    a=torch.zeros(b,p,dtype=dtype,requires_grad=True);output=native(a);gs=[];hs=[]
    for j in range(o):
        g=torch.autograd.grad(output[:,j].sum(),a,create_graph=True,retain_graph=True)[0];gs.append(g)
        hs.append(torch.stack([torch.autograd.grad(g[:,k].sum(),a,retain_graph=True)[0] for k in range(p)],dim=1))
    g,h=derivatives_at_zero(core);ge=float((g-torch.stack(gs,dim=1)).abs().max());he=float((h-torch.stack(hs,dim=1)).abs().max())
    scale=torch.logspace(-2,2,m,dtype=dtype);gauged=compile_core(left*scale[:,None],right/scale[:,None],down,bias,background,directions,readers,eps)
    gauge=max(float((core[k]-gauged[k]).abs().max()) for k in core)
    wrong=dict(core,bias=torch.zeros_like(core['bias']));bias_tripwire=float((evaluate(wrong,cases[0])-native(cases[0])).abs().max())
    assert replay<1e-10 and ge<1e-10 and he<1e-9 and gauge<1e-10 and bias_tripwire>1e-3
    result=dict(exact_replay_max_abs=replay,gradient_max_abs=ge,hessian_max_abs=he,gauge_coefficient_max_abs=gauge,omitted_bias_error=bias_tripwire,runtime_values=sum(x.numel() for x in core.values()),shapes={k:list(v.shape) for k,v in core.items()},scope='CPU planted native algebra; fixed local readers. Not yet verified on captured native model contexts or claimed as a full-suffix replacement.')
    Path(__file__).with_name('NORMALIZED_MLP_SOURCE_CORE_V1_CPU_RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
