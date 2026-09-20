import torch
import torch.nn.functional as F
from projected_two_qk_attention import compile_attention,execute


def test_full_two_qk_value_fold_with_nested_rms_rope_and_causality():
    torch.manual_seed(656);batch,t,d,heads,r,a=2,5,12,3,3,2;hd=d//heads
    weights={k:torch.randn(d,d,dtype=torch.float64)/3 for k in ['q','k','q2','k2','v','o']}
    background=torch.randn(batch,t,d,dtype=torch.float64)
    P=torch.randn(d,r,dtype=torch.float64)/3;Q=torch.randn(d,a,dtype=torch.float64)/3
    first=torch.randn(batch,t,heads,hd,dtype=torch.float64);mixture=.37
    angle=torch.randn(t,hd//2);cos=angle.cos().bfloat16().double();sin=angle.sin().bfloat16().double()
    ex,eh=.03,.05  # Non-negligible eps catches incorrect cancellation of nested RMS.
    program=compile_attention(weights,background,P,Q,first,mixture,heads,cos,sin,ex,eh)
    z=torch.randn(batch,t,r,dtype=torch.float64)/3
    def native(z):
        x=F.rms_norm(background+z@P.T,(d,),eps=ex)
        def branch(name):
            y=F.rms_norm((x@weights[name].T).reshape(batch,t,heads,hd),(hd,),eps=eh)
            u,v=y.chunk(2,-1)
            return torch.cat([u*cos[None,:,None,:]+v*sin[None,:,None,:],
                              -u*sin[None,:,None,:]+v*cos[None,:,None,:]],-1)
        score1=torch.einsum('bthk,bshk->bhts',branch('q'),branch('k'))/hd
        score2=torch.einsum('bthk,bshk->bhts',branch('q2'),branch('k2'))/hd
        pattern=(score1*score2).masked_fill(~torch.ones(t,t,dtype=torch.bool).tril(),0)
        value=(1-mixture)*(x@weights['v'].T).reshape(batch,t,heads,hd)+mixture*first
        out=torch.einsum('bhts,bshk->bthk',pattern,value).reshape(batch,t,d)@weights['o'].T
        return out@Q
    for amplitude in [-1.,0.,.5,1.,1.5]:
        torch.testing.assert_close(execute(program,amplitude*z),native(amplitude*z),atol=1e-10,rtol=1e-10)
    changed=z.clone();changed[:,-1]+=2
    torch.testing.assert_close(execute(program,changed)[:,:-1],execute(program,z)[:,:-1])
