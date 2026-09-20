import torch
from finite_attention_readers import forward,pullback


def fixture():
    torch.manual_seed(675);batch,length,heads,hd=2,5,3,4;d=heads*hd
    weights={k:torch.randn(d,d,dtype=torch.float64)/d**.5 for k in ['q','k','q2','k2','v','o']}
    x=torch.randn(batch,length,d,dtype=torch.float64)
    y=x+torch.randn_like(x)*.7;first=torch.randn(batch,length,heads,hd,dtype=torch.float64)
    angles=torch.randn(length,hd//2,dtype=torch.float64)
    cos=angles.cos().bfloat16().double();sin=angles.sin().bfloat16().double()
    return weights,x,y,first,torch.tensor(.37,dtype=torch.float64),heads,cos,sin,.02,.03


def test_full_two_qk_finite_reader_closes_per_example():
    weights,x,y,*args=fixture();g=torch.randn_like(x)
    f0,_=forward(weights,x,*args);f1,_=forward(weights,y,*args)
    q=pullback(weights,x,y,*args,g)
    torch.testing.assert_close((q*(y-x)).sum((1,2)),(g*(f1-f0)).sum((1,2)),atol=1e-11,rtol=1e-11)
    # A reader at token zero cannot reach later inputs through causal attention.
    local=torch.zeros_like(g);local[:,0]=g[:,0]
    causal=pullback(weights,x,y,*args,local)
    torch.testing.assert_close(causal[:,1:],torch.zeros_like(causal[:,1:]),atol=0,rtol=0)
    multi=pullback(weights,x,y,*args,torch.stack([g,local]))
    torch.testing.assert_close(multi,torch.stack([q,causal]),atol=1e-11,rtol=1e-11)


def test_equal_endpoint_reader_matches_autograd():
    weights,x,_,*args=fixture();x.requires_grad_();g=torch.randn_like(x)
    value,_=forward(weights,x,*args)
    expected=torch.autograd.grad((value*g).sum(),x)[0]
    actual=pullback(weights,x.detach(),x.detach(),*args,g)
    torch.testing.assert_close(actual,expected,atol=1e-11,rtol=1e-11)
