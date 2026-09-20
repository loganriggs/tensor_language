import torch
from bilinear_chain_readers import backward_readers,secant_readers


def test_pullback_matches_independent_autograd_with_norm_bias_and_softcap():
    torch.manual_seed(651);d,k=9,15;eps=1e-7
    start=torch.randn(5,d,dtype=torch.float64,requires_grad=True)
    blocks=[];inputs=[];states=[start];x=start
    for _ in range(3):
        b=dict(left=torch.randn(k,d,dtype=torch.float64)/3,
               right=torch.randn(k,d,dtype=torch.float64)/3,
               down=torch.randn(d,k,dtype=torch.float64)/3,
               bias=torch.randn(d,dtype=torch.float64)/5,
               lambdas=torch.tensor([.8,.2],dtype=torch.float64))
        h=.8*x+torch.randn_like(x)/4;inputs.append(h);blocks.append(b)
        x=h+((h@b['left'].T)*(h@b['right'].T))@b['down'].T/(h.square().mean(-1,keepdim=True)+eps)+b['bias']
        states.append(x)
    U=torch.randn(2,d,dtype=torch.float64)*10
    logits=30*torch.tanh((x@U.T)/(x.square().mean(-1,keepdim=True)+eps).sqrt()/30)
    expected=torch.autograd.grad((logits[:,0]-logits[:,1]).sum(),states)
    actual=backward_readers(blocks,inputs,x,U,eps)
    for a,e in zip(actual,expected):torch.testing.assert_close(a,e,atol=1e-10,rtol=1e-10)


def test_finite_secant_closes_every_boundary_and_repeated_endpoint():
    torch.manual_seed(652);d,k=9,15;eps=1e-7
    blocks=[dict(left=torch.randn(k,d,dtype=torch.float64)/3,
                 right=torch.randn(k,d,dtype=torch.float64)/3,
                 down=torch.randn(d,k,dtype=torch.float64)/3,
                 bias=torch.randn(d,dtype=torch.float64)/5,
                 lambdas=torch.tensor([.8,.2],dtype=torch.float64)) for _ in range(3)]
    backgrounds=[torch.randn(5,d,dtype=torch.float64)/4 for _ in blocks]
    def forward(start):
        x=start;states=[x];inputs=[]
        for b,a in zip(blocks,backgrounds):
            h=.8*x+a;inputs.append(h)
            x=h+((h@b['left'].T)*(h@b['right'].T))@b['down'].T/(h.square().mean(-1,keepdim=True)+eps)+b['bias']
            states.append(x)
        return states,inputs
    x0=torch.randn(5,d,dtype=torch.float64);x1=x0+torch.randn_like(x0)
    s0,h0=forward(x0);s1,h1=forward(x1);U=torch.randn(2,d,dtype=torch.float64)*10
    def margin(x):
        y=30*torch.tanh((x@U.T)/(x.square().mean(-1,keepdim=True)+eps).sqrt()/30)
        return y[:,0]-y[:,1]
    target=margin(s1[-1])-margin(s0[-1])
    readers=secant_readers(blocks,h0,h1,s0[-1],s1[-1],U,eps)
    for q,b,e in zip(readers,s0,s1):
        torch.testing.assert_close((q*(e-b)).sum(-1),target,atol=1e-10,rtol=1e-10)
    repeated=secant_readers(blocks,h0,h0,s0[-1],s0[-1],U,eps)
    derivatives=backward_readers(blocks,h0,s0[-1],U,eps)
    for q,g in zip(repeated,derivatives):torch.testing.assert_close(q,g,atol=1e-10,rtol=1e-10)


def test_endpoint_derivatives_can_miss_finite_bilinear_composition():
    # h=x*(2-x), f=h*h: quartic from two bilinear operations plus a constant.
    x=torch.tensor([0.,1.],dtype=torch.float64,requires_grad=True)
    h=x*(2-x);y=h*h
    grad=torch.autograd.grad(y.sum(),x)[0]
    torch.testing.assert_close(grad,torch.zeros_like(grad))
    assert float((y[1]-y[0]).detach())==1.
    exact_secant=(2-x.sum())*h.sum()
    assert float((exact_secant*(x[1]-x[0])).detach())==1.
