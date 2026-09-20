import torch
from bilinear_chain_readers import backward_readers


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
