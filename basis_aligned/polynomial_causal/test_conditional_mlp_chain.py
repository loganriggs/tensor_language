import torch
from conditional_mlp_chain import execute, response


def test_independent_rational_quadratic_reference_and_bias():
    torch.manual_seed(645)
    d, hidden = 7, 13
    blocks = [dict(left=torch.randn(hidden,d,dtype=torch.float64)/5,
                   right=torch.randn(hidden,d,dtype=torch.float64)/5,
                   down=torch.randn(d,hidden,dtype=torch.float64)/5,
                   bias=torch.randn(d,dtype=torch.float64)/9,
                   lambdas=torch.tensor([.8,.2],dtype=torch.float64)) for _ in range(3)]
    start=torch.randn(4,d,dtype=torch.float64); initial=torch.randn_like(start)
    backgrounds=[torch.randn_like(start)/3 for _ in blocks]
    readout=torch.randn(2,d,dtype=torch.float64);eps=1e-7
    actual,state=execute(blocks,start,initial,backgrounds,readout,eps)
    x=start
    for b,a in zip(blocks,backgrounds):
        h=b['lambdas'][0]*x+b['lambdas'][1]*initial+a
        numerator=((h@b['left'].T)*(h@b['right'].T))@b['down'].T
        x=h+numerator/(h.square().mean(-1,keepdim=True)+eps)+b['bias']
    expected=30*torch.tanh((x/(x.square().mean(-1,keepdim=True)+eps).sqrt())@readout.T/30)
    torch.testing.assert_close(state,x,atol=1e-12,rtol=1e-12)
    torch.testing.assert_close(actual,expected,atol=1e-12,rtol=1e-12)
    no_bias=[dict(b,bias=torch.zeros_like(b['bias'])) for b in blocks]
    assert (execute(no_bias,start,initial,backgrounds,readout,eps)[0]-actual).abs().max()>.01
    edited = start + torch.randn_like(start)/2
    ref,ref_state = execute(blocks,edited,initial,backgrounds,readout,eps)
    base_logits,edited_logits,delta = response(blocks,start,edited,initial,backgrounds,readout,eps)
    torch.testing.assert_close(base_logits,actual,atol=1e-12,rtol=1e-12)
    torch.testing.assert_close(edited_logits,ref,atol=1e-12,rtol=1e-12)
    torch.testing.assert_close(delta,ref_state-state,atol=1e-12,rtol=1e-12)
    omitted = response(blocks,start,edited,initial,backgrounds,readout,eps,False)[1]
    assert (omitted-ref).abs().max()>.001
