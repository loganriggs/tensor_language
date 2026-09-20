import torch
from projected_bilinear_response import (compile_response, evaluate, prepare_context,
    evaluate_prepared, prepare_readout, readout_prepared)


def test_joint_contraction_nonorthogonal_geometry_and_gauge():
    torch.manual_seed(647)
    d,k,r,o=11,17,3,4
    L=torch.randn(k,d,dtype=torch.float64)
    R=torch.randn_like(L);D=torch.randn(d,k,dtype=torch.float64)
    P=torch.randn(d,r,dtype=torch.float64);Q=torch.randn(d,o,dtype=torch.float64)
    h=torch.randn(9,d,dtype=torch.float64);z=torch.randn(9,r,dtype=torch.float64)
    eps=1e-7
    def mlp(x):return ((x@L.T)*(x@R.T))@D.T/(x.square().mean(-1,keepdim=True)+eps)
    baseline=mlp(h)@Q
    expected=(z@P.T+mlp(h+z@P.T)-mlp(h))@Q
    program=compile_response(L,R,D,P,Q)
    actual=evaluate(program,z,h,baseline,eps)
    torch.testing.assert_close(actual,expected,atol=1e-10,rtol=1e-10)
    context=prepare_context(program,h,baseline,eps)
    torch.testing.assert_close(evaluate_prepared(program,z,context),expected,atol=1e-10,rtol=1e-10)
    # Reuse the same scalar context under multiple independently chosen edits.
    for scale in [-.5,0.,.5,1.5]:
        zs=scale*z
        torch.testing.assert_close(evaluate_prepared(program,zs,context),evaluate(program,zs,h,baseline,eps),atol=1e-10,rtol=1e-10)
    U=torch.randn(2,d,dtype=torch.float64)
    fixed,readout_context=prepare_readout(h,P,U,eps)
    state=h+z@P.T
    reference=30*torch.tanh((state@U.T)/(state.square().mean(-1,keepdim=True)+eps).sqrt()/30)
    torch.testing.assert_close(readout_prepared(fixed,z,readout_context),reference,atol=1e-10,rtol=1e-10)
    scale=torch.linspace(.1,10,k,dtype=torch.float64)[:,None]
    gauged=compile_response(L*scale,R/scale,D,P,Q)
    for key in program:torch.testing.assert_close(gauged[key],program[key]) if isinstance(program[key],torch.Tensor) else None
    # A missing quadratic interaction is a scientifically distinct program.
    assert (evaluate(program,z,h,baseline,eps,False)-actual).norm()>1


def test_multiblock_planted_shared_response_space():
    """A known small circuit survives arbitrary baseline states and biases."""
    torch.manual_seed(648)
    d,r,k,batch=13,3,19,12
    P=torch.linalg.qr(torch.randn(d,r,dtype=torch.float64)).Q
    wrong=torch.linalg.qr(torch.randn(d,r,dtype=torch.float64)).Q
    base=torch.randn(batch,d,dtype=torch.float64)
    z=torch.randn(batch,r,dtype=torch.float64)/3
    edited=base+z@P.T;bad_z=(edited-base)@wrong
    x0=torch.randn_like(base);eps=1e-7
    for _ in range(4):
        L=torch.randn(k,d,dtype=torch.float64)/5
        R=torch.randn_like(L)/5
        D=P@torch.randn(r,k,dtype=torch.float64)/5
        bias=torch.randn(d,dtype=torch.float64)/9
        attention=torch.randn_like(base)/4
        h=.8*base+.2*x0+attention
        he=.8*edited+.2*x0+attention
        def write(x):return ((x@L.T)*(x@R.T))@D.T/(x.square().mean(-1,keepdim=True)+eps)
        baseline_write=write(h)
        program=compile_response(L,R,D,P,P)
        z=evaluate(program,.8*z,h,baseline_write@P,eps)
        bad_program=compile_response(L,R,D,wrong,wrong)
        bad_z=evaluate(bad_program,.8*bad_z,h,baseline_write@wrong,eps)
        base=h+baseline_write+bias
        edited=he+write(he)+bias
        torch.testing.assert_close(z@P.T,edited-base,atol=1e-12,rtol=1e-12)
    assert (bad_z@wrong.T-(edited-base)).norm()/(edited-base).norm()>.1
