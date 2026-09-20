import torch
from projected_bilinear_response import compile_response, evaluate


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
    scale=torch.linspace(.1,10,k,dtype=torch.float64)[:,None]
    gauged=compile_response(L*scale,R/scale,D,P,Q)
    for key in program:torch.testing.assert_close(gauged[key],program[key]) if isinstance(program[key],torch.Tensor) else None
    # A missing quadratic interaction is a scientifically distinct program.
    assert (evaluate(program,z,h,baseline,eps,False)-actual).norm()>1
