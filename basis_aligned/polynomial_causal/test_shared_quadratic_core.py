import torch
from shared_quadratic_core import compile_core,execute


def test_shared_pairs_match_symmetric_core_with_signs_and_empty_support():
    torch.manual_seed(639)
    P=torch.linalg.qr(torch.randn(5,4,dtype=torch.float64)).Q
    W=torch.randn(7,3,dtype=torch.float64);z=torch.randn(2,6,5,dtype=torch.float64)
    core=torch.randn(3,4,4,dtype=torch.float64);core=(core+core.transpose(1,2))/2
    for g in [core,torch.zeros_like(core)]:
        program=compile_core(g);s=z@P
        expected=torch.einsum('...p,apq,...q->...a',s,g,s)@W.T
        torch.testing.assert_close(execute(z,W,P,program),expected)
    program=compile_core(core)
    assert len(program['p'])==10 and len(program['coefficients'])==30
