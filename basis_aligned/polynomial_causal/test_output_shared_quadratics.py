import torch
from output_shared_quadratics import quadratic_blocks, execute_component, allocate_squares


def test_blocks_and_allocation_against_dense_tensor():
    torch.manual_seed(619)
    C,A,B = [torch.randn(*s,dtype=torch.float64) for s in [(4,7),(7,5),(7,5)]]
    W=torch.linalg.qr(torch.randn(4,4,dtype=C.dtype)).Q
    blocks=quadratic_blocks(C,A,B,W)
    vals,vecs=torch.linalg.eigh(blocks)
    raw=torch.einsum('vk,ki,kj->vij',C,A,B)
    target=(raw+raw.transpose(-1,-2))/2
    for budget in [0,1,7,20]:
        ids,active,energy=allocate_squares(vals,budget)
        approx=torch.zeros_like(target)
        for ix in ids:
            a=int(ix)//5;p=int(ix)%5;v=vecs[a,:,p]
            approx+=torch.einsum('v,i,j->vij',W[:,a]*vals[a,p],v,v)
        torch.testing.assert_close((target-approx).square().sum(),target.square().sum()-energy)


def test_extracted_component_preserves_signed_form_and_denominator():
    Q=torch.tensor([[2.,1.],[1.,-3.]],dtype=torch.float64)
    vals,vecs=torch.linalg.eigh(Q)
    x=torch.randn(13,2,dtype=Q.dtype)
    writer=torch.tensor([1.,-2.,3.],dtype=Q.dtype)
    expected=torch.einsum('bi,ij,bj->b',x,Q,x)/x.square().mean(-1)
    torch.testing.assert_close(execute_component(x,vecs,vals,writer,h=x),expected[:,None]*writer)
