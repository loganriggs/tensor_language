import torch
from sparse_response_pairs import pair_scores,prune_pairs
from shared_response_runtime import step


def test_native_atom_norm_and_diagonal_gauge():
    torch.manual_seed(668);d,r=7,3
    W=torch.randn(d,r,dtype=torch.float64);V=torch.randn(d,r,dtype=torch.float64)
    c=torch.randn(r,6,dtype=torch.float64);block=dict(inputs=r,coefficients=c)
    scores=pair_scores(block,W,V)
    i,j=torch.triu_indices(r,r)
    for n,(p,q) in enumerate(zip(i,j)):
        kernel=(torch.outer(W[:,p],W[:,q])+torch.outer(W[:,q],W[:,p]))/2
        tensor=torch.einsum('a,ij->aij',V@c[:,n],kernel)
        torch.testing.assert_close(scores[n],tensor.norm())
    s=torch.tensor([.1,-3.,4.],dtype=torch.float64)
    t=torch.tensor([2.,-.3,5.],dtype=torch.float64)
    changed=dict(block,coefficients=c*(s[i]*s[j])[None,:]/t[:,None])
    torch.testing.assert_close(pair_scores(changed,W/s,V*t),scores)


def test_sparse_product_execution_and_zero_quadratic_control():
    torch.manual_seed(669);r=3;d=7
    W=torch.randn(d,r,dtype=torch.float64);V=torch.randn(d,r,dtype=torch.float64)
    block=dict(inputs=r,coefficients=torch.zeros(r,6,dtype=torch.float64),
               carry=torch.eye(r,dtype=torch.float64),geometry=torch.eye(r,dtype=torch.float64),
               width=d,scale=torch.tensor(.8,dtype=torch.float64))
    block['coefficients'][:,2]=torch.tensor([1.,2.,3.]);block['coefficients'][:,4]=torch.tensor([2.,-1.,3.])
    runtime=dict(blocks=[block]);z=torch.randn(5,r,dtype=torch.float64)/10
    context=dict(linear=torch.randn(5,r,r,dtype=torch.float64),overlap=torch.zeros(5,r,dtype=torch.float64),
                 s0=torch.ones(5,1,dtype=torch.float64),baseline_write=torch.randn(5,r,dtype=torch.float64))
    sparse,_=prune_pairs(runtime,[W],[V],2)
    torch.testing.assert_close(step(sparse['blocks'][0],z,context),step(block,z,context))
    empty,_=prune_pairs(runtime,[W],[V],0)
    expected=step(dict(block,coefficients=torch.zeros_like(block['coefficients'])),z,context)
    torch.testing.assert_close(step(empty['blocks'][0],z,context),expected)
