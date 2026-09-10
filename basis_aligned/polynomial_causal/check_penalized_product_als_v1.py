"""The explicit energy penalty adds exactly lambda times the diagonal blocks."""
import json
from pathlib import Path
import torch
from symmetric_product_als_v1 import normal_operator,native_rhs,block_precondition,pcg
P=Path(__file__).resolve().parent
def main():
    torch.set_num_threads(2);torch.manual_seed(91162156)
    q,d,o,k=4,6,5,9;penalty=.01
    l,r=torch.randn(k,d,dtype=torch.float64),torch.randn(k,d,dtype=torch.float64)
    down=torch.randn(o,k,dtype=torch.float64);w=torch.randn(o,q,dtype=torch.float64);b=torch.randn(q,d,dtype=torch.float64)
    root=torch.randn(o,o,dtype=torch.float64);m=root@root.T+torch.eye(o,dtype=torch.float64);og=w.T@m@w;regularized=og+penalty*torch.diag(og.diag())
    def tensor(a,b,w):
        t=torch.einsum('ok,ki,kj->oij',w,a,b);return .5*(t+t.transpose(-1,-2))
    target=tensor(l,r,down)
    def loss(a):
        error=tensor(a,b,w)-target
        individual=.5*(a.square().sum(1)*b.square().sum(1)+(a*b).sum(1).square())*og.diag()
        return torch.einsum('oij,op,pij->',error,m,error)+penalty*individual.sum()
    a=torch.randn(q,d,dtype=torch.float64,requires_grad=True);old=loss(a);old.backward();rhs=native_rhs(l,r,down,b,w,m)
    expected=2*(normal_operator(a.detach(),b,regularized)-rhs);error=float((a.grad-expected).abs().max());assert error<=1e-10
    solution,diag=pcg(lambda z:normal_operator(z,b,regularized),rhs,lambda z:block_precondition(z,b,regularized),initial=a.detach(),tolerance=1e-10,max_iterations=200)
    assert diag['converged'] and float(loss(solution))<=float(old.detach())
    result=dict(schema='penalized.product.als.control.v1',passed=True,penalty=penalty,gradient_max_absolute_error=error,pcg=diag,dense_penalized_before=float(old.detach()),dense_penalized_after=float(loss(solution)),scope='Penalty is explicit sum of component energies. Exact one-reader conditional solve; neither a proximal damping reinterpretation nor a native/global convergence result.')
    with (P/'PENALIZED_PRODUCT_ALS_V1_CONTROL.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))
if __name__=='__main__':main()
