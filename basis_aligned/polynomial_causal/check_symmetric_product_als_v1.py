"""Independent dense tensor/gradient/direct-normal-solve controls."""
import json
from pathlib import Path
import torch
from symmetric_product_als_v1 import normal_operator,native_rhs,block_precondition,pcg
P=Path(__file__).resolve().parent
def main():
    torch.set_num_threads(2);torch.manual_seed(91162149)
    d,q,o,k=5,3,4,7
    l,r=torch.randn(k,d,dtype=torch.float64),torch.randn(k,d,dtype=torch.float64)
    down=torch.randn(o,k,dtype=torch.float64);b=torch.randn(q,d,dtype=torch.float64);w=torch.randn(o,q,dtype=torch.float64)
    root=torch.randn(o,o,dtype=torch.float64);metric=root@root.T+torch.eye(o,dtype=torch.float64);og=w.T@metric@w
    def tensor(left,right,writer):
        t=torch.einsum('ok,ki,kj->oij',writer,left,right);return (t+t.transpose(-1,-2))/2
    target=tensor(l,r,down);a=torch.randn(q,d,dtype=torch.float64,requires_grad=True)
    error=tensor(a,b,w)-target;loss=torch.einsum('oij,op,pij->',error,metric,error);loss.backward()
    rhs=native_rhs(l,r,down,b,w,metric);gradient=2*(normal_operator(a.detach(),b,og)-rhs)
    gradient_error=float((gradient-a.grad).abs().max());assert gradient_error<=1e-10
    columns=[]
    for e in torch.eye(q*d,dtype=torch.float64):columns.append(normal_operator(e.reshape(q,d),b,og).flatten())
    dense=torch.stack(columns,1);symmetry=float((dense-dense.T).abs().max());assert symmetry<=1e-12
    exact=torch.linalg.solve(dense,rhs.flatten()).reshape(q,d)
    got,diag=pcg(lambda x:normal_operator(x,b,og),rhs,lambda x:block_precondition(x,b,og),tolerance=1e-11,max_iterations=200)
    solution_error=float((got-exact).norm()/exact.norm());assert diag['converged'] and solution_error<=1e-9
    before=float(loss.detach());after_error=tensor(got,b,w)-target;after=float(torch.einsum('oij,op,pij->',after_error,metric,after_error));assert after<=before
    # Validate each inverse diagonal block, not just overall CG convergence.
    v=torch.randn_like(b);pre=block_precondition(v,b,og);block_error=0.
    for j in range(q):
        block=.5*og[j,j]*(b[j].square().sum()*torch.eye(d,dtype=torch.float64)+torch.outer(b[j],b[j]))
        block_error=max(block_error,float((block@pre[j]-v[j]).abs().max()))
    assert block_error<=1e-10
    result=dict(schema='symmetric.product.als.control.v1',passed=True,gradient_max_absolute_error=gradient_error,normal_matrix_symmetry_max_absolute_error=symmetry,pcg=diag,pcg_direct_solution_relative_l2=solution_error,block_preconditioner_replay_max_absolute_error=block_error,dense_loss_before=before,dense_loss_after=after,scope='Exact weight-only one-reader-family least squares, fixed other readers/writers. No data, native model fit, square-factor tying, global optimum or circuit identification claim.')
    with (P/'SYMMETRIC_PRODUCT_ALS_V1_CONTROL.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))
if __name__=='__main__':main()
