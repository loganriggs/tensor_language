"""A dense/diagonal<=1e-10 B basis norm<=1e-10 C factor gradients<=1e-9.
Reuses existing quartic symmetrizer from joint-router work, new two-MLP contraction.
"""
import itertools,json
from pathlib import Path
import torch
from composed_quartic_contraction_v1 import contract
from chunked_bilinear_coefficient_v1 import dense
from joint_router_polynomial_gram_v1 import dense_quartic


def tensor(weights,scale):
    l0,r0,d0,l1,r1,d1=weights
    q=dense(l0,r0,d0*scale)
    a=torch.einsum('ma,aij->mij',l1,q);b=torch.einsum('ma,aij->mij',r1,q)
    return sum(d1[:,k,None,None,None,None]*dense_quartic(a[k],b[k]) for k in range(len(a)))


def rel(a,b):return float(((a-b).norm()/b.norm().clamp_min(1e-30)).detach())


def main():
    torch.set_default_dtype(torch.float64);torch.set_num_threads(2);torch.manual_seed(11401)
    out=Path(__file__).with_name('COMPOSED_QUARTIC_CONTRACTION_V1_CONTROL.json');assert not out.exists()
    w=[torch.randn(*shape) for shape in [(7,4),(7,4),(5,7),(9,5),(9,5),(3,9)]];scale=1.0234375
    t=tensor(w,scale);x=torch.randn(17,4,4)
    expected=torch.einsum('oijkl,ni,nj,nk,nl->no',t,x[:,0],x[:,1],x[:,2],x[:,3])
    contraction_error=rel(contract(x,*w,scale),expected)
    v=torch.randn(19,4);l0,r0,d0,l1,r1,d1=w;p=((v@l0.T)*(v@r0.T))@d0.T*scale
    diagonal=((p@l1.T)*(p@r1.T))@d1.T
    diagonal_error=rel(contract(v[:,None,:].expand(-1,4,-1),*w,scale),diagonal)
    indices=torch.tensor(list(itertools.product(range(4),repeat=4)));basis=torch.eye(4)[indices]
    allvalues=contract(basis,*w,scale);norm_error=abs(float(allvalues.square().sum()/t.square().sum()-1))
    candidate=[(a+.1*torch.randn_like(a)).requires_grad_() for a in w]
    proposed=tensor(candidate,scale);dense_loss=(proposed-t).square().sum()/t.square().sum();gd=torch.autograd.grad(dense_loss,candidate)
    values=contract(basis,*candidate,scale);probe_loss=(values-allvalues).square().sum()/allvalues.square().sum();gp=torch.autograd.grad(probe_loss,candidate)
    gradients=[rel(a,b) for a,b in zip(gp,gd)]
    # Deterministic finite sample diagnostic, not an exact-convergence assertion.
    torch.manual_seed(11402);z=torch.randn(8192,4,4);samples=contract(z,*w,scale).square().sum(-1)
    estimate=float(samples.mean());se=float(samples.std(unbiased=True)/len(samples)**.5);exact=float(t.square().sum())
    result=dict(pred_a=max(contraction_error,diagonal_error)<=1e-10,pred_b=norm_error<=1e-10,pred_c=max(gradients)<=1e-9,
        contraction_relative_error=contraction_error,diagonal_relative_error=diagonal_error,basis_norm_relative_error=norm_error,gradient_relative_errors=gradients,
        synthetic_estimator=dict(probes=len(samples),exact_norm2=exact,estimated_norm2=estimate,estimated_standard_error=se,z_score=(estimate-exact)/se),
        scope='Exact four-linear coefficient contraction and gradient oracle for homogeneous two-bilinear composition. Gaussian estimate unbiased in expectation; finite sample uncertainty is estimated, not a rigorous bound. No native result, bias/normalizer omission from complete model allowed, text fitting or circuit claim.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));assert result['pred_a'] and result['pred_b'] and result['pred_c']

if __name__=='__main__':main()
