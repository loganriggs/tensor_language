"""Implicit full-output operator for Q_v Z = Z.T Q_v, with indefinite Q_v.

Nonorthogonal congruence blocks, unlike ordinary orthogonal commutant blocks.
L(Z)=2 sum_v Q_v (Q_v Z-Z.T Q_v); <Z,LZ>=sum_v||Q_v Z-Z.T Q_v||².
No vocabulary tensor or d²-by-d² native operator is materialized.
"""
import json
from pathlib import Path
import torch


def sandwich(left, right, output_gram, matrix):
    """sum_ij K_ij H_i matrix H_j, H_i=sym(l_i r_i.T)."""
    l,r,k,x = left,right,output_gram,matrix
    return (l.T @ (k*(r@x@l.T)) @ r
            + l.T @ (k*(r@x@r.T)) @ l
            + r.T @ (k*(l@x@l.T)) @ r
            + r.T @ (k*(l@x@r.T)) @ l) / 4


class Operator:
    def __init__(self, left, right, output_gram):
        self.left,self.right,self.output_gram = left,right,output_gram
        self.second = sandwich(left,right,output_gram,
                               torch.eye(left.shape[1],dtype=left.dtype,device=left.device))
        self.total = self.second.trace()

    def __call__(self, z):
        return 2*(self.second@z-sandwich(self.left,self.right,self.output_gram,z.T))


def factor_forms(forms):
    values,vectors = torch.linalg.eigh(forms)
    v,d = values.shape
    left = vectors.transpose(1,2).reshape(v*d,d)
    writer = forms.new_zeros(v,v*d)
    for i in range(v):
        writer[i,i*d:(i+1)*d] = values[i]
    return left,left.clone(),writer


def controls():
    torch.manual_seed(951)
    torch.set_default_dtype(torch.float64)
    d, count, sizes = 8,6,[3,3,2]
    original=[]
    for _ in range(count):
        blocks=[]
        for size in sizes:
            x=torch.randn(size,size);x=(x+x.T)/2
            x-=torch.eye(size)*x.trace()/size  # Every nonzero block is indefinite.
            blocks.append(x)
        original.append(torch.block_diag(*blocks))
    original=torch.stack(original)
    a=torch.linalg.qr(torch.randn(d,d)).Q
    b=torch.linalg.qr(torch.randn(d,d)).Q
    mixing=a@torch.diag(torch.linspace(1.,4.,d))@b.T
    forms=mixing.T@original@mixing
    labels=torch.repeat_interleave(torch.arange(3),torch.tensor(sizes))
    witness=torch.linalg.solve(mixing,torch.diag(labels.double())@mixing)
    l,r,w=factor_forms(forms)
    operator=Operator(l,r,w.T@w)
    z=torch.randn(d,d);other=torch.randn(d,d)
    residual=forms@z-z.T@forms
    direct=2*(forms@residual).sum(0)
    action_error=float((operator(z)-direct).norm()/direct.norm())
    energy_error=abs(float((z*operator(z)).sum()/residual.square().sum())-1)
    adjoint_error=abs(float((other*operator(z)).sum()-(z*operator(other)).sum()))/float(direct.norm()*other.norm())
    identity_error=float(operator(torch.eye(d)).norm()/operator.total)
    witness_error=float(operator(witness).norm()/(operator.total*witness.norm()))
    basis=torch.eye(d*d).reshape(d*d,d,d)
    dense=torch.stack([operator(e).flatten() for e in basis],dim=1)
    ev,vec=torch.linalg.eigh((dense+dense.T)/2)
    null=ev.abs()<1e-10*ev.abs().max()
    coefficients=torch.randn(int(null.sum()))
    recovered=(vec[:,null]@coefficients).reshape(d,d)
    eigen=torch.linalg.eigvals(recovered)
    order=sorted(range(d),key=lambda i:(float(eigen[i].real),float(eigen[i].imag.abs())))
    eigen=eigen[order]
    change=torch.zeros_like(recovered)
    recovered_labels=torch.zeros(d,dtype=torch.long)
    # A real invariant block can have a conjugate pair, not just real eigenvalues.
    # Group by conjugacy class and recover its REAL polynomial nullspace.
    for i in range(1,d):
        separation=abs(float(eigen[i].real-eigen[i-1].real))+abs(float(eigen[i].imag.abs()-eigen[i-1].imag.abs()))
        recovered_labels[i]=recovered_labels[i-1]+int(separation>1e-7)
    for group in recovered_labels.unique():
        mask=recovered_labels==group
        real=eigen[mask].real.mean();imag=eigen[mask].imag.abs().mean()
        shifted=recovered-real*torch.eye(d)
        polynomial=shifted if float(imag)<1e-7 else shifted@shifted+imag.square()*torch.eye(d)
        _,_,vh=torch.linalg.svd(polynomial)
        change[:,mask]=vh[-int(mask.sum()):].T
    transformed=change.T@forms@change
    off=recovered_labels[:,None]!=recovered_labels[None,:]
    off_fraction=float(transformed[:,off].square().sum()/transformed.square().sum())
    recovered_sizes=sorted(int((recovered_labels==g).sum()) for g in recovered_labels.unique())
    # Ordinary commutation is an additional restriction and must fail here.
    ordinary=float((forms@witness-witness@forms).norm()/(forms.norm()*witness.norm()))
    free_l,free_r,free_w=torch.randn(13,d),torch.randn(13,d),torch.randn(count,13)
    free_forms=torch.einsum('vi,ijk->vjk',free_w,
        (free_l[:,:,None]*free_r[:,None,:]+free_r[:,:,None]*free_l[:,None,:])/2)
    free_operator=Operator(free_l,free_r,free_w.T@free_w)
    free_direct=2*(free_forms@(free_forms@z-z.T@free_forms)).sum(0)
    free_error=float((free_operator(z)-free_direct).norm()/free_direct.norm())
    dense_null=torch.randn(count,d,d);dense_null=(dense_null+dense_null.transpose(1,2))/2
    null_constraints=torch.stack([(dense_null@e-e.T@dense_null).flatten() for e in basis],1)
    singular=torch.linalg.svdvals(null_constraints)
    dense_nullity=int((singular<1e-10*singular.max()).sum())
    passed=max(action_error,energy_error,adjoint_error,identity_error,witness_error,free_error)<1e-10 and int(null.sum())==4 and recovered_sizes==[2,3,3] and off_fraction<1e-20 and ordinary>1e-3 and dense_nullity==1
    result=dict(instrument_passed=passed,action_relative_error=action_error,energy_relative_error=energy_error,
                adjoint_relative_error=adjoint_error,identity_relative_error=identity_error,
                planted_witness_relative_error=witness_error,null_dimension=int(null.sum()),
                recovered_block_sizes=recovered_sizes,offblock_energy_fraction=off_fraction,
                ordinary_commutator_relative_error=ordinary,mixing_condition=float(torch.linalg.cond(mixing)),
                free_product_action_relative_error=free_error,dense_nullity=dense_nullity,
                complex_eigenvalues=int((eigen.imag.abs()>1e-7).sum()),
                scope='Synthetic indefinite congruence blocks only. Trace-free2x2block has an extra complex symmetry; four null directions represent three real blocks. Native structure untested.')
    Path(__file__).with_name('CONGRUENCE_BLOCK_OPERATOR_V1_CONTROL.json').write_text(json.dumps(result,indent=2)+'\n')
    assert passed,result
    return result


if __name__=='__main__':
    print(json.dumps(controls(),indent=2))
