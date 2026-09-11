"""A coefficient/adjoint/diagonal/gauge identities<=1e-10.
B x0^2*x1^2 has rank3 canonical Gram but rank1 alternative; native hierarchy inference guard.
"""
import json
from pathlib import Path
import torch
from quartic_gram_map_v1 import layout,coefficients,canonical


def relative(a,b):return float((a-b).norm()/b.norm().clamp_min(1e-30))


def main():
    torch.set_num_threads(2);torch.set_default_dtype(torch.float64);torch.manual_seed(11530)
    out=Path(__file__).with_name('QUARTIC_GRAM_MAP_V1_CONTROL.json');assert not out.exists();errors=[]
    for rank in [2,3,4]:
        m=layout(rank);n=m['pairs'].shape[1];a=torch.randn(5,n,n);a=(a+a.transpose(-1,-2))/2
        c=coefficients(a,m);back=canonical(c,m);errors.append(relative(coefficients(back,m),c))
        errors.append(relative(back.square().sum((-1,-2)),c.square().sum(-1)))
        test=torch.randn_like(c);errors.append(abs(float((a*canonical(test,m)).sum()-(c*test).sum()))/max(1,float((c*test).sum().abs())))
        x=torch.randn(17,rank);pairs=m['pairs'];v=x[:,pairs[0]]*x[:,pairs[1]]*torch.where(pairs[0]==pairs[1],1.,2.**.5)
        first=torch.einsum('bi,oij,bj->bo',v,a,v)
        scalar=x[:,m['terms']].prod(1)*m['multiplicity_root'];second=scalar@c.T
        errors.append(relative(first,second));null=a-back
        errors.append(float(coefficients(null,m).norm()/c.norm()))
        assert back.norm()<=a.norm()+1e-10
    m=layout(2);alt=torch.zeros(3,3);alt[1,1]=.5;c=coefficients(alt,m);base=canonical(c,m)
    x=torch.randn(100,2);scalar=(x[:,m['terms']].prod(1)*m['multiplicity_root'])@c
    errors.append(relative(scalar,x[:,0].square()*x[:,1].square()))
    result=dict(pred_a=max(errors)<=1e-10,pred_b=int(torch.linalg.matrix_rank(base))==3 and int(torch.linalg.matrix_rank(alt))==1,
        maximum_identity_error=max(errors),canonical_eigenvalues=torch.linalg.eigvalsh(base).tolist(),alternative_eigenvalues=torch.linalg.eigvalsh(alt).tolist(),
        same_polynomial_coefficient_error=float((coefficients(base,m)-coefficients(alt,m)).norm()),
        rank16_symmetric_gram_dimension=136*137//2,rank16_quartic_dimension=3876,rank16_null_dimension=136*137//2-3876,
        scope='Canonical matricization rank is not a lower bound on quadratic-square representations of a diagonal polynomial. No native Gram optimization.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));assert result['pred_a'] and result['pred_b']


if __name__=='__main__':main()
