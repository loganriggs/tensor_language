"""Coefficient Grams for two-QK routing and complete squared denominators.

The numerator is separately symmetric in query/key variables. Each squared
normalizer is fully symmetric degree4 plus its exact epsilon terms. Matching
the numerator alone is not a normalized-routing equivalence test.
"""
import torch


def biquadratic_inner(a,b,c,d):
    # (q^T A k)(q^T B k), separately symmetrized query and key slots.
    return (((a*c).sum()*(b*d).sum()+(a*d).sum()*(b*c).sum())
            +torch.trace((a@c.T)@(b@d.T))
            +torch.trace((a@d.T)@(b@c.T)))/4


def quartic_inner(a,b,c,d):
    # (x^T A x)(x^T B x), with all four x slots symmetrized.
    return ((a*c).sum()*(b*d).sum()+(a*d).sum()*(b*c).sum()
            +4*torch.trace((a@c)@(b@d)))/6


def normalizer_inner(a,b,c,d,epsilon_a,epsilon_c):
    # P=(x^T A x+e_a)(x^T B x+e_a), including quadratic and constant
    # coefficients. Equal head widths have equal e=head_width*RMS epsilon.
    return (quartic_inner(a,b,c,d)+epsilon_a*epsilon_c*((a+b)*(c+d)).sum()
            +epsilon_a**2*epsilon_c**2)


def dense_biquadratic(a,b):
    t=torch.einsum('ia,jb->ijab',a,b)
    return (t+t.transpose(0,1)+t.transpose(2,3)+t.transpose(0,1).transpose(2,3))/4


def dense_quartic(a,b):
    return (torch.einsum('ij,kl->ijkl',a,b)+torch.einsum('ik,jl->ijkl',a,b)
            +torch.einsum('il,jk->ijkl',a,b)+torch.einsum('ij,kl->ijkl',b,a)
            +torch.einsum('ik,jl->ijkl',b,a)+torch.einsum('il,jk->ijkl',b,a))/6


def controls():
    torch.manual_seed(409);dt=torch.float64
    a,b,c,d=[torch.randn(4,5,dtype=dt) for _ in range(4)]
    expected=(dense_biquadratic(a,b)*dense_biquadratic(c,d)).sum()
    bierror=float(abs(biquadratic_inner(a,b,c,d)-expected))
    matrices=[torch.randn(5,5,dtype=dt) for _ in range(4)]
    a,b,c,d=[(v+v.T)/2 for v in matrices]
    expected=(dense_quartic(a,b)*dense_quartic(c,d)).sum()
    quarticerror=float(abs(quartic_inner(a,b,c,d)-expected))
    ea,ec=.03,.08
    expected+=ea*ec*((a+b)*(c+d)).sum()+ea**2*ec**2
    normalizererror=float(abs(normalizer_inner(a,b,c,d,ea,ec)-expected))
    # Branch swap leaves the full numerator and normalizer polynomial invariant.
    swap=float((dense_quartic(a,b)-dense_quartic(b,a)).abs().max())
    out=dict(biquadratic_dense_absolute_error=bierror,quartic_dense_absolute_error=quarticerror,
             complete_normalizer_absolute_error=normalizererror,branch_swap_error=swap)
    out['passed']=max(out.values())<=1e-10
    return out

if __name__=='__main__':
    import json
    from pathlib import Path
    torch.set_num_threads(2);r=controls();print(json.dumps(r,indent=2))
    Path(__file__).with_name('JOINT_ROUTER_POLYNOMIAL_GRAM_V1_CONTROL.json').write_text(json.dumps(r,indent=2)+'\n');assert r['passed']
