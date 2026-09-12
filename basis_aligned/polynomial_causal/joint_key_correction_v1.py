"""Exact product and denominator accounting for a fixed pair of native queries.

Projected key vectors already include fixed-position rotary maps. Denominators
must be computed from unrotated projections, preserving rounded-RoPE semantics.
All terms refer to two factors of one routing operation, not separate tasks.
"""
import torch


def corrections(q1,q2,k1,k2,delta1,delta2,den_base,den_full):
    a=torch.einsum('nthd,nhd->nht',q1,k1)
    b=torch.einsum('nthd,nhd->nht',q2,k2)
    da=torch.einsum('nthd,nhd->nht',q1,delta1)
    db=torch.einsum('nthd,nhd->nht',q2,delta2)
    n0=a*b; cross=a*db+b*da; omitted=da*db
    d0=den_base[:,:,None];d=den_full[:,:,None]
    return dict(base=n0/d0,denominator_only=n0/d,numerator_only=(n0+cross+omitted)/d0,
                mixed_true_denominator=(n0+cross)/d,full=(n0+cross+omitted)/d)


def control():
    torch.manual_seed(6131101)
    q1,q2=[torch.randn(2,5,3,4,dtype=torch.float64) for _ in range(2)]
    k1,k2,d1,d2=[torch.randn(2,3,4,dtype=torch.float64) for _ in range(4)]
    den0,den=[torch.rand(2,3,dtype=torch.float64)+.5 for _ in range(2)]
    c=corrections(q1,q2,k1,k2,d1,d2,den0,den)
    direct=torch.einsum('nthd,nhd->nht',q1,k1+d1)*torch.einsum('nthd,nhd->nht',q2,k2+d2)/den[:,:,None]
    error=float((c['full']-direct).norm()/direct.norm());assert error<1e-12
    return dict(relative_error=error)


if __name__=='__main__':
    import json
    from pathlib import Path
    result=control();Path(__file__).with_name('JOINT_KEY_CORRECTION_V1_CONTROL.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
