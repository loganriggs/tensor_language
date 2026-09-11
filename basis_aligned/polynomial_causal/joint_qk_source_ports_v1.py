"""Exact source-port energies of the joint two-QK biquadratic numerator.

Both key slots refer to the same source vector; query/source are distinct.
Numerator terms share the original normalizers. No normalized edit identity.
"""
import json
from pathlib import Path
import torch
from joint_router_polynomial_gram_v1 import dense_biquadratic


def low_rank_norm(q11,q22,q12,k11,k22,k12):
    aa=(q11*k11).sum();bb=(q22*k22).sum();ab=(q12*k12).sum()
    crossed=torch.trace(k11@q12@k22@q12.T)
    other_crossed=torch.trace(k12@q22@k12.T@q11)
    return (aa*bb+ab.square()+crossed+other_crossed)/4


def source_ports(q1,q2,k1,k2,basis):
    """basis has orthonormal columns in the common source-input space."""
    q11,q22,q12=q1@q1.T,q2@q2.T,q1@q2.T
    k11,k22,k12=k1@k1.T,k2@k2.T,k1@k2.T
    p1,p2=k1@basis,k2@basis
    p11,p22,p12=p1@p1.T,p2@p2.T,p1@p2.T
    total=low_rank_norm(q11,q22,q12,k11,k22,k12)
    inside=low_rank_norm(q11,q22,q12,p11,p22,p12)
    outside=low_rank_norm(q11,q22,q12,k11-p11,k22-p22,k12-p12)
    mixed=total-inside-outside
    return dict(total=total,inside=inside,mixed=mixed,outside=outside)


def control():
    torch.set_num_threads(2);torch.set_default_dtype(torch.float64);torch.manual_seed(1201)
    q1,q2=torch.randn(3,5),torch.randn(3,5)
    k1,k2=torch.randn(3,7),torch.randn(3,7)
    basis=torch.linalg.qr(torch.randn(7,2)).Q;projector=basis@basis.T
    a,b=q1.T@k1,q2.T@k2
    full=dense_biquadratic(a,b)
    inside=dense_biquadratic(a@projector,b@projector)
    outside=dense_biquadratic(a@(torch.eye(7)-projector),b@(torch.eye(7)-projector))
    mixed=full-inside-outside
    implicit=source_ports(q1,q2,k1,k2,basis)
    direct=dict(total=full.square().sum(),inside=inside.square().sum(),
                mixed=mixed.square().sum(),outside=outside.square().sum())
    errors={key:float(abs(implicit[key]-direct[key])/direct['total']) for key in direct}
    orth=max(float(abs((x*y).sum())/direct['total']) for x,y in [(inside,mixed),(inside,outside),(mixed,outside)])
    swap=source_ports(q2,q1,k2,k1,basis)
    swap_error=max(float(abs(swap[key]-implicit[key])/direct['total']) for key in direct)
    planted=source_ports(q1,q2,torch.randn(3,2)@basis.T,torch.randn(3,2)@basis.T,basis)
    planted_touch=float((planted['inside']+planted['mixed'])/planted['total'])
    cross=source_ports(q1,q2,torch.randn(3,2)@basis.T,k2@(torch.eye(7)-projector),basis)
    crossing=float(cross['mixed']/cross['total'])
    out=dict(instrument_passed=max(list(errors.values())+[orth,swap_error,abs(1-planted_touch),abs(1-crossing)])<1e-10,
             dense_relative_errors=errors,orthogonality_error=orth,branch_swap_error=swap_error,
             planted_inside_touch_fraction=planted_touch,planted_crossing_fraction=crossing,
             scope='Exact biquadratic coefficient geometry in distinct query/source variables; source slots tied and symmetrized. Full normalized routing not decomposed away.')
    Path(__file__).with_name('JOINT_QK_SOURCE_PORTS_V1_CONTROL.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2));assert out['instrument_passed']


if __name__=='__main__':control()
