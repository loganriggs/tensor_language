"""Analytic joint-QK source influence with shared key coordinates across positions."""
import json
from pathlib import Path
import torch
from joint_qk_source_influence_v1 import influence


def coefficient(q11,q22,q12,k11,k22,k12):
    """Possibly batched query Grams; S_source=Kstack.T @ coefficient @ Kstack."""
    aa=(q11*k11).sum(dim=(-2,-1));bb=(q22*k22).sum(dim=(-2,-1));ab=(q12*k12).sum(dim=(-2,-1))
    g11=.25*(bb[...,None,None]*q11+q12@k22@q12.transpose(-1,-2))
    g22=.25*(aa[...,None,None]*q22+q12.transpose(-1,-2)@k11@q12)
    g12=.5*(ab[...,None,None]*q12+q11@k12@q22)
    return torch.cat((torch.cat((.5*g11,.25*g12),dim=-1),
                      torch.cat((.25*g12.transpose(-1,-2),.5*g22),dim=-1)),dim=-2)


def control():
    torch.set_num_threads(2);torch.set_default_dtype(torch.float64);torch.manual_seed(1283)
    q1,q2=torch.randn(4,7),torch.randn(4,7);k1,k2=torch.randn(4,9),torch.randn(4,9)
    stack=torch.cat((k1,k2),dim=0);keygrams=(k1@k1.T,k2@k2.T,k1@k2.T)
    # General nonorthogonal rotations deliberately include rounding-like gain.
    rotations=torch.randn(5,4,4)
    first=rotations.transpose(-1,-2)@q1;second=rotations.transpose(-1,-2)@q2
    coefficients=coefficient(first@first.transpose(-1,-2),second@second.transpose(-1,-2),
                             first@second.transpose(-1,-2),*keygrams)
    computed=stack.T@coefficients@stack
    expected=torch.stack([influence(first[i],second[i],k1,k2) for i in range(5)])
    error=float((computed-expected).norm()/expected.norm())
    mean=stack.T@coefficients.mean(0)@stack
    mean_error=float((mean-expected.mean(0)).norm()/expected.mean(0).norm())
    symmetry=float((mean-mean.T).norm()/mean.norm())
    out=dict(instrument_passed=max(error,mean_error,symmetry)<1e-10,
             analytic_autodiff_relative_error=error,position_mean_relative_error=mean_error,
             symmetry_relative_error=symmetry,
             scope='Exact common-key-coordinate influence and averaging across fixed position maps. Enables position-shared source-space selection without materializing per-position full tensors. Native all-position result pending.')
    Path(__file__).with_name('JOINT_QK_POSITION_INFLUENCE_V1_CONTROL.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2));assert out['instrument_passed']


if __name__=='__main__':control()
