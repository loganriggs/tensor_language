"""Conditional donor transport with explicit recipient background units."""
import json
from pathlib import Path
import torch
from calibration_two_readers_v1 import EPS32
from calibration_scalar_path_v1 import path


def units(background):
    b=background.double();r=b.square().mean(-1).sqrt()
    if not bool(torch.isfinite(r).all()) or not bool((r>0).all()):
        raise ValueError('dimensionless donor interface requires finite nonzero background RMS')
    return r


def read_dimensionless(background,tau,w,U):
    b=background.double();r=units(b);v=b/r[...,None]+tau.double()[...,None]*w.double()
    rho=(v.square().mean(-1)+EPS32/r.square()).sqrt()
    return 30*torch.tanh((v@U.double().T)/(30*rho[...,None]))


def transport(recipient_background,donor_background,donor_q):
    return units(recipient_background)*donor_q.double()/units(donor_background)


def controls():
    g=torch.Generator().manual_seed(9111202)
    rand=lambda *shape:torch.randn(*shape,generator=g,dtype=torch.float64)
    b=rand(9,6);w=rand(6);U=rand(13,6);s=rand(9)
    expected=path(b,s,w,U)[0];actual=read_dimensionless(b,s/units(b),w,U)
    error=float((actual-expected).abs().max());assert error<1e-10
    # Same dimensionless scalar, 8x donor scale: raw interchange changes the
    # recipient, while transporting units restores its original q exactly.
    donor_b=b*8;donor_q=s*8;adjusted=transport(b,donor_b,donor_q)
    compensated=float((path(b,adjusted,w,U)[0]-expected).abs().max())
    raw=float((path(b,donor_q,w,U)[0]-expected).norm())
    assert compensated<1e-10 and raw>.1
    tiny=b*1e-7;tiny_q=s*1e-7;tau=tiny_q/units(tiny)
    exact=read_dimensionless(tiny,tau,w,U)
    v=tiny/units(tiny)[...,None]+tau[...,None]*w
    wrong=30*torch.tanh((v@U.T)/(30*(v.square().mean(-1,keepdim=True)+EPS32).sqrt()))
    epsilon_witness=float((exact-wrong).norm());assert epsilon_witness>.1
    try:units(torch.zeros(2,6))
    except ValueError:zero_rejected=True
    else:zero_rejected=False
    assert zero_rejected
    return dict(passed=True,arbitrary_replacement_max_abs=error,equal_tau_raw_swap_error_norm=raw,equal_tau_adjusted_swap_max_abs=compensated,epsilon_rescaling_error_witness=epsilon_witness,zero_background_rejected=zero_rejected)


if __name__=='__main__':
    out=dict(experiment='calibration_dimensionless_donor_v1',model_forwards=0,trained_model_claim=False,controls=controls(),identity='R=RMS_without_epsilon(g)>0; tau=q/R; z=30tanh(U(g/R+tau*w)/(30sqrt(mean((g/R+tau*w)^2)+eps/R^2)))',native_test_required='Compare fixed raw and scale-adjusted donor swaps, plus FIT mean tau, with independent online edits on both cohorts; no semantic interpretation from algebra alone.')
    f=Path(__file__).with_name('CALIBRATION_DIMENSIONLESS_DONOR_V1_CPU_RESULT.json')
    with f.open('x') as h:json.dump(out,h,indent=2);h.write('\n')
    print(json.dumps(out,indent=2))
