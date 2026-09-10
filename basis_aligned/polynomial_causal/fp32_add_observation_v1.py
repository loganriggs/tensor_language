"""Observe a finite FP32 addition without confusing rounding with a skipped edit."""
import numpy as np


def observe(before,total,after):
    arrays=[np.asarray(v) for v in (before,total,after)]
    if any(v.dtype!=np.dtype('float32') for v in arrays):raise ValueError('FP32 required')
    before,total,after=arrays
    if before.shape!=total.shape or before.shape!=after.shape:raise ValueError('Shape mismatch')
    if not all(np.isfinite(v).all() for v in arrays):raise ValueError('Finite operands required')
    exact=before.astype(np.float64)+total.astype(np.float64)
    expected=exact.astype(np.float32)
    if not np.isfinite(expected).all():raise ValueError('Overflow outside domain')
    actual_delta=after.astype(np.float64)-before.astype(np.float64)
    residual=actual_delta-total.astype(np.float64)
    down=np.nextafter(expected,np.float32(-np.inf)).astype(np.float64)
    up=np.nextafter(expected,np.float32(np.inf)).astype(np.float64)
    center=expected.astype(np.float64)
    # Directed half-ULP interval handles powers of two and subnormal spacing.
    lower=(down-center)/2;upper=(up-center)/2
    rounding_error=center-exact
    envelope=bool(np.all(rounding_error>=-upper)&np.all(rounding_error<=-lower))
    return dict(passed=bool(np.array_equal(after.view(np.uint32),expected.view(np.uint32))) and envelope,
        actual_delta=actual_delta,rounding_residual=residual,expected=expected,
        maximum_rounding_residual=float(np.max(np.abs(residual))),
        maximum_poststate_error=float(np.max(np.abs(after.astype(np.float64)-center))),
        rounding_envelope_passed=envelope)


def controls():
    from fractions import Fraction
    before=np.array([1000.,1700.,1024.,-1024.,1.,0.,np.nextafter(np.float32(0),np.float32(1))],dtype=np.float32)
    total=np.array([.00004,-.00005,-.00004,.00004,2**-24,-0.,0.],dtype=np.float32)
    after=np.add(before,total,dtype=np.float32)
    r=observe(before,total,after)
    # Fraction arithmetic independently establishes the nearest adjacent FP32 number.
    exact=[Fraction(float(b))+Fraction(float(d)) for b,d in zip(before,total)]
    nearest=[]
    for x,a in zip(exact,after):
        candidates=[np.nextafter(a,np.float32(-np.inf)),a,np.nextafter(a,np.float32(np.inf))]
        distances=[abs(x-Fraction(float(c))) for c in candidates]
        nearest.append(distances[1]==min(distances))
    corrupted=after.copy();corrupted[0]=np.nextafter(corrupted[0],np.float32(np.inf))
    skipped=before.copy();wrong=np.add(before,-total,dtype=np.float32)
    checks=dict(correct_add=r['passed'],fraction_nearest=all(nearest),
        old_absolute_delta_check_fails=r['maximum_rounding_residual']>1e-5,
        one_ulp_rejected=not observe(before,total,corrupted)['passed'],
        skipped_rejected=not observe(before,total,skipped)['passed'],
        wrong_sign_rejected=not observe(before,total,wrong)['passed'],
        zero_identity=observe(before,np.zeros_like(before),before)['passed'])
    assert all(checks.values()),checks
    return dict(passed=True,checks=checks,maximum_representability_residual=r['maximum_rounding_residual'])
