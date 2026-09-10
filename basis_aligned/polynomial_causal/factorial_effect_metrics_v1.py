"""Shared signed-cube effect metrics with explicit target and residual terms."""
import numpy as np
from factorial_semantic_support_v1 import signed_coefficients


def measure(corners, native, changed, *, factors=(2,4), prediction=None):
    native=np.asarray(native,dtype=float);changed=np.asarray(changed,dtype=float)
    cn=signed_coefficients(corners,native)
    effect=native-changed;ce=signed_coefficients(corners,effect)
    n=len(corners[0])
    if not factors or len(set(factors))!=len(factors) or any(i<0 or i>=n for i in factors):
        raise ValueError('distinct valid target factors required')
    required=sum(1<<i for i in factors)
    selected=np.array([mask & required == required for mask in range(2**n)])
    target2=float(cn[selected]@cn[selected]);effect2=float(ce[selected]@ce[selected])
    if min(target2,effect2)<=0:raise ValueError('live natural and intervention target required')
    spill2=float(ce[~selected]@ce[~selected])
    parseval_error=abs(float(effect@effect/len(effect))-float(ce@ce))
    result={'live_natural_mixed_projection':float(ce[selected]@cn[selected]/target2),
            'nonmixed_to_mixed_margin_ratio':float(np.sqrt(spill2/effect2)),
            'target_effect_rms':float(np.sqrt(effect2)),
            'remaining_natural_mixed_ratio':float(np.linalg.norm(cn[selected]-ce[selected])/np.sqrt(target2)),
            'effect_coefficients':ce.tolist(),'target_masks':np.flatnonzero(selected).tolist(),
            'spill_masks':np.flatnonzero(~selected).tolist(),'parseval_error':parseval_error}
    if prediction is not None:
        cp=signed_coefficients(corners,native-np.asarray(prediction,dtype=float))
        result['direct_margin_relative_error']=float(np.linalg.norm(cp[selected]-ce[selected])/np.sqrt(effect2))
    return result
