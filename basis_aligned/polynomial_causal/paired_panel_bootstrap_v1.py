"""Shared paired-group resampling for intervention receipt statistics."""
import numpy as np


class PairedPanelBootstrap:
    """One index draw reused across arms within a panel; no population inference."""
    def __init__(self,n,seed,draws=4000):
        if n<2 or draws<1:raise ValueError('Need at least two groups and one draw')
        self.n=n;self.indices=np.random.default_rng(seed).integers(0,n,size=(draws,n))

    def _array(self,values):
        a=np.asarray(values,dtype=float)
        if a.shape!=(self.n,) or not np.isfinite(a).all():raise ValueError('Expected one finite scalar per group')
        return a

    def mean(self,values):
        a=self._array(values);return np.quantile(a[self.indices].mean(1),[.025,.975]).tolist()

    def relative_l2(self,error_squared,reference_squared):
        e=self._array(error_squared);r=self._array(reference_squared)
        if (e<0).any() or (r<0).any():raise ValueError('Squared errors must be nonnegative')
        den=r[self.indices].sum(1)
        if not (den>0).all():raise ValueError('Zero bootstrap reference norm')
        return np.quantile(np.sqrt(e[self.indices].sum(1)/den),[.025,.975]).tolist()
