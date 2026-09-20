"""Conditional three-source quadratic damage; no torch/model dependency."""
import numpy as np
PAIRS=((0,0),(0,1),(0,2),(1,1),(1,2),(2,2))
def effect(coefficients,amplitudes):
    c=np.asarray(coefficients,dtype=np.float64);a=np.asarray(amplitudes,dtype=np.float64)
    if c.shape[-1]!=9 or a.shape[-1]!=3:raise ValueError('expected nine coefficients and three amplitudes')
    y=-np.sum(c[...,:3]*a,axis=-1)
    for k,(i,j) in enumerate(PAIRS):y=y-c[...,3+k]*a[...,i]*a[...,j]*(.5 if i==j else 1.)
    return y
