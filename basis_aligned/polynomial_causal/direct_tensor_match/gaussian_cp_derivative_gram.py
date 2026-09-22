"""Exact covariance-weighted derivative Gram for affine quartic CP features.
Factors are expressed in standard-normal coordinates z. Derivatives are in z,
so x=mu+S z corresponds to grad_x(f)^T S S^T grad_x(g).
"""
from functools import lru_cache
import torch


def gram(factors,biases,other=None,other_biases=None):
    if other is None:other=factors;other_biases=biases
    vectors=list(factors)+list(other);means=[b[:,None] for b in biases]+[b[None,:] for b in other_biases];dots={}
    for i in range(8):
        for j in range(i+1,8):
            if i<4<=j:value=vectors[i]@vectors[j].T
            else:
                value=(vectors[i]*vectors[j]).sum(1);value=value[:,None] if j<4 else value[None,:]
            dots[i,j]=value
    @lru_cache(None)
    def moment(mask):
        if mask==0:return factors[0].new_ones((1,1))
        i=(mask & -mask).bit_length()-1;rest=mask^(1<<i);result=means[i]*moment(rest)
        for j in range(i+1,8):
            if rest&(1<<j):result=result+dots[i,j]*moment(rest^(1<<j))
        return result
    result=0.
    for i in range(4):
        for j in range(4,8):result=result+dots[i,j]*moment(255^(1<<i)^(1<<j))
    moment.cache_clear();dots.clear();means.clear()
    return result
