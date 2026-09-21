"""Shared-subset Gaussian moments and profiled mixed native quartic matching."""
from functools import lru_cache
import torch
from quartic_cp import cp_gram, directional
from quartic_cp_profile import profile
from noncentral_gaussian_cp import cross


def gram_dynamic(factors,biases,other,other_biases,return_stats=False):
    # Noncentral Gaussian moment recurrence: a selected factor is either a
    # mean singleton or paired with one remaining factor. Reuse subset moments.
    vectors=list(factors)+list(other)
    means=[v[:,None] for v in biases]+[v[None,:] for v in other_biases]
    dots={}
    for i in range(8):
        for j in range(i+1,8):
            if i<4<=j:value=vectors[i]@vectors[j].T
            else:
                value=(vectors[i]*vectors[j]).sum(1)
                value=value[:,None] if j<4 else value[None,:]
            dots[i,j]=value
    operations=0
    unit=factors[0].new_ones((1,1))
    @lru_cache(None)
    def moment(mask):
        nonlocal operations
        if mask==0:return unit
        i=(mask & -mask).bit_length()-1;rest=mask^(1<<i)
        result=means[i]*moment(rest);operations+=1
        for j in range(i+1,8):
            if rest&(1<<j):
                result=result+dots[i,j]*moment(rest^(1<<j));operations+=1
        return result
    result=moment(255)
    stats=dict(subsets=moment.cache_info().currsize,product_terms=operations)
    moment.cache_clear()
    dots.clear(); means.clear()  # Do not retain tensor graphs in recursive closure cycles.
    return (result,stats) if return_stats else result


def native_mixed_objective(teacher,transformed,location,projection,S,mu,factors,
                           coefficient_weight,ridge=1e-6,envelope=True):
    if coefficient_weight<0:raise ValueError('coefficient weight must be nonnegative')
    fw=[a@S for a in factors];bias=[a@mu for a in factors]
    g0=cp_gram(factors,factors);x0=directional(*teacher,factors).T
    g1=gram_dynamic(fw,bias,fw,bias);x1=cross(transformed,location,projection,fw,bias)
    g=(g1+coefficient_weight*g0)/(1+coefficient_weight)
    x=(x1+coefficient_weight*x0)/(1+coefficient_weight)
    return profile(g,x,ridge=ridge,envelope=envelope)
