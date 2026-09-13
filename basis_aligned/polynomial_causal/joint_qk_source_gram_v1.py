"""Exact coefficient Gram for (q^T A s)(q^T B s) F s.

Symmetric degree2 in query and degree3 in source. Gaussian fourth moments
are an algebraic tool only: subtract query trace to get coefficient Frobenius.
"""
import torch


def gram(a,b,f):
    sa=a@a.T;sb=b@b.T
    sab=a@b.T;sab=(sab+sab.T)/2
    ca=f@a.T;cb=f@b.T
    ff=f@f.T
    scalar=sa.trace()*sb.trace()+2*(sa*sb).sum()+sab.trace().square()+2*sab.square().sum()
    eaa=sa.trace()*(cb@cb.T)+2*cb@sa@cb.T
    ebb=sb.trace()*(ca@ca.T)+2*ca@sb@ca.T
    cross=sab.trace()*(ca@cb.T)+2*ca@sab@cb.T
    gaussian=(scalar*ff+eaa+ebb+cross+cross.T)/6
    q=a.T@b;q=(q+q.T)/2
    fq=f@q
    query_trace=(q.square().sum()*ff+2*fq@fq.T)/3
    coefficient=(gaussian-query_trace)/2
    return (coefficient+coefficient.T)/2,dict(gaussian=gaussian,query_trace=query_trace)
