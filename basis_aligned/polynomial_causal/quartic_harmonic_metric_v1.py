"""Exact Frobenius metric after removing quartic trace components."""
from quartic_repeated_lowrank_v1 import metric as repeated_metric

def metric(k,c,b,n,target_traces):
    d=b.shape[-2]
    beta=(n*n.sum(-1,keepdim=True)+2*n.square())/3
    tr=beta.sum(-1)
    targettr=target_traces.diagonal(dim1=-2,dim2=-1).sum(-1)
    outer=tr[:,None]*tr[None,:];crossouter=tr[:,None]*targettr[None,:]
    wk,wc=repeated_metric(k,c,b,n,target_traces)
    tg=(wk-24*k-9*outer)/72;tc=(wc-24*c-9*crossouter)/72
    factor=3/((d+4)*(d+2))
    return k-6*tg/(d+4)+factor*outer,c-6*tc/(d+4)+factor*crossouter
