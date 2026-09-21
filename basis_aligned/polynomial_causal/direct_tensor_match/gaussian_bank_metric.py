"""Exact zero-mean Gaussian functional Gram of shared quadratic pair products.

U,V: [m,k,d], q_i=sum_k (U_ik x)(V_ik x), roots q_i q_j for i<=j.
Apply a covariance square root to U,V before calling for nonidentity laws.
QR compresses only the span of leaf directions, preserving all inner products.
No output targets or sampled activations are required.
"""
import torch
from shared_quadratic_bank import bank_gram,roots


def gram_by_degree(U,V):
    m,k,d=U.shape
    if U.shape!=V.shape:raise ValueError('U,V shapes must match')
    basis=torch.linalg.qr(torch.cat([U.reshape(-1,d),V.reshape(-1,d)]).T,mode='reduced').Q
    u,v=U@basis,V@basis
    Q=torch.einsum('ika,ikb->iab',u,v)
    Q=(Q+Q.transpose(-1,-2))/2
    trace=Q.diagonal(dim1=-2,dim2=-1).sum(-1)
    i,j=roots(m,U.device)
    mean=trace[i]*trace[j]+2*(Q[i]*Q[j]).sum((-1,-2))
    product=Q[i]@Q[j]
    # The degree-two Hermite coefficient is 6 Tr(sym(Q_i tensor Q_j)).
    second=trace[i,None,None]*Q[j]+trace[j,None,None]*Q[i]+2*(product+product.transpose(-1,-2))
    flat=second.flatten(1)
    return dict(degree0=mean[:,None]*mean[None,:],degree2=2*(flat@flat.T),degree4=24*bank_gram(u,v)),dict(mean=mean,span_width=basis.shape[1],basis=basis,second=second)
