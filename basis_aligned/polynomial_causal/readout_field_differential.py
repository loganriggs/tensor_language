"""Analytic shared-normalizer and softcap contraction of field differentials."""
def contract(fields,differentials):
    # fields[B,2*outputs+1], differentials[B,2*outputs+1,K].
    import torch
    n=fields[:,:-1];q=fields[:,-1:]
    if not bool((q>0).all()):raise ValueError('Nonpositive squared RMS field')
    s=q.sqrt();score=n/s
    local=1-torch.tanh(score/30).square()
    dv=differentials[:,:-1]/s[:,:,None]-n[:,:,None]*differentials[:,-1:]/(2*q*s)[:,:,None]
    token=local[:,:,None]*dv
    paired=token.reshape(len(fields),-1,2,differentials.shape[-1])
    return paired[:,:,0]-paired[:,:,1]
