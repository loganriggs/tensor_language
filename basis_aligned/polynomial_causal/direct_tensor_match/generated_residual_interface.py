"""Selected two-MLP scalar fold with supplied RMS scales and attention/skip reads.
The context carry is kept at the recipient for a source donor interchange.
"""
import torch

def components(reads,carry,scale,alpha,beta):
 return ((carry+.5*reads[...,::2])/scale[...,None]-alpha)*(reads[...,1::2]/scale[...,None]-beta)

def control():
 g=torch.Generator().manual_seed(918)
 reads=torch.randn(2,7,6,generator=g,dtype=torch.float64)
 carry=torch.randn(2,7,3,generator=g,dtype=torch.float64)
 s=torch.rand(2,7,generator=g,dtype=torch.float64)+.1
 alpha=torch.randn(3,generator=g,dtype=torch.float64);beta=torch.randn(3,generator=g,dtype=torch.float64)
 h_read=carry+reads[...,::2]
 old=((h_read-.5*reads[...,::2])/s[...,None]-alpha)*(reads[...,1::2]/s[...,None]-beta)
 replay=float((old-components(reads,carry,s,alpha,beta)).abs().max())
 assert replay<1e-12
 return dict(replay=replay,shape=list(old.shape))
