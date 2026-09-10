"""Invert actual rounded position maps and transport joint product coordinates.
Cos/sin tables are not assumed orthogonal: inverse divides by cos^2+sin^2.
"""
import torch

def unrotate_vector(y,c,s):
    a,b=y.double().chunk(2,dim=-1);den=c.square()+s.square()
    return torch.cat([(c*a-s*b)/den,(s*a+c*b)/den],-1)

def rotate_feature(f,c,s):
    x=f.reshape(*f.shape[:-1],128,128);a,b=x.chunk(2,dim=-2)
    x=torch.cat([c[..., :,None]*a+s[..., :,None]*b,-s[..., :,None]*a+c[..., :,None]*b],-2)
    a,b=x.chunk(2,dim=-1)
    return torch.cat([c[...,None,:]*a+s[...,None,:]*b,-s[...,None,:]*a+c[...,None,:]*b],-1).flatten(-2)
