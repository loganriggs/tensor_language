"""Complete even-key numerator replacement error with full reads preserved."""
import torch

def grams(m1,m2,basis):
    """m_i[query_width,key_width], orthonormal basis[key_width,k]."""
    n1=m1@basis;n2=m2@basis;o1=m1-n1@basis.T;o2=m2-n2@basis.T
    g1=n1.T@n1;g2=n2.T@n2;c=n1.T@n2
    z11=o1.T@n1;z12=o1.T@n2;z21=o2.T@n1;z22=o2.T@n2
    h=((o1*o1).sum()*g2+(o2*o2).sum()*g1+z12.T@z12+z21.T@z21
       +(o1*o2).sum()*(c+c.T)+z22.T@z11+z11.T@z22)/4
    return g1,g2,c,h

def loss(p,g1,g2,c,h):
    """Sum over any leading position axis; P must be an orthogonal projector.

    Polynomial extension off the projector manifold is used for gradients.
    """
    d=torch.eye(p.shape[-1],dtype=p.dtype,device=p.device)-p
    tr=lambda x:x.diagonal(dim1=-2,dim2=-1).sum(-1)
    ct=c.transpose(-1,-2)
    inside=(tr(p@g1)*tr(d@g2)+tr(p@g2)*tr(d@g1)
            +tr(p@c@d@ct)+tr(p@ct@d@c)
            +2*tr(p@c)*tr(d@ct)+2*tr(p@g1@d@g2))/4
    return (tr(d@h)+inside).sum()
