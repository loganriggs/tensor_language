"""Exact two-edit normalized bilinear interaction, split by multiplication/norm."""
import torch
EPS=torch.finfo(torch.float32).eps


def decompose(z,c,r,left,right,down):
    rho=lambda x:x.square().mean(-1,keepdim=True)+EPS
    Lz,Lc,Lr=(x@left.T for x in (z,c,r))
    Rz,Rc,Rr=(x@right.T for x in (z,c,r))
    q0=(Lz*Rz)@down.T
    qc=(Lz*Rc+Lc*Rz+Lc*Rc)@down.T
    qr=(Lz*Rr+Lr*Rz+Lr*Rr)@down.T
    invN,invC,invR,invA=(1/rho(x) for x in (z,z+c,z+r,z+c+r))
    cross=((Lc*Rr+Lr*Rc)@down.T)*invA
    norm=q0*(invA-invC-invR+invN)+qc*(invA-invC)+qr*(invA-invR)
    return dict(cross=cross,normalization=norm,total=cross+norm)
