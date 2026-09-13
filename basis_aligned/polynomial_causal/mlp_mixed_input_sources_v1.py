"""Exact RR/mixed/AA split of a symmetric bilinear input-change product."""
import torch
EPS=torch.finfo(torch.float32).eps


def split(z,c,r,c_residual,r_residual,left,right,down):
    ca,ra=c-c_residual,r-r_residual
    rho=(z+c+r).square().mean(-1,keepdim=True)+EPS
    def cross(x,y):
        return ((x@left.T)*(y@right.T)+(y@left.T)*(x@right.T))@down.T/rho
    rr=cross(c_residual,r_residual)
    mixed=cross(c_residual,ra)+cross(ca,r_residual)
    aa=cross(ca,ra)
    return dict(residual_residual=rr,residual_attention=mixed,attention_attention=aa,total=rr+mixed+aa)
