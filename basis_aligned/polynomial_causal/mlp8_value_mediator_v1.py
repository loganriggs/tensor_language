"""Exact selected MLP8-mediated current-value correction at a declared native interface."""
import torch
import torch.nn.functional as F

def execute(program,z,delta,edited_rho9):
    if z.ndim!=3 or z.shape[0]!=1 or z.shape[-1]!=1152 or delta.shape!=z.shape:
        raise ValueError('Expected batch1 z and delta with shape[1,T,1152]')
    if edited_rho9.shape!=(*z.shape[:2],1) or not bool(torch.isfinite(edited_rho9).all()) or not bool((edited_rho9>0).all()):
        raise ValueError('Expected positive finite edited_rho9[1,T,1]')
    if z.device.type!='cpu' or delta.device.type!='cpu' or edited_rho9.device.type!='cpu':raise ValueError('CPU interface only')
    z,delta,rho=z.double(),delta.double(),edited_rho9.double();eps=torch.finfo(torch.float32).eps
    s0=z.square().mean(-1,keepdim=True)+eps;s1=(z+delta).square().mean(-1,keepdim=True)+eps
    left,right=program['left'].double(),program['right'].double()
    lz,rz=F.linear(z,left),F.linear(z,right);ld,rd=F.linear(delta,left),F.linear(delta,right)
    hidden=(ld*rz+lz*rd+ld*rd)/s1+(lz*rz)*(1/s1-1/s0)
    return (1-program['mixture'].double())*program['lambda9'].double()/rho*F.linear(hidden,program['folded_down'].double())
