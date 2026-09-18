"""Weight-folded six-piece value response; supplied native normalization contexts."""
import torch
import torch.nn.functional as F
NAMES=('direct','left_cross','right_cross','quadratic','mlp8_normalization','block9_normalization')

def execute(program,z,delta,mixed9_native,mixed9_edited):
    z,delta,m0,m1=[x.double() for x in (z,delta,mixed9_native,mixed9_edited)]
    eps=torch.finfo(torch.float32).eps
    s0=z.square().mean(-1,keepdim=True)+eps
    s1=(z+delta).square().mean(-1,keepdim=True)+eps
    rho0=(m0.square().mean(-1,keepdim=True)+eps).sqrt()
    rho1=(m1.square().mean(-1,keepdim=True)+eps).sqrt()
    left,right,fold,value=[program[k].double() for k in ['left','right','folded_down','value_reader']]
    lz,rz=F.linear(z,left),F.linear(z,right)
    ld,rd=F.linear(delta,left),F.linear(delta,right)
    scale=(1-program['mixture'].double())*program['lambda9'].double()/rho1
    direct=scale*F.linear(delta,value)
    cross_l=scale*F.linear(ld*rz/s1,fold)
    cross_r=scale*F.linear(lz*rd/s1,fold)
    quadratic=scale*F.linear(ld*rd/s1,fold)
    mlp_norm=scale*F.linear(lz*rz*(1/s1-1/s0),fold)
    next_norm=(1-program['mixture'].double())*F.linear(m0,value)*(1/rho1-1/rho0)
    return torch.stack([direct,cross_l,cross_r,quadratic,mlp_norm,next_norm])
