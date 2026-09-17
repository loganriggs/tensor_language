"""Complete local head8/MLP8 edit from three declared native-state arrays.
No model loader, repository graph, or fitted coefficients. Native replay pending.
"""
import head8
import torch
import torch.nn.functional as F

def mlp_response(p,g,delta):
    # Evaluate the response terms in double precision; preserve FP32 RMS epsilon.
    z=g.double();d=delta.double();L=p['left'].double();R=p['right'].double();D=p['down'].double()
    eps=torch.finfo(torch.float32).eps
    s0=z.square().mean(-1,keepdim=True)+eps;s1=(z+d).square().mean(-1,keepdim=True)+eps
    lz=F.linear(z,L);rz=F.linear(z,R);ld=F.linear(d,L);rd=F.linear(d,R)
    norm=(s0/s1-1)*F.linear(lz*rz,D)/s0
    cross=F.linear(ld*rz+lz*rd,D)/s1
    quadratic=F.linear(ld*rd,D)/s1
    return norm+cross+quadratic

def execute(p,current8,donor_city8,post_attention8,recipient_token,donor_token,city,destination,strength=.5):
    delta=strength*head8.execute(p['head8'],current8,donor_city8,recipient_token,donor_token,city,destination)
    return delta.double()+mlp_response(p['mlp8'],post_attention8,delta)
