"""Signed symmetric LL1 full-tensor objective; C output, A input, S signed values.

F(x)=sum_g C_g sum_k S_gk (A_gk.x)^2. Input/output rows are normalized
only by Objective, not by the raw analytic kernel. No shared input orthogonality.
"""
import torch
from chunked_bilinear_coefficient_v1 import value_gradient as cp_gradient


def cp(a,s,c):
    m,r,d=a.shape
    x=a.reshape(m*r,d)
    w=(c[:,:,None]*s[:,None,:]).permute(1,0,2).reshape(c.shape[1],m*r)
    return x,x,w


@torch.no_grad()
def value_gradient(target,a,s,c,total,penalty=0.,chunk=256):
    m,r,d=a.shape
    loss,(ga,gb,gw),details=cp_gradient(*target,*cp(a,s,c),total,0.,chunk)
    ga=(ga+gb).reshape_as(a);gw=gw.reshape(c.shape[1],m,r).permute(1,0,2)
    gs=(gw*c[:,:,None]).sum(1);gc=(gw*s[:,None,:]).sum(2)
    gram=a@a.transpose(1,2);gram2=gram.square()
    h=(gram2@s[:,:,None]).squeeze(-1);qnorm=(s*h).sum(1);cnorm=c.square().sum(1)
    scale=penalty/total
    ga+=4*scale*cnorm[:,None,None]*((s[:,:,None]*s[:,None,:]*gram)@a)
    gs+=2*scale*cnorm[:,None]*h
    gc+=2*scale*qnorm[:,None]*c
    energy=cnorm*qnorm
    details['group_energy']=float(energy.sum()/total)
    details['group_energy_by_group']=(energy/total).tolist()
    return loss+penalty*energy.sum()/total,(ga,gs,gc),details


class Objective:
    def __init__(self,target,parts,total,penalty):
        self.target,self.total,self.penalty=target,total,penalty
        self.shapes=[x.shape for x in parts];self.sizes=[x.numel() for x in parts]
        self.device=parts[0].device
        a,s,c=parts;an=a.norm(dim=-1,keepdim=True);cn=c.norm(dim=-1,keepdim=True)
        parts=(a/an,s*an.squeeze(-1).square()*cn,c/cn)
        self.scales=[float(x.norm()) for x in parts]
        self.initial=torch.cat([(x/scale).flatten().cpu() for x,scale in zip(parts,self.scales)]).numpy()

    def unpack(self,point):
        values=torch.as_tensor(point,device=self.device,dtype=torch.float64).split(self.sizes)
        return tuple(v.reshape(shape)*scale for v,shape,scale in zip(values,self.shapes,self.scales))

    def physical(self,point):
        a,s,c=self.unpack(point)
        return a/a.norm(dim=-1,keepdim=True),s,c/c.norm(dim=-1,keepdim=True)

    def evaluate(self,point):
        a,s,c=self.unpack(point);an=a.norm(dim=-1,keepdim=True);cn=c.norm(dim=-1,keepdim=True)
        aa,cc=a/an,c/cn
        loss,(ga,gs,gc),details=value_gradient(self.target,aa,s,cc,self.total,self.penalty)
        ga=(ga-aa*(aa*ga).sum(-1,keepdim=True))/an
        gc=(gc-cc*(cc*gc).sum(-1,keepdim=True))/cn
        gradient=torch.cat([(g*scale).flatten().cpu() for g,scale in zip((ga,gs,gc),self.scales)]).numpy()
        self.last=details
        return float(loss),gradient
