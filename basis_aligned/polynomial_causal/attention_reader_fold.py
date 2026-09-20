"""Explicit five-factor attention adjoint. Readers [B,O,T,D]; no autograd."""
import torch

def rms(x,eps):return x/(x.square().mean(-1,keepdim=True)+eps).sqrt()
def rms_back(x,g,eps):
    s=x.square().mean(-1,keepdim=True)+eps
    return (g-x[:,None]*(g*x[:,None]).sum(-1,keepdim=True)/(x.shape[-1]*s[:,None]))/s[:,None].sqrt()

def fold(x,first,weights,output,lamb,cos,sin,heads,reader,eps):
    b,t,d=x.shape;hd=d//heads;z=rms(x,eps)
    raw=[(z@w.T).reshape(b,t,heads,hd) for w in weights]
    def rotate(a):
        u=rms(a,eps);u1,u2=u.chunk(2,-1)
        return torch.cat((u1*cos+u2*sin,-u1*sin+u2*cos),-1)
    q,k,q2,k2=[rotate(v) for v in raw[:4]]
    value=(1-lamb)*raw[4]+lamb*first.reshape_as(raw[4])
    a=torch.einsum('bthd,bshd->bhts',q,k)/hd
    c=torch.einsum('bthd,bshd->bhts',q2,k2)/hd
    mask=torch.ones(t,t,device=x.device,dtype=torch.bool).tril()
    pat=(a*c).masked_fill(~mask,0)
    gout=(reader@output).reshape(b,reader.shape[1],t,heads,hd)
    gp=torch.einsum('bothd,bshd->bohts',gout,value).masked_fill(~mask,0)
    ga=gp*c[:,None]/hd;gc=gp*a[:,None]/hd
    grads=[torch.einsum('bohts,bshd->bothd',ga,k),torch.einsum('bohts,bthd->boshd',ga,q),torch.einsum('bohts,bshd->bothd',gc,k2),torch.einsum('bohts,bthd->boshd',gc,q2)]
    parts=[]
    for i,g in enumerate(grads):
        g1,g2=g.chunk(2,-1)
        inv=torch.cat((g1*cos-g2*sin,g1*sin+g2*cos),-1)
        back=rms_back(raw[i],inv,eps).flatten(-2)@weights[i]
        parts.append(rms_back(x,back,eps))
    gv=(1-lamb)*torch.einsum('bhts,bothd->boshd',pat,gout)
    parts.append(rms_back(x,gv.flatten(-2)@weights[4],eps))
    return torch.stack(parts), (torch.einsum('bhts,bshd->bthd',pat,value).reshape(b,t,d)@output.T)
