"""Double-symmetric query/key product coefficient norm, implicit 64x64 Grams."""
import torch

def captured(p,g1,g2,c):
    trace=lambda x:x.diagonal(dim1=-2,dim2=-1).sum(-1)
    return (trace(p@g1)*trace(p@g2)+trace(p@g1@p@g2)
            +trace(p@c.transpose(-1,-2)@p@c)+trace(p@c).square()).sum()/4

def gradient(p,g1,g2,c):
    trace=lambda x:x.diagonal(dim1=-2,dim2=-1).sum(-1)
    ct=c.transpose(-1,-2)
    return (trace(p@g2)[...,None,None]*g1+trace(p@g1)[...,None,None]*g2
            +g1@p@g2+g2@p@g1+ct@p@c+c@p@ct
            +trace(p@c)[...,None,None]*(c+ct)).sum(0)/4

def rotation(position):
    angle=position/(10000**(torch.arange(0,128,2,dtype=torch.float32)/128))
    co=angle.cos().bfloat16().double();si=angle.sin().bfloat16().double()
    return torch.cat([torch.cat([torch.diag(co),torch.diag(si)],1),
                      torch.cat([-torch.diag(si),torch.diag(co)],1)],0)

def fit(g1,g2,c,initial,tolerance=1e-7,max_steps=3000):
    """Monotone QR-retracted gradient ascent, with explicit stationarity residual.

    This is a local solver. No global recovery guarantee. Objective and stopping
    residual are normalized by full tensor norm squared.
    """
    total=captured(torch.eye(g1.shape[-1],dtype=g1.dtype),g1,g2,c)
    u=torch.linalg.qr(initial,mode='reduced').Q;history=[]
    for iteration in range(max_steps):
        p=u@u.T;value=captured(p,g1,g2,c)/total
        grad=2*gradient(p,g1,g2,c)@u/total
        tangent=grad-u@(u.T@grad);stationarity=float(tangent.norm())
        history.append(float(value))
        if stationarity<=tolerance:break
        step=10.
        for backtrack in range(40):
            candidate=torch.linalg.qr(u+step*tangent,mode='reduced').Q
            improved=captured(candidate@candidate.T,g1,g2,c)/total
            if improved>=value+1e-4*step*tangent.square().sum():break
            step/=2
        else:break
        u=candidate
    p=u@u.T;grad=2*gradient(p,g1,g2,c)@u/total;stationarity=float((grad-u@(u.T@grad)).norm())
    return u,dict(capture=float(captured(p,g1,g2,c)/total),iterations=iteration+1,
                   stationarity=stationarity,converged=stationarity<=tolerance,
                   monotone=all(b>=a-1e-14 for a,b in zip(history,history[1:])))
