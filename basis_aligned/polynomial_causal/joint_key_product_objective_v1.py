"""Implicit symmetric product tensor objective for a shared key-input subspace."""
import torch

def captured(projector,g1,g2):
    return ((projector@g1).trace()*(projector@g2).trace()
            +(projector@g1@projector@g2).trace())/2

def gradient(projector,g1,g2):
    return ((projector@g2).trace()*g1+(projector@g1).trace()*g2
            +g1@projector@g2+g2@projector@g1)/2

def fit(g1,g2,initial,tolerance=1e-7,max_steps=3000):
    """Monotone QR-retracted gradient ascent, with explicit stationarity residual.

    This is a local solver. No global recovery guarantee. Objective and stopping
    residual are normalized by full tensor norm squared.
    """
    total=captured(torch.eye(g1.shape[0],dtype=g1.dtype),g1,g2)
    u=torch.linalg.qr(initial,mode='reduced').Q;history=[]
    for iteration in range(max_steps):
        p=u@u.T;value=captured(p,g1,g2)/total
        grad=2*gradient(p,g1,g2)@u/total
        tangent=grad-u@(u.T@grad);stationarity=float(tangent.norm())
        history.append(float(value))
        if stationarity<=tolerance:break
        step=10.
        for backtrack in range(40):
            candidate=torch.linalg.qr(u+step*tangent,mode='reduced').Q
            improved=captured(candidate@candidate.T,g1,g2)/total
            if improved>=value+1e-4*step*tangent.square().sum():break
            step/=2
        else:break
        u=candidate
    p=u@u.T;grad=2*gradient(p,g1,g2)@u/total;stationarity=float((grad-u@(u.T@grad)).norm())
    return u,dict(capture=float(captured(p,g1,g2)/total),iterations=iteration+1,
                   stationarity=stationarity,converged=stationarity<=tolerance,
                   monotone=all(b>=a-1e-14 for a,b in zip(history,history[1:])))
