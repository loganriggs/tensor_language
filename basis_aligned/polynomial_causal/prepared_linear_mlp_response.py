"""Conditional linear numerator with exact RMS denominator; no quadratic output core."""
import torch

def compile(left,right,down,h,w,eps):
    d,r=w.shape[-2:];a=left@w;c=right@w
    lh=h@left.T;rh=h@right.T;s0=h.square().mean(-1,keepdim=True)+eps
    base=((lh*rh)@down.T)/s0
    norm_linear=2*torch.einsum('bd,bdr->br',h,w)/d
    linear=down@(a*rh[:,:,None]+c*lh[:,:,None])-base[:,:,None]*norm_linear[:,None]
    idx=torch.triu_indices(r,r,device=w.device);gram=w.transpose(1,2)@w/d
    norm_quad=gram[:,idx[0],idx[1]]*torch.where(idx[0]==idx[1],1.,2.)
    return dict(linear=linear,carry=w,norm=torch.cat([norm_linear,norm_quad],-1),s0=s0)

def evaluate(g,z):
    r=z.shape[-1];i,j=torch.triu_indices(r,r,device=z.device)
    phi=torch.cat([z,z[:,i]*z[:,j]],-1);s=g['s0']+(g['norm']*phi).sum(-1,keepdim=True)
    if not bool((s>0).all()):raise ValueError('Nonpositive RMS denominator')
    return torch.einsum('bdr,br->bd',g['carry'],z)+torch.einsum('bdr,br->bd',g['linear'],z)/s
