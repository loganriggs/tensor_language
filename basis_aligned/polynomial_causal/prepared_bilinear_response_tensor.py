"""Context-prepared joint tensor for an exact normalized bilinear response.

Reuses projected_bilinear_response's identity, avoiding its large background
mixed tensor. Bias cancels. Packed off-diagonals include BOTH input orders.
"""
import torch

def compile(left,right,down,background,basis,eps):
    # h[B,D], basis[B,D,R]; output dimension must equal D for residual carry.
    b,d,r=basis.shape;idx=torch.triu_indices(r,r,device=basis.device)
    a=torch.einsum('md,bdr->bmr',left,basis);c=torch.einsum('md,bdr->bmr',right,basis)
    lh=background@left.T;rh=background@right.T
    s0=background.square().mean(-1,keepdim=True)+eps
    base=(lh*rh)@down.T/s0
    norm_linear=2*torch.einsum('bd,bdr->br',background,basis)/d
    gram=basis.transpose(1,2)@basis/d
    scale=torch.where(idx[0]==idx[1],1.,2.)
    norm_quad=gram[:,idx[0],idx[1]]*scale
    linear=torch.einsum('om,bmr->bor',down,a*rh[:,:,None]+c*lh[:,:,None])
    products=a[:,:,idx[0]]*c[:,:,idx[1]]
    products+=a[:,:,idx[1]]*c[:,:,idx[0]]*(idx[0]!=idx[1])
    quadratic=torch.einsum('om,bmf->bof',down,products)
    norm=torch.cat([norm_linear,norm_quad],-1)
    coeff=torch.cat([linear,quadratic],-1)-base[:,:,None]*norm[:,None]
    return dict(coefficients=coeff,carry=basis,norm=norm,s0=s0)

def evaluate(program,z):
    r=z.shape[-1];idx=torch.triu_indices(r,r,device=z.device)
    phi=torch.cat([z,z[:,idx[0]]*z[:,idx[1]]],-1)
    s=program['s0']+(program['norm']*phi).sum(-1,keepdim=True)
    if not bool((s>0).all()):raise ValueError('Nonpositive RMS denominator')
    return torch.einsum('bdr,br->bd',program['carry'],z)+torch.einsum('bof,bf->bo',program['coefficients'],phi)/s
