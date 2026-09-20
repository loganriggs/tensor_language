"""Exact mean/remainder interaction through a normalized bilinear map."""
import torch


def decompose(L,R,D,background,mean,remainder,eps):
    """Output bias cancels in the four-corner difference.

    The 'cross' term freezes only the RMS denominator at the both-present corner.
    'normalization' is the exact correction for the other three denominators.
    Inputs may be float64 while eps explicitly matches the native float32 RMS.
    """
    states=[background,background+mean,background+remainder,background+mean+remainder]
    numerators=[((x@L.T)*(x@R.T))@D.T for x in states]
    inverse=[1/(x.square().mean(-1,keepdim=True)+eps) for x in states]
    writes=[p*s for p,s in zip(numerators,inverse)]
    actual=writes[3]-writes[1]-writes[2]+writes[0]
    cross=((mean@L.T)*(remainder@R.T)+(remainder@L.T)*(mean@R.T))@D.T*inverse[3]
    correction=(numerators[1]*(inverse[3]-inverse[1])+numerators[2]*(inverse[3]-inverse[2])
                +numerators[0]*(inverse[0]-inverse[3]))
    return dict(interaction=actual,cross=cross,normalization=correction,writes=writes)


def prepare_head_coordinates(L,R,D,O,background,eps,bias=0.):
    """Conditional rational-quadratic program; charge background-dependent ports.

    Fold the fixed head writer into readers and norm geometry. Reuse the returned
    objects across interventions z in head coordinates. Background projections
    must be recomputed if the background changes; they are not fixed weights.
    """
    return dict(left=L@O,right=R@O,down=D,geometry=O.T@O,
                left_background=background@L.T,right_background=background@R.T,
                norm_background=background.square().sum(-1,keepdim=True),
                overlap_background=background@O,width=background.shape[-1],eps=eps,bias=bias)


def evaluate_head_coordinates(program,z):
    left=program['left_background']+z@program['left'].T
    right=program['right_background']+z@program['right'].T
    numerator=(left*right)@program['down'].T
    squared=program['norm_background']+2*(program['overlap_background']*z).sum(-1,keepdim=True)
    squared=squared+torch.einsum('...i,ij,...j->...',z,program['geometry'],z)[...,None]
    return numerator/(squared/program['width']+program['eps'])+program['bias']
