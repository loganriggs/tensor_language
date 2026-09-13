"""Two arithmetic rewrites of the retained three-contraction attention predictor."""
import torch


def execute(native,child,remainder,additive,shared_product=False):
    a,b,v=native
    ac,bc,vc=(x-y for x,y in zip(child,native))
    ar,br,vr=(x-y for x,y in zip(remainder,native))
    abar=a+ac+ar;bbar=b+bc+br;vbar=v+vc+vr
    ea=additive[0]-abar;eb=additive[1]-bbar
    if shared_product:
        parent=abar*bbar
        wc=parent-(a+ac)*(b+bc)
        wr=parent-(a+ar)*(b+br)
    else:
        wc=ar*bbar+(a+ac)*br
        wr=ac*bbar+(a+ar)*bc
    cross=torch.einsum('...s,...sd->...d',wc,vc)+torch.einsum('...s,...sd->...d',wr,vr)
    inherited=torch.einsum('...s,...sd->...d',ea*bbar+abar*eb,vbar)
    return cross+inherited
