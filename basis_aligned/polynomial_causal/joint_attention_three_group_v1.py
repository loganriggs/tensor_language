"""Three-contraction fixed approximation to the finite mixed attention output."""
import torch

def execute(native,child,remainder,additive):
    a,b,v=native;ac,bc,vc=(x-y for x,y in zip(child,native));ar,br,vr=(x-y for x,y in zip(remainder,native))
    abar=a+ac+ar;bbar=b+bc+br;vbar=v+vc+vr
    ea=additive[0]-abar;eb=additive[1]-bbar
    wc=ar*(b+bc)+(a+ac)*br+ar*br
    wr=ac*(b+br)+(a+ar)*bc+ac*bc
    cross=torch.einsum('...s,...sd->...d',wc,vc)+torch.einsum('...s,...sd->...d',wr,vr)
    inherited=torch.einsum('...s,...sd->...d',ea*bbar+abar*eb,vbar)
    return cross+inherited
