"""Shared-writer complementary tail; uses the existing fixed-writer cache."""
import torch

def prepare(context,left,right,down,bias,global_state):
    basis=context['basis'];g=global_state
    if not torch.equal(basis[...,0,:],g['writer'].expand_as(basis[...,0,:])):
        raise ValueError('Global cache belongs to a different writer')
    v2=basis[...,2,:].contiguous()
    p02=((v2@left.T)*g['right_writer']+g['left_writer']*(v2@right.T))@down.T
    return dict(response=context['response'],left=left,right=right,down=down,bias=bias,
                private_product=p02,global_state=g)

def omitted(amplitude,prepared):
    r=prepared['response'];gamma=r['writer_rms'];beta=r['cross_rms']
    rho=r['perpendicular_rms']+gamma*(amplitude-r['parallel']).square()
    return amplitude.pow(5)/(2*rho.square())*(-gamma*prepared['private_product']+
        (amplitude*gamma.square()-4*gamma*beta)*prepared['global_state']['self_product'])

def evaluate(z,amplitude,prepared):
    p=prepared
    numerator=((z@p['left'].T)*(z@p['right'].T))@p['down'].T-omitted(amplitude,p)
    return z+numerator/(z.square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps)+p['bias']
