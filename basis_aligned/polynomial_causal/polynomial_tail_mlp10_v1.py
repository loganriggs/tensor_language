"""Same degree-two self approximation, executed as native minus omitted tail."""
import torch

def prepare(context,left,right,down,bias):
    response=context['response']
    beta=response['cross_rms'];gamma=response['writer_rms']
    rho0=response['perpendicular_rms']+gamma*response['parallel'].square()
    scale=(rho0/gamma).sqrt()
    lb=context['basis']@left.T;rb=context['basis']@right.T
    l0,l1,l2=lb.unbind(-2);r0,r1,r2=rb.unbind(-2)
    a1=2*beta*l0+l2/2;b1=2*beta*r0+r2/2
    a2=-gamma*l0;b2=-gamma*r0
    tail=torch.stack((2*(a1*b2+a2*b1)*scale.pow(3),
                      2*a2*b2*scale.pow(4)),-2)@down.T
    return dict(response=response,left=left,right=right,down=down,bias=bias,
                tail=tail,scale=scale)

def evaluate(z,amplitude,prepared):
    p=prepared;r=p['response'];t=amplitude/p['scale']
    rho=r['perpendicular_rms']+r['writer_rms']*(amplitude-r['parallel']).square()
    tail=amplitude.square()/rho.square()*t.pow(3)*(p['tail'][...,0,:]+t*p['tail'][...,1,:])
    numerator=((z@p['left'].T)*(z@p['right'].T))@p['down'].T-.5*tail
    return z+numerator/(z.square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps)+p['bias']
