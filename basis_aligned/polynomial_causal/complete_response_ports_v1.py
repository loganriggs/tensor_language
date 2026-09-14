"""Complete normalized directional response from fixed sufficient input ports."""
import torch
from contracted_qk_response_v1 import EPS
from composed_key_span_v1 import scalar


def compile_weights(p):
    keys=['read_left','read_direction','gram','left_direction','direction_norm2','gain','key_coordinates','inside_adapters']
    return dict({k:p[k] for k in keys},right_direction=p['right']@p['direction'])


def project_ports(z,h,u,p):
    z=z.double();h=h.double();u=u.double()
    return dict(bz=z@p['right'].T,zd=z@p['direction'],z2=z.square().sum(-1),
                ch=h@p['readers'].T,ath=h@p['left'],hd=h@p['direction'],h2=h.square().sum(-1),
                cu=u@p['readers'].T,atu=u@p['left'],ud=u@p['direction'],u2=u.square().sum(-1),hu=(h*u).sum(-1))


def execute(ports,amplitude,p):
    a=amplitude.double()[...,None];gain=p['gain']
    r=ports['z2'][...,None]/1152+EPS
    rm=(ports['z2'][...,None]-2*a*ports['zd'][...,None]+a.square()*p['direction_norm2'])/1152+EPS
    beta=r/rm-1
    c=-a/rm*(ports['bz']-a/2*p['right_direction'])
    reads=ports['ch']+gain*(-a*p['read_direction']+beta*ports['cu']+c@p['read_left'].T)
    hw=-a*ports['hd'][...,None]+(ports['ath']*c).sum(-1,keepdim=True)
    uw=-a*ports['ud'][...,None]+(ports['atu']*c).sum(-1,keepdim=True)
    ww=a.square()*p['direction_norm2']-2*a*(c*p['left_direction']).sum(-1,keepdim=True)+(c*(c@p['gram'])).sum(-1,keepdim=True)
    norm=ports['h2'][...,None]+2*gain*beta*ports['hu'][...,None]+gain.square()*beta.square()*ports['u2'][...,None]
    norm=norm+2*gain*(hw+gain*beta*uw)+gain.square()*ww
    rho2=norm/1152+EPS
    return scalar(reads,rho2,p),rho2
