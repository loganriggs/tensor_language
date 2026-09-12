"""Conditional scalar interaction executor with explicitly contracted pristine ports."""
import torch
from contracted_qk_response_v1 import scalar,EPS

def compile_weights(p):
 return dict(read_left=p['read_left'],read_direction=p['read_direction'],gram=p['gram'],left_direction=p['left_direction'],right_direction=p['right']@p['direction'],direction_norm2=p['direction_norm2'],gain=p['gain'])

def project_ports(z,h,p):
 z=z.double();h=h.double();return dict(bz=z@p['right'].T,zd=z@p['direction'],z2=z.square().sum(-1),ch=h@p['readers'].T,ath=h@p['left'],hd=h@p['direction'],h2=h.square().sum(-1))

def execute(ports,amplitude,p):
 a=amplitude.double()[...,None];rminus=(ports['z2'][...,None]-2*a*ports['zd'][...,None]+a.square()*p['direction_norm2'])/1152+EPS;c=-a/rminus*(ports['bz']-a/2*p['right_direction']);gain=p['gain'];reads=ports['ch']+gain*(-a*p['read_direction']+c@p['read_left'].T)
 cross=-a*ports['hd'][...,None]+(ports['ath']*c).sum(-1,keepdim=True);ww=a.square()*p['direction_norm2']-2*a*(c*p['left_direction']).sum(-1,keepdim=True)+(c*(c@p['gram'])).sum(-1,keepdim=True);rho2=(ports['h2'][...,None]+2*gain*cross+gain.square()*ww)/1152+EPS
 return scalar(reads,rho2),rho2
