"""Four-mode MLP8 -> head9 value response using pristine context only.
Predicts change, not absolute value; baseline head9 RMS denominator held fixed.
"""
import torch
EPS=torch.finfo(torch.float32).eps

def compile_response(generator,writer):
 readers=generator['eigenvectors'][:,:4].T.clone();d=writer.double().clone()
 return dict(readers=readers,writer=d,eigenvalues=generator['eigenvalues'][:4].clone(),reader_shifts=readers@d,read_dot_writer=generator['reader']@d,writer_norm2=d.square().sum(),downstream_gain=generator['lambdas'][0].clone())

def execute(z,amplitude,baseline_norm9,p):
 """z[...,1152]; amplitude/norm9 match leading shape. Edit is z-amplitude*writer. No changed-state input."""
 z=z.double();a=amplitude.double();rho9=baseline_norm9.double();reads=z@p['readers'].T;dot=z@p['writer'];width=z.shape[-1];rho0=z.square().mean(-1)+EPS
 rho1=rho0+(-2*a*dot+a.square()*p['writer_norm2'])/width
 changed=reads-a[...,None]*p['reader_shifts'];quad0=(reads.square()*p['eigenvalues']).sum(-1)/rho0;quad1=(changed.square()*p['eigenvalues']).sum(-1)/rho1
 return p['downstream_gain']*(-a*p['read_dot_writer']+quad1-quad0)/rho9
