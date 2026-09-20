"""Folded scalar gradients and equal-reader, equal-norm spherical probes."""
import torch

def scalar_value_gradient(factors,x):
 c,l,r,D,A,B=factors
 left=x@A.T;right=x@B.T;p=left*right;z=p@D.T
 lz=z@l.T;rz=z@r.T;value=((lz*rz)*c).sum(-1)
 gz=(rz*c)@l+(lz*c)@r;gp=gz@D
 gradient=(gp*right)@A+(gp*left)@B
 return value,gradient

def invisible_tangent(x,vector,frame):
 xn=x-(x@frame)@frame.T
 v=vector-(vector@frame)@frame.T
 v=v-xn*((v*xn).sum(-1,keepdim=True)/xn.square().sum(-1,keepdim=True))
 return v,xn

def spherical_pair(x,direction,frame,angle):
 v,xn=invisible_tangent(x,direction,frame)
 radius=xn.norm(dim=-1,keepdim=True);unit=v/v.norm(dim=-1,keepdim=True)
 parallel=x-xn
 center=parallel+torch.cos(x.new_tensor(angle))*xn
 shift=torch.sin(x.new_tensor(angle))*radius*unit
 return center+shift,center-shift,unit,radius

def toy_check():
 gen=torch.Generator().manual_seed(260952)
 rand=lambda *shape:torch.randn(*shape,generator=gen,dtype=torch.float64)
 factors=[rand(6),rand(6,4),rand(6,4),rand(4,5),rand(5,7),rand(5,7)]
 with torch.enable_grad():
  x=rand(16,7).requires_grad_();value,gradient=scalar_value_gradient(factors,x);ref=torch.autograd.grad(value.sum(),x)[0]
  error=float(((ref-gradient).norm()/ref.norm()).detach());assert error<1e-12
 x=x.detach();frame=torch.linalg.qr(rand(7,2)).Q;plus,minus,unit,radius=spherical_pair(x,gradient.detach(),frame,.001)
 reader_error=float(((plus-minus)@frame).norm()/(x@frame).norm());norm_error=float((plus.norm(dim=1)-x.norm(dim=1)).abs().max()/x.norm(dim=1).max());assert max(reader_error,norm_error)<1e-12
 fd=(scalar_value_gradient(factors,plus)[0]-scalar_value_gradient(factors,minus)[0])/(2*torch.sin(torch.tensor(.001,dtype=x.dtype))*radius[:,0]);derivative=(gradient.detach()*unit).sum(1);fd_error=float((fd-derivative).norm()/derivative.norm());assert fd_error<1e-4
 return dict(gradient_relative_error=error,reader_pair_error=reader_error,norm_error=norm_error,finite_difference_relative_error=fd_error)
if __name__=='__main__':
 import json
 from pathlib import Path
 r=toy_check();Path(__file__).with_name('READER_SPHERE_ORACLE_V1.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
