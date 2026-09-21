"""Exact directional derivative of a residual bilinear layer and normalized readout."""
import torch

def rms_pair(x,v,eps):
 s=(x.square().mean(-1,keepdim=True)+eps).sqrt()
 return x/s,v/s-x*(x*v).mean(-1,keepdim=True)/s.pow(3)

def residual_pair(h,v,a,b,writer,bias,linear=None,eps=1.1920928955078125e-7):
 z,dz=rms_pair(h,v,eps);az,bz=z@a.T,z@b.T
 r=h+(az*bz)@writer.T+bias
 dr=v+((dz@a.T)*bz+az*(dz@b.T))@writer.T
 if linear is not None:r=r+z@linear.T;dr=dr+dz@linear.T
 return r,dr

def output_pair(r,dr,U,eps=1.1920928955078125e-7):
 n,dn=rms_pair(r,dr,eps);q=n@U.T;t=torch.tanh(q/30)
 return 30*t,(1-t.square())*(dn@U.T)

def self_test():
 gen=torch.Generator().manual_seed(942);checks=[]
 for seed in range(5):
  d=4+seed;k=6+seed;rand=lambda *shape:torch.randn(*shape,generator=gen,dtype=torch.float64)
  h,v=rand(3,d),rand(3,d);a,b=rand(k,d),rand(k,d);w,bias,U=rand(d,k),rand(d),rand(9,d);linear=rand(d,d) if seed%2 else None;eps=1e-5
  def reference(x):
   z=torch.nn.functional.rms_norm(x,(d,),eps=eps);r=x+((z@a.T)*(z@b.T))@w.T+bias
   if linear is not None:r=r+z@linear.T
   return 30*torch.tanh((torch.nn.functional.rms_norm(r,(d,),eps=eps)@U.T)/30)
  actual,derivative=output_pair(*residual_pair(h,v,a,b,w,bias,linear,eps),U,eps)
  expected,jvp=torch.autograd.functional.jvp(reference,h,v)
  checks.append(dict(value_error=float((actual-expected).norm()/expected.norm()),derivative_error=float((derivative-jvp).norm()/jvp.norm())))
 assert max(max(c.values()) for c in checks)<1e-12
 return checks
if __name__=='__main__':print(self_test())
