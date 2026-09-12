"""Fold fixed-writer response into joint QK/value reads; no changed residual output."""
import torch
EPS=torch.finfo(torch.float32).eps

def compile_program(p,low,gain):
 b=p['key_basis'][1];k1=p['k1'][1].double();k2=p['k2'][1].double()
 readers=torch.cat([p['q1'][1].double(),k1,p['q2'][1].double(),k2,(k1@b)@b.T,(k2@b)@b.T,p['current_value_reader'][None]],0)
 left=low['left'].double();d=low['direction'].double()
 return dict(readers=readers,left=left,right=low['right'].double(),direction=d,gain=torch.as_tensor(gain).double(),read_left=readers@left,read_direction=readers@d,gram=left.T@left,left_direction=left.T@d,direction_norm2=d@d)

def features(z,h,u,a,p):
 z=z.double();h=h.double();u=u.double();a=a.double()[...,None];d=p['direction'];left=p['left'];gain=p['gain'];r=z.square().mean(-1,keepdim=True)+EPS;rm=(z-a*d).square().mean(-1,keepdim=True)+EPS;beta=r/rm-1;c=-a/rm*((z-a*d/2)@p['right'].T)
 reads=h@p['readers'].T+gain*(-a*p['read_direction']+beta*(u@p['readers'].T)+c@p['read_left'].T)
 norm2=h.square().sum(-1,keepdim=True)+2*gain*beta*(h*u).sum(-1,keepdim=True)+gain.square()*beta.square()*u.square().sum(-1,keepdim=True)
 hw=-a*(h*d).sum(-1,keepdim=True)+((h@left)*c).sum(-1,keepdim=True);uw=-a*(u*d).sum(-1,keepdim=True)+((u@left)*c).sum(-1,keepdim=True)
 ww=a.square()*p['direction_norm2']-2*a*(c*p['left_direction']).sum(-1,keepdim=True)+(c*(c@p['gram'])).sum(-1,keepdim=True)
 norm2=norm2+2*gain*(hw+gain*beta*uw)+gain.square()*ww
 return reads,norm2/1152+EPS

def scalar(reads,rho2):
 n=reads.shape[-2];inv=1/(10000**(torch.arange(0,128,2,dtype=torch.float32)/128));angle=torch.outer(torch.arange(n,dtype=torch.float32),inv);co=angle.cos().bfloat16().to(reads.device);si=angle.sin().bfloat16().to(reads.device)
 def rot(x):
  a,b=x.chunk(2,-1);return torch.cat((a*co+b*si,-a*si+b*co),-1)
 full=[];ref=[]
 for qi,ki,ii in [(0,128,512),(256,384,640)]:
  q=reads[...,qi:qi+128];k=reads[...,ki:ki+128];inside=reads[...,ii:ii+128];q=rot(q/(q.square().mean(-1,keepdim=True)+EPS*rho2).sqrt());den=(k.square().mean(-1,keepdim=True)+EPS*rho2).sqrt();full.append(q@rot(k/den).transpose(-1,-2)/128);ref.append(q@rot((k-2*inside)/den).transpose(-1,-2)/128)
 gamma=(full[0]*full[1]+ref[0]*ref[1])/2;gamma=gamma.masked_fill(~torch.ones(n,n,dtype=torch.bool,device=reads.device).tril(),0);v=reads[...,-1:]/rho2.sqrt();return (gamma@v)[...,0]
