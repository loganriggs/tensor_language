"""Reflection-even joint-key numerator, retaining original normalizers and values."""
import torch
import torch.nn.functional as F

def even_sectors(current,tokens,p,index,basis):
 n=current.shape[1];inv=1/(10000**(torch.arange(0,128,2,dtype=torch.float32)/128));angle=torch.outer(torch.arange(n,dtype=torch.float32),inv);co=angle.cos().bfloat16().to(current.device);si=angle.sin().bfloat16().to(current.device)
 def rot(x):
  a,b=x.chunk(2,-1);return torch.cat((a*co+b*si,-a*si+b*co),-1)
 full=[];reflected=[]
 for qn,kn in (('q1','k1'),('q2','k2')):
  q=rot(F.rms_norm(F.linear(current,p[qn][index]),(128,),eps=torch.finfo(torch.float32).eps)).double()
  native_key=F.linear(current,p[kn][index]);den=(native_key.square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps).sqrt().double()
  num=current.double()@p[kn][index].double().T;inside=(current.double()@basis)@(p[kn][index].double()@basis).T
  full.append(q@rot(num/den).transpose(-1,-2)/128);reflected.append(q@rot((num-2*inside)/den).transpose(-1,-2)/128)
 gamma=(full[0]*full[1]+reflected[0]*reflected[1])/2;gamma=gamma.masked_fill(~torch.ones(n,n,dtype=torch.bool,device=current.device).tril(),0)
 values=torch.stack((current.double()@p['current_value_readers'][index],p['first_token_values'][tokens,index]),-1)
 return gamma@values
