"""Reflection-even joint-key numerator, retaining original normalizers and values."""
import torch
import torch.nn.functional as F

def scalar(current,tokens,p,index):
 basis=p['key_basis'][index]
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
 value=p['first_token_value'][tokens] if index==0 else current.double()@p['current_value_reader']
 return (gamma@value[...,None])[...,0]


def execute(current8,current9,tokens,p):
 """Return physical writes for supplied layer8/layer9 normalized contexts.
 These contexts must correspond to the actual sequential intervention state.
 """
 return tuple(scalar(x,tokens,p,i)[...,None]*p['writers'][i] for i,x in enumerate((current8,current9)))


def load_program(path, device="cpu"):
 """Load the packed conditional component; decode on CPU for reference rounding.
 Native normalized contexts must still be supplied at each actual intervention state.
 """
 packed=torch.load(path,weights_only=True,map_location="cpu")
 if packed['version']!=1:raise ValueError("Unsupported program version")
 item=packed['basis9'];shape=tuple(item['shape'])
 if shape!=(1152,64):raise ValueError("Unexpected source-reader shape")
 bits=((item['mask'][:,None].to(torch.int64)>>torch.arange(8))&1).reshape(-1).bool()
 if bits.numel()!=1152*64 or int(bits.sum())!=item['values'].numel():raise ValueError("Invalid packed reader")
 s=torch.zeros(bits.numel(),dtype=torch.float64);s[bits]=item['values'].double();s=s.reshape(shape)
 q=s@torch.linalg.inv(torch.linalg.cholesky(s.T@s)).T
 p={k:v.to(device) for k,v in packed['weights'].items()}
 p['key_basis']=[packed['basis8'].to(device),q.to(device)]
 return p
