"""Complete ten joint-key band-pair paths times two value sectors.
Full native key denominators retained. These are routing-edge contributions,
not independently normalized heads or raw input-state interventions.
"""
import torch
import torch.nn.functional as F
PAIRS=[(i,j) for i in range(4) for j in range(i,4)]

def paths(current,tokens,p,index,bands):
 width=128;n=current.shape[1];dtype=current.dtype
 inv=1/(10000**(torch.arange(0,width,2,dtype=torch.float32)/width));ang=torch.outer(torch.arange(n,dtype=torch.float32),inv)
 co=ang.cos().bfloat16().to(current.device).double();si=ang.sin().bfloat16().to(current.device).double()
 def rot(v):
  a,b=v.chunk(2,-1);return torch.cat((a*co+b*si,-a*si+b*co),-1)
 queries=[];scores=[]
 for qname,kname in [('q1','k1'),('q2','k2')]:
  # Query uses actual native FP32 projection, normalization and rotary rounding arithmetic.
  qr=F.rms_norm(F.linear(current,p[qname][index]),(128,),eps=torch.finfo(torch.float32).eps)
  qa,qb=qr.chunk(2,-1);q=torch.cat((qa*co.to(dtype)+qb*si.to(dtype),-qa*si.to(dtype)+qb*co.to(dtype)),-1).double()
  raw=F.linear(current,p[kname][index]);denom=(raw.square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps).sqrt().double()
  band_scores=[]
  for b in bands:
   numerator=(current.double()@b)@(p[kname][index].double()@b).T
   key=rot(numerator/denom);band_scores.append(q@key.transpose(-1,-2)/width)
  scores.append(band_scores)
 values=torch.stack((current.double()@p['current_value_readers'][index],p['first_token_values'][tokens,index]),-1)
 mask=torch.ones(n,n,dtype=torch.bool,device=current.device).tril();pieces=[]
 for i,j in PAIRS:
  gamma=scores[0][i]*scores[1][j]
  if i!=j:gamma=gamma+scores[0][j]*scores[1][i]
  pieces.append(gamma.masked_fill(~mask,0)@values)
 return torch.stack(pieces,-2)
