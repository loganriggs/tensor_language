"""Joint producer QK/value numerator folded into four downstream source readings."""
import torch
LAYERS=(8,9,13)
def weights(state,readers,device):
 def w(key):return state[key].to(device=device,dtype=torch.float64)
 q1=[];k1=[];q2=[];k2=[];vs=[];outs=[];scales=[];reentry=[]
 for layer in LAYERS:
  prefix=f'transformer.h.{layer}.attn.'
  qq1,kk1,qq2,kk2=[w(prefix+key+'.weight').reshape(9,128,1152) for key in ('c_q','c_k','c_q2','c_k2')]
  mix=float(state[prefix+'lamb']);v=torch.cat([(1-mix)*w(prefix+'c_v.weight'),mix*w('transformer.h.0.attn.c_v.weight')],-1).reshape(9,128,2304)
  scale=1/(qq1.flatten(1).norm(dim=1)*kk1.flatten(1).norm(dim=1)*qq2.flatten(1).norm(dim=1)*kk2.flatten(1).norm(dim=1))
  coefficient=1.
  for j in range(layer+1,18):coefficient*=float(state[f'transformer.h.{j}.lambdas'][0])
  out=(coefficient*readers.to(device)@w(prefix+'c_proj.weight')).reshape(4,9,128)*scale[None,:,None]
  q1.append(qq1);k1.append(kk1);q2.append(qq2);k2.append(kk2);vs.append(v);outs.append(out);scales.append(scale);reentry.append(coefficient)
 return (torch.cat(q1),torch.cat(k1),torch.cat(q2),torch.cat(k2),torch.cat(vs),torch.cat(outs,dim=1)),torch.cat(scales),reentry
